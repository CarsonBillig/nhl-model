"""Side-by-side NHL team stats for each game, with which team has the edge and each team's league percentile.

Stats are the model's pre-game team ratings (recency-weighted; early in the season last season still counts, fading as
games are played), plus the projected starting goalie and rest. An edge is only called when the gap is at least a
quarter of the league-wide spread for that stat; smaller gaps are "even".
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

EDGE_SD = 0.25

# (group, label, key, higher is better, format)
SPECS = [
    ("Offense", "Goals / game", "gf_pg", True, ".2f"),
    ("Offense", "5v5 expected goals for / 60", "xgf60_5", True, ".2f"),
    ("Offense", "5v5 shot share (Corsi)", "cf_pct5", True, "pct"),
    ("Offense", "Power-play expected goals / 60", "pp_xgf60", True, ".2f"),
    ("Offense", "Finishing (goals vs expected)", "finish", True, "+pct"),
    ("Offense", "Penalties drawn / game", "pen_drawn_pg", True, ".2f"),
    ("Defense", "Goals against / game", "ga_pg", False, ".2f"),
    ("Defense", "5v5 expected goals against / 60", "xga60_5", False, ".2f"),
    ("Defense", "Penalty-kill expected goals against / 60", "pk_xga60", False, ".2f"),
    ("Defense", "Penalties taken / game", "pen_taken_pg", False, ".2f"),
    ("Goalie & schedule", "Starting goalie: saves above expected (per xG faced)", "goalie", True, "+pct"),
    ("Goalie & schedule", "Days of rest", "rest_days", True, ".0f"),
]


def _fmt(v, f):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if f == "pct":
        return f"{v * 100:.1f}%"
    if f == "+pct":
        return f"{v * 100:+.1f}%".replace("-", "−")
    return format(v, f).replace("-", "−")


def _pct_rank(v, values, higher):
    if v is None or values is None or len(values) == 0:
        return None
    p = float(((values < v).mean() + 0.5 * (values == v).mean()) * 100)
    return round(100 - p) if higher is False else round(p)


def build(games: pd.DataFrame, L: pd.DataFrame, goalie_rate: dict, starters: dict) -> dict:
    """{game_id: json string}. `L` = team-game table with pre-game ratings; `goalie_rate` = goalie_id -> rating."""
    cur_season = int(games["season"].max())
    latest = L[L["season"] == cur_season].sort_values("date").groupby("team").tail(1)      # each team's current rating
    dist = {k: latest[k].dropna().to_numpy(float) for _, _, k, _, _ in SPECS if k in latest}
    dist["goalie"] = np.array([v for v in goalie_rate.values() if v is not None and not np.isnan(v)], float)
    dist["rest_days"] = latest["rest_days"].clip(upper=5).dropna().to_numpy(float)
    sd = {k: float(np.std(v)) if len(v) > 5 else 0.0 for k, v in dist.items()}
    idx = L.set_index(["game_id", "team"])
    out = {}
    for _, g in games.iterrows():
        vals = {}
        for side in ("away", "home"):
            key = (g["game_id"], g[side])
            row = idx.loc[key].to_dict() if key in idx.index else {}
            gid = starters.get(key, (None,))[0]
            row["goalie"] = goalie_rate.get(gid) if gid is not None else None
            if row.get("rest_days") is not None and not pd.isna(row.get("rest_days")):
                row["rest_days"] = min(float(row["rest_days"]), 5.0)
            vals[side] = row
        rows = []
        for group, label, key, higher, f in SPECS:
            a, h = vals["away"].get(key), vals["home"].get(key)
            a = None if a is None or pd.isna(a) else float(a)
            h = None if h is None or pd.isna(h) else float(h)
            if a is None and h is None:
                continue
            edge = ""
            if a is not None and h is not None:
                gap = h - a if higher else a - h
                edge = "even" if abs(gap) <= EDGE_SD * sd.get(key, 0.0) else ("home" if gap > 0 else "away")
            rows.append({"group": group, "label": label, "away": _fmt(a, f), "home": _fmt(h, f), "edge": edge,
                         "ap": _pct_rank(a, dist.get(key), higher), "hp": _pct_rank(h, dist.get(key), higher)})
        out[g["game_id"]] = json.dumps(rows)
    return out
