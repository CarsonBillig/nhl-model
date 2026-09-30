"""NHL betting lines from ESPN's public feeds (DraftKings / ESPN BET): moneyline, puck line, total, open and close.

    python nhl_odds.py 2024 2025      # cache historical seasons (season = starting year) -> cache/odds_<season>.parquet

ESPN keeps opening + closing prices from the 2024-25 season on. Team codes are normalised to NHL tricodes.
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
SITE = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"
CORE = "https://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl"
ESPN_TO_NHL = {"TB": "TBL", "NJ": "NJD", "SJ": "SJS", "LA": "LAK", "UTAH": "UTA", "WAS": "WSH", "MON": "MTL"}


def tricode(abbr: str) -> str:
    return ESPN_TO_NHL.get(abbr, abbr)


def _get(url, **params):
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return {}


def _num(x):
    if x is None:
        return np.nan
    s = str(x).strip().upper().lstrip("OU")
    if s in ("EVEN", "EV"):
        return 100.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def _prices(side: dict, when: str) -> dict:
    d = side.get(when, {}) or {}
    return {"ml": _num(d.get("moneyLine", {}).get("american")),
            "pl": _num(d.get("pointSpread", {}).get("american")),
            "pl_odds": _num(d.get("spread", {}).get("american"))}


def event_odds(eid: str) -> dict:
    js = _get(f"{CORE}/events/{eid}/competitions/{eid}/odds")
    items = [i for i in js.get("items", []) if "Live" not in i.get("provider", {}).get("name", "")]
    if not items:
        return {}
    it = items[0]
    out = {"book": it.get("provider", {}).get("name")}
    for when in ("open", "close"):
        h, a = _prices(it.get("homeTeamOdds", {}), when), _prices(it.get("awayTeamOdds", {}), when)
        tot = it.get(when, {}).get("total", {}) or {}
        out.update({f"{when}_home_ml": h["ml"], f"{when}_away_ml": a["ml"], f"{when}_home_pl": h["pl"],
                    f"{when}_home_pl_odds": h["pl_odds"], f"{when}_away_pl_odds": a["pl_odds"],
                    f"{when}_total": _num(tot.get("american")),
                    f"{when}_over_odds": _num((it.get(when, {}).get("over") or {}).get("american")),
                    f"{when}_under_odds": _num((it.get(when, {}).get("under") or {}).get("american"))})
    if np.isnan(out["close_total"]) and it.get("overUnder") is not None:
        out["close_total"] = float(it["overUnder"])
    return out


def site_odds(c: dict) -> dict:
    """Current + opening prices from the live scoreboard feed (upcoming games). 'close' = current price."""
    o = (c.get("odds") or [{}])[0]
    if not o.get("moneyline"):
        return {}
    ml, ps, tot = o.get("moneyline", {}), o.get("pointSpread", {}), o.get("total", {})
    out = {"book": o.get("provider", {}).get("name")}
    for when in ("open", "close"):
        g = lambda block, side, k: _num(((block.get(side) or {}).get(when) or {}).get(k))
        out.update({f"{when}_home_ml": g(ml, "home", "odds"), f"{when}_away_ml": g(ml, "away", "odds"),
                    f"{when}_home_pl": g(ps, "home", "line"), f"{when}_home_pl_odds": g(ps, "home", "odds"),
                    f"{when}_away_pl_odds": g(ps, "away", "odds"),
                    f"{when}_total": g(tot, "over", "line"), f"{when}_over_odds": g(tot, "over", "odds"),
                    f"{when}_under_odds": g(tot, "under", "odds")})
    if np.isnan(out["close_total"]) and o.get("overUnder") is not None:
        out["close_total"] = float(o["overUnder"])
    return out


def day_board(d: date, with_odds: bool = True) -> list[dict]:
    js = _get(f"{SITE}/scoreboard", dates=d.strftime("%Y%m%d"))
    rows = []
    for ev in js.get("events", []):
        c = ev["competitions"][0]
        if ev.get("season", {}).get("type") not in (2, None):          # regular season only
            continue
        t = {x["homeAway"]: x for x in c["competitors"]}
        rows.append({"date": pd.Timestamp(ev["date"]).tz_convert("America/New_York").date().isoformat(),
                     "kickoff": ev["date"], "espn_id": ev["id"],
                     "home": tricode(t["home"]["team"]["abbreviation"]), "away": tricode(t["away"]["team"]["abbreviation"]),
                     "home_name": t["home"]["team"].get("displayName"), "away_name": t["away"]["team"].get("displayName"),
                     "home_logo": t["home"]["team"].get("logo"), "away_logo": t["away"]["team"].get("logo"),
                     "home_color": t["home"]["team"].get("color"), "away_color": t["away"]["team"].get("color"),
                     "tv": " / ".join(n for b in c.get("broadcasts", []) for n in b.get("names", [])),
                     **((site_odds(c) or event_odds(ev["id"])) if with_odds else {})})
        time.sleep(0.1)
    return rows


def season_odds(season: int, refresh: bool = False) -> pd.DataFrame:
    f = CACHE / f"odds_{season}.parquet"
    if f.exists() and not refresh:
        return pd.read_parquet(f)
    rows, d = [], date(season, 10, 1)
    while d <= date(season + 1, 4, 20):
        rows += day_board(d)
        d += timedelta(days=1)
    df = pd.DataFrame(rows)
    df.to_parquet(f)
    return df


if __name__ == "__main__":
    for s in map(int, sys.argv[1:] or [2024, 2025]):
        df = season_odds(s, refresh=True)
        print(s, len(df), "games,", int(df["close_home_ml"].notna().sum()), "with closing ML,",
              int(df["open_home_ml"].notna().sum()), "with opening ML", flush=True)
