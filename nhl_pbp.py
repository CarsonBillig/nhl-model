"""NHL play-by-play from the NHL's official API (api-web.nhle.com), parsed into shots, penalties and time-on-ice by
situation. Replaces MoneyPuck entirely: expected goals come from our own model (nhl_xg.py).

    python nhl_pbp.py 2020 2026        # download + cache seasons (season = starting year), politely, once

Cache: cache/pbp/events_<season>.parquet (one row per shot attempt or penalty) and toi_<season>.parquet
(seconds per game per situation code). Finished games are fetched once; later runs only fetch new games.
"""
from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import nhl_data

PBP = nhl_data.CACHE / "pbp"
PBP.mkdir(parents=True, exist_ok=True)
URL = "https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play"
SHOTS = {"shot-on-goal": "sog", "missed-shot": "miss", "goal": "goal", "blocked-shot": "block"}
_session = requests.Session()
PACE = 1.0                      # seconds between requests: the API rate-limits (HTTP 429) anything faster
_last = [0.0]


def _get(gid: int):
    """One game's play-by-play, one request at a time; on HTTP 429 wait as long as the API asks, then retry."""
    for attempt in range(8):
        time.sleep(max(0.0, _last[0] + PACE - time.time()))
        _last[0] = time.time()
        try:
            r = _session.get(URL.format(gid=gid), timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                wait = float(r.headers.get("retry-after") or 30)
                time.sleep(wait + 2)
                continue
        except (requests.RequestException, ValueError):
            pass
        time.sleep(3 * (attempt + 1))
    return None


def _secs(p) -> int:
    mm, ss = p["timeInPeriod"].split(":")
    return (p["periodDescriptor"]["number"] - 1) * 1200 + int(mm) * 60 + int(ss)


def parse_game(j: dict) -> tuple[list[dict], list[dict]]:
    gid = int(j["id"])
    season = int(str(j["season"])[:4])
    home_id, away_id = j["homeTeam"]["id"], j["awayTeam"]["id"]
    home, away = nhl_data.code(j["homeTeam"]["abbrev"]), nhl_data.code(j["awayTeam"]["abbrev"])
    names = {p["playerId"]: f'{p["firstName"]["default"]} {p["lastName"]["default"]}' for p in j.get("rosterSpots", [])}
    plays = [p for p in j.get("plays", []) if (p.get("periodDescriptor") or {}).get("periodType") != "SO" and p.get("timeInPeriod")]
    # time on ice by situation code: time between consecutive events in the same period
    dur = defaultdict(float)
    for a, b in zip(plays, plays[1:]):
        if a.get("situationCode") and a["periodDescriptor"]["number"] == b["periodDescriptor"]["number"]:
            dur[a["situationCode"]] += max(_secs(b) - _secs(a), 0)
    toi = [{"game_id": gid, "season": season, "home": home, "away": away, "situation": sc, "seconds": s} for sc, s in dur.items()]

    rows, prev, last_shot = [], None, {}
    for p in plays:
        kind, d = p.get("typeDescKey"), p.get("details") or {}
        t = _secs(p)
        if kind in SHOTS or kind == "penalty":
            sc = str(p.get("situationCode") or "1551").zfill(4)
            owner = d.get("eventOwnerTeamId")
            if kind == "blocked-shot":                      # the blocking (defending) team owns a blocked shot
                shooter_home = owner == away_id
            else:
                shooter_home = owner == home_id
            team, opp = (home, away) if shooter_home else (away, home)
            row = {"game_id": gid, "season": season, "game_type": j.get("gameType"), "period": p["periodDescriptor"]["number"],
                   "t": t, "home": home, "away": away, "team": team, "opp": opp, "is_home": int(shooter_home),
                   "kind": SHOTS.get(kind, "pen"), "situation": sc}
            if kind == "penalty":
                row["pen_minutes"] = d.get("duration")
            else:
                own_sk, opp_sk = (int(sc[2]), int(sc[1])) if shooter_home else (int(sc[1]), int(sc[2]))
                empty = (sc[0] == "0") if shooter_home else (sc[3] == "0")      # defending goalie pulled
                x, y = d.get("xCoord"), d.get("yCoord")
                dist = ang = np.nan
                if x is not None and y is not None:
                    side = p.get("homeTeamDefendingSide")
                    if side in ("left", "right"):
                        home_attacks_x = -89 if side == "right" else 89
                        net_x = home_attacks_x if shooter_home else -home_attacks_x
                    else:
                        net_x = 89 if x >= 0 else -89
                    dx, dy = abs(net_x - x), abs(y)
                    dist, ang = math.hypot(dx, dy), math.degrees(math.atan2(dy, max(dx, 0.1)))
                ls = last_shot.get(team)
                row.update({"own_sk": own_sk, "opp_sk": opp_sk, "empty_net": int(empty), "distance": dist, "angle": ang,
                            "shot_type": d.get("shotType") or "unknown",
                            "rebound": int(ls is not None and 0 <= t - ls <= 3),
                            "rush": int(prev is not None and 0 <= t - prev[0] <= 4 and prev[1] in ("N", "D")
                                        and prev[2] != team),
                            "goalie_id": d.get("goalieInNetId"),
                            "goalie": names.get(d.get("goalieInNetId"))})
                last_shot[team] = t
            rows.append(row)
        # remember the previous event's time / zone / owner (for rush shots)
        owner_team = home if (p.get("details") or {}).get("eventOwnerTeamId") == home_id else away
        prev = (t, (p.get("details") or {}).get("zoneCode"), owner_team)
    return rows, toi


def season(season_: int, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parsed events + situation TOI for every finished regular-season/playoff game of a season (cached)."""
    ef, tf = PBP / f"events_{season_}.parquet", PBP / f"toi_{season_}.parquet"
    ev = pd.read_parquet(ef) if ef.exists() else pd.DataFrame()
    toi = pd.read_parquet(tf) if tf.exists() else pd.DataFrame()
    sched = nhl_data.schedule(season_, refresh=refresh)
    done = sched[sched["state"].isin(["OFF", "FINAL"]) & (sched["game_id"] // 1_000_000 == season_)]["game_id"].astype(int)
    have = set(toi["game_id"].unique()) if not toi.empty else set()
    todo = [g for g in done if g not in have]
    if todo:
        print(f"  NHL play-by-play {season_}-{season_ + 1 - 2000}: fetching {len(todo)} games ...", flush=True)
        new_ev, new_toi, failed = [], [], 0

        def flush():
            nonlocal ev, toi, new_ev, new_toi
            if new_toi:                                        # saved as we go: an interrupted run keeps its progress
                ev = pd.concat([ev, pd.DataFrame(new_ev)], ignore_index=True)
                toi = pd.concat([toi, pd.DataFrame(new_toi)], ignore_index=True)
                ev.to_parquet(ef)
                toi.to_parquet(tf)
            new_ev, new_toi = [], []
        for k, gid in enumerate(todo, 1):
            j = _get(gid)
            if j and j.get("plays"):
                r, t = parse_game(j)
                new_ev += r
                new_toi += t
            else:
                failed += 1
            if k % 50 == 0:
                flush()
            if k % 250 == 0:
                print(f"    {k}/{len(todo)}  ({failed} failed)", flush=True)
        flush()
    return ev, toi


if __name__ == "__main__":
    a = list(map(int, sys.argv[1:])) or [2020, nhl_data.current_season()]
    for s in range(a[0], (a[1] if len(a) > 1 else a[0]) + 1):
        ev, toi = season(s)
        print(s, f"{toi['game_id'].nunique() if not toi.empty else 0} games,", len(ev), "events", flush=True)
