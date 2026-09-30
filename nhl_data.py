"""NHL data: game results + schedule (NHL API), team stats (MoneyPuck), starting goalies (MoneyPuck shot files).

Sources (all free, public):
  * NHL API  https://api-web.nhle.com   schedule, final scores (incl. OT/SO), current rosters
  * MoneyPuck https://moneypuck.com     team game-by-game stats with expected goals; shot-level data (goalie faced)
    MoneyPuck data is free for non-commercial use with credit.

Everything is cached in cache/. Seasons are labelled by starting year (2025 = 2025-26).
"""
from __future__ import annotations

import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
NHL = "https://api-web.nhle.com/v1"
FIRST_SEASON = 2016
# one code per franchise (Arizona -> Utah, Atlanta -> Winnipeg; MoneyPuck's dotted codes -> NHL tricodes)
CODE = {"L.A": "LAK", "N.J": "NJD", "S.J": "SJS", "T.B": "TBL", "ARI": "UTA", "ATL": "WPG", "PHX": "UTA", "UTAH": "UTA"}


def code(t: str) -> str:
    return CODE.get(t, t)


def current_season(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


def _get(url, **params):
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return {}


# ----------------------------------------------------------------------------------------------- schedule
def _schedule_rows(js) -> list[dict]:
    rows = []
    for day in js.get("gameWeek", []):
        for g in day.get("games", []):
            if g.get("gameType") not in (2, 3):             # 2 regular season, 3 playoffs
                continue
            h, a = g["homeTeam"], g["awayTeam"]
            rows.append({"game_id": int(g["id"]), "season": int(str(g["season"])[:4]), "game_type": "REG" if g["gameType"] == 2 else "POST",
                         "date": day["date"], "start_utc": g.get("startTimeUTC"), "state": g.get("gameState"),
                         "home": code(h["abbrev"]), "away": code(a["abbrev"]),
                         "home_score": h.get("score"), "away_score": a.get("score"),
                         "last_period": (g.get("gameOutcome") or {}).get("lastPeriodType"),
                         "venue": (g.get("venue") or {}).get("default")})
    return rows


def schedule(season: int, refresh: bool = False) -> pd.DataFrame:
    """Every regular-season and playoff game of a season (one call per week, cached; current season re-fetched)."""
    f = CACHE / f"schedule_{season}.parquet"
    if f.exists() and not (refresh or season >= current_season()):
        return pd.read_parquet(f)
    rows, d = [], date(season, 9, 25)
    while d <= date(season + 1, 6, 30):
        js = _get(f"{NHL}/schedule/{d.isoformat()}")
        rows += _schedule_rows(js)
        nxt = js.get("nextStartDate")
        d = date.fromisoformat(nxt) if nxt and date.fromisoformat(nxt) > d else d + timedelta(days=7)
    df = pd.DataFrame(rows).drop_duplicates("game_id")
    if not df.empty:
        df.to_parquet(f)
    return df


# ----------------------------------------------------------------------------------------------- team stats
def team_games() -> pd.DataFrame:
    """One row per team per game: all-situation, 5v5, power-play and penalty-kill stats (MoneyPuck)."""
    f = CACHE / "team_games.parquet"
    src = CACHE / "all_teams.csv"
    if f.exists() and f.stat().st_mtime >= src.stat().st_mtime:
        return pd.read_parquet(f)
    cols = ["team", "season", "gameId", "situation", "home_or_away", "gameDate", "playoffGame", "iceTime",
            "xGoalsFor", "xGoalsAgainst", "scoreVenueAdjustedxGoalsFor", "scoreVenueAdjustedxGoalsAgainst",
            "goalsFor", "goalsAgainst", "shotAttemptsFor", "shotAttemptsAgainst", "penaltiesFor", "penaltiesAgainst"]
    d = pd.read_csv(src, usecols=cols, low_memory=False)
    d = d[d["season"] >= FIRST_SEASON - 1]
    d["team"] = d["team"].map(code)
    key = ["team", "season", "gameId", "home_or_away", "gameDate", "playoffGame"]
    piv = {}
    for sit, spec in {"all": {"goalsFor": "gf", "goalsAgainst": "ga", "xGoalsFor": "xgf", "xGoalsAgainst": "xga",
                              "penaltiesFor": "pen_drawn", "penaltiesAgainst": "pen_taken"},
                      "5on5": {"scoreVenueAdjustedxGoalsFor": "xgf5", "scoreVenueAdjustedxGoalsAgainst": "xga5",
                               "shotAttemptsFor": "cf5", "shotAttemptsAgainst": "ca5", "iceTime": "toi5"},
                      "5on4": {"xGoalsFor": "pp_xgf", "iceTime": "pp_toi"},
                      "4on5": {"xGoalsAgainst": "pk_xga", "iceTime": "pk_toi"}}.items():
        piv[sit] = d[d["situation"] == sit].set_index(key)[list(spec)].rename(columns=spec)
    out = pd.concat(piv.values(), axis=1).reset_index().rename(columns={"gameId": "game_id"})
    out["is_home"] = (out["home_or_away"] == "HOME").astype(int)
    out.to_parquet(f)
    return out


def refresh_team_games():
    """Re-download MoneyPuck's team file (~125 MB) - do this during the season so last night's games are included."""
    r = requests.get("https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv", timeout=300)
    r.raise_for_status()
    (CACHE / "all_teams.csv").write_bytes(r.content)


# ----------------------------------------------------------------------------------------------- goalies
def goalie_games(season: int, refresh: bool = False) -> pd.DataFrame:
    """Per game and defending team: starting goalie, and expected vs actual goals against for each goalie who played."""
    f = CACHE / f"goalie_games_{season}.parquet"
    z = CACHE / f"shots_{season}.zip"
    if f.exists() and not refresh and (not z.exists() or f.stat().st_mtime >= z.stat().st_mtime):
        return pd.read_parquet(f)
    if not z.exists() or refresh:
        r = requests.get(f"https://peter-tanner.com/moneypuck/downloads/shots_{season}.zip", timeout=300)
        if r.status_code != 200:
            return pd.DataFrame()
        z.write_bytes(r.content)
    zf = zipfile.ZipFile(z)
    s = pd.read_csv(zf.open(zf.namelist()[0]), usecols=["season", "game_id", "isPlayoffGame", "teamCode", "homeTeamCode", "awayTeamCode",
                                                        "goalieIdForShot", "goalieNameForShot", "xGoal", "goal", "shotOnEmptyNet", "period", "time"],
                    low_memory=False)
    s = s[(s["shotOnEmptyNet"] == 0) & (s["period"] <= 4) & s["goalieIdForShot"].notna() & (s["goalieIdForShot"] > 0)]
    s["game_id"] = s["season"] * 1_000_000 + (np.where(s["isPlayoffGame"] == 1, 30000, 20000) + s["game_id"] % 10000)
    s["def_team"] = np.where(s["teamCode"] == s["homeTeamCode"], s["awayTeamCode"], s["homeTeamCode"])
    s["def_team"] = s["def_team"].map(code)
    s = s.sort_values(["game_id", "period", "time"])
    g = s.groupby(["game_id", "def_team", "goalieIdForShot"]).agg(
        goalie=("goalieNameForShot", "first"), xga=("xGoal", "sum"), ga=("goal", "sum"), shots=("xGoal", "size"),
        first_seen=("time", lambda t: 0)).reset_index()
    first = s.groupby(["game_id", "def_team"])["goalieIdForShot"].first().rename("starter_id")
    g = g.merge(first, on=["game_id", "def_team"])
    g["started"] = (g["goalieIdForShot"] == g["starter_id"]).astype(int)
    g = g.drop(columns=["first_seen", "starter_id"]).rename(columns={"goalieIdForShot": "goalie_id", "def_team": "team"})
    g["goalie_id"] = g["goalie_id"].astype(int)
    g["season"] = season
    g.to_parquet(f)
    return g


def current_goalies(team: str) -> list[dict]:
    """Goalies on a team's roster right now (NHL API), to handle off-season moves."""
    js = _get(f"{NHL}/roster/{team}/current")
    return [{"goalie_id": int(p["id"]), "goalie": f"{p['firstName']['default']} {p['lastName']['default']}"}
            for p in js.get("goalies", [])]
