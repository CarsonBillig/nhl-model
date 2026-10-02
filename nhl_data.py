"""NHL data, all from the NHL's official API (https://api-web.nhle.com): schedule, final scores (incl. OT/SO),
current rosters, and play-by-play, from which team stats, goalie stats and empty-net goals are built with our own
expected-goals model (nhl_xg.py). No third-party data.

Everything is cached in cache/. Seasons are labelled by starting year (2025 = 2025-26).
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
NHL = "https://api-web.nhle.com/v1"
FIRST_SEASON = 2020                  # first season of play-by-play we use
# one code per franchise (Arizona -> Utah, Atlanta -> Winnipeg; older dotted codes -> NHL tricodes)
CODE = {"L.A": "LAK", "N.J": "NJD", "S.J": "SJS", "T.B": "TBL", "ARI": "UTA", "ATL": "WPG", "PHX": "UTA", "UTAH": "UTA"}


def code(t: str) -> str:
    return CODE.get(t, t)


def current_season(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


def _get(url, **params):
    for attempt in range(6):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:                  # rate-limited: wait as long as the API asks
                time.sleep(float(r.headers.get("retry-after") or 30) + 2)
                continue
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 * (attempt + 1))
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
        if not js:                                   # throttled: wait and try this week once more
            time.sleep(20)
            js = _get(f"{NHL}/schedule/{d.isoformat()}")
        rows += _schedule_rows(js)
        nxt = js.get("nextStartDate")
        d = date.fromisoformat(nxt) if nxt and date.fromisoformat(nxt) > d else d + timedelta(days=7)
    df = pd.DataFrame(rows)
    if f.exists():                                   # a failed week must never drop games we already have
        df = pd.concat([df, pd.read_parquet(f)], ignore_index=True)
    df = df.drop_duplicates("game_id", keep="first").sort_values(["date", "game_id"]).reset_index(drop=True)
    if not df.empty:
        df.to_parquet(f)
    return df


# ----------------------------------------------------------------------------------------------- derived stats
# Everything below is built from the NHL's own play-by-play (nhl_pbp.py) and our own xG model (nhl_xg.py).
FIVE_V_FIVE = "1551"                       # situation code: away goalie, away skaters, home skaters, home goalie
HOME_PP, HOME_PK = "1451", "1541"          # home 5 vs away 4 / home 4 vs away 5 (both goalies in)


def _season_events(season: int, refresh: bool = False):
    import nhl_pbp
    import nhl_xg
    ev, toi = nhl_pbp.season(season, refresh=refresh)
    if ev.empty:
        return ev, toi
    return nhl_xg.add_xg(ev), toi


def _derived(season: int, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(team games, goalie games, empty-net goals) for one season, cached; the current season is rebuilt each run."""
    f_t, f_g, f_e = (CACHE / f"derived_{n}_{season}.parquet" for n in ("team", "goalie", "en"))
    if all(f.exists() for f in (f_t, f_g, f_e)) and not (refresh or season >= current_season()):
        return pd.read_parquet(f_t), pd.read_parquet(f_g), pd.read_parquet(f_e)
    ev, toi = _season_events(season, refresh and season >= current_season())
    if not ev.empty:     # a season's schedule file can include the previous season's late playoffs (2020 bubble)
        ev, toi = ev[ev["season"] == season], toi[toi["season"] == season]
    if ev.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    sched = schedule(season)[["game_id", "date", "game_type"]]
    shots = ev[ev["kind"].isin(["sog", "miss", "goal", "block"])]
    unblocked = shots[shots["kind"] != "block"]
    goals = shots[shots["kind"] == "goal"]
    pens = ev[ev["kind"] == "pen"]
    rows = []
    for side in ("home", "away"):
        is_home = side == "home"
        games = toi.groupby("game_id")[["home", "away"]].first().reset_index()
        g = games.assign(team=games[side], opp=games["away" if is_home else "home"], is_home=int(is_home))
        def by_team(df, col=None):
            s = df.groupby(["game_id", "team"]).size() if col is None else df.groupby(["game_id", "team"])[col].sum()
            return s
        def by_opp(df, col=None):
            s = df.groupby(["game_id", "opp"]).size() if col is None else df.groupby(["game_id", "opp"])[col].sum()
            return s.rename_axis(["game_id", "team"])
        five = unblocked[unblocked["situation"] == FIVE_V_FIVE]
        att5 = shots[shots["situation"] == FIVE_V_FIVE]
        pp_code, pk_code = (HOME_PP, HOME_PK) if is_home else (HOME_PK, HOME_PP)
        sec = toi.pivot_table(index="game_id", columns="situation", values="seconds", aggfunc="sum").fillna(0)
        stat = pd.DataFrame(index=pd.MultiIndex.from_frame(g[["game_id", "team"]]))
        stat["gf"] = by_team(goals)
        stat["ga"] = by_opp(goals)
        stat["en_goals"] = by_team(goals[goals["empty_net"] == 1])
        stat["xgf"] = by_team(unblocked, "xg")
        stat["xga"] = by_opp(unblocked, "xg")
        stat["xgf5"] = by_team(five, "xg")
        stat["xga5"] = by_opp(five, "xg")
        stat["cf5"] = by_team(att5)
        stat["ca5"] = by_opp(att5)
        team_pp = unblocked[unblocked["situation"] == pp_code]
        team_pk = unblocked[unblocked["situation"] == pk_code]
        stat["pp_xgf"] = by_team(team_pp[team_pp["is_home"] == int(is_home)], "xg")
        stat["pk_xga"] = by_opp(team_pk[team_pk["is_home"] != int(is_home)], "xg")
        stat["pen_taken"] = by_team(pens)
        stat["pen_drawn"] = by_opp(pens)
        stat = stat.fillna(0).reset_index()
        stat["toi5"] = stat["game_id"].map(sec.get(FIVE_V_FIVE, pd.Series(dtype=float)))
        stat["pp_toi"] = stat["game_id"].map(sec.get(pp_code, pd.Series(dtype=float)))
        stat["pk_toi"] = stat["game_id"].map(sec.get(pk_code, pd.Series(dtype=float)))
        stat["is_home"] = int(is_home)
        rows.append(stat)
    tg = pd.concat(rows, ignore_index=True).merge(sched, on="game_id", how="left")
    tg["season"] = season
    tg["home_or_away"] = np.where(tg["is_home"] == 1, "HOME", "AWAY")
    tg["gameDate"] = pd.to_datetime(tg["date"]).dt.strftime("%Y%m%d").astype(int)
    tg["playoffGame"] = (tg["game_type"] == "POST").astype(int)
    tg = tg.drop(columns=["date", "game_type"])

    # goalies: every goalie who faced unblocked, non-empty-net shots; the starter faced his team's first shot
    gs = unblocked[(unblocked["empty_net"] == 0) & unblocked["goalie_id"].notna()].sort_values(["game_id", "t"])
    gg = gs.groupby(["game_id", "opp", "goalie_id"]).agg(goalie=("goalie", "first"), xga=("xg", "sum"),
                                                         ga=("kind", lambda k: int((k == "goal").sum())),
                                                         shots=("xg", "size")).reset_index().rename(columns={"opp": "team"})
    first = gs.groupby(["game_id", "opp"])["goalie_id"].first().rename("starter").reset_index().rename(columns={"opp": "team"})
    gg = gg.merge(first, on=["game_id", "team"])
    gg["started"] = (gg["goalie_id"] == gg["starter"]).astype(int)
    gg = gg.drop(columns="starter")
    gg["goalie_id"] = gg["goalie_id"].astype(int)
    gg["season"] = season
    en = tg[["game_id", "team", "en_goals"]]
    for f, df in ((f_t, tg), (f_g, gg), (f_e, en)):
        df.to_parquet(f)
    return tg, gg, en


def team_games(seasons=None, refresh_current: bool = False) -> pd.DataFrame:
    """One row per team per game: all-situation, 5v5, power-play and penalty-kill stats from NHL play-by-play."""
    seasons = seasons or range(FIRST_SEASON, current_season() + 1)
    parts = [_derived(s, refresh_current and s == current_season())[0] for s in seasons]
    return pd.concat([p for p in parts if not p.empty], ignore_index=True)


def goalie_games(season: int, refresh: bool = False) -> pd.DataFrame:
    """Per game and defending team: each goalie's expected vs actual goals against, and who started."""
    return _derived(season, refresh)[1]


def en_goals(season: int, refresh: bool = False) -> pd.DataFrame:
    """Empty-net goals per game and team."""
    return _derived(season, refresh)[2]


def current_goalies(team: str) -> list[dict]:
    """Goalies on a team's roster right now (NHL API), to handle off-season moves."""
    js = _get(f"{NHL}/roster/{team}/current")
    return [{"goalie_id": int(p["id"]), "goalie": f"{p['firstName']['default']} {p['lastName']['default']}"}
            for p in js.get("goalies", [])]
