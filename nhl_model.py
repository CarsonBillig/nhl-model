"""NHL model, following the structure Rob Pizzola describes (team ratings -> goalie model -> game-day adjustments -> prices):

  1. Team ratings: 5v5 expected goals for/against per 60 (our own xG), shot-attempt share (Corsi), power-play
     and penalty-kill xG rates, penalty tendencies, finishing (goals minus xG, heavily shrunk). Recency-weighted, and the
     previous season carries over as a stabiliser that fades as the new season's games pile up.
  2. Goalie ratings: goals saved above expected per expected goal faced, recency-weighted over several seasons and
     shrunk hard toward league average when a goalie has faced few shots.
  3. Game-day: starting goalies, home ice, rest (back-to-backs).
  4. A Poisson regression turns those into each team's expected goals; goals -> win / puck line / total probabilities.

Everything is pre-game: a game's own stats never feed its own prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nhl_data

TEAM_HALFLIFE, TEAM_CARRY, TEAM_PRIOR = 25.0, 0.55, 8.0          # games; share kept over the summer; prior strength in games
GOALIE_HALFLIFE, GOALIE_CARRY, GOALIE_PRIOR_XG = 60.0, 0.75, 60.0  # goalie: games; summer carry; shrinkage in xG faced

# (name, numerator, denominator or None=per game, prior strength multiplier)
TEAM_STATS = [
    ("xgf60_5", "xgf5", "toi5", 3600), ("xga60_5", "xga5", "toi5", 3600),
    ("cf_pct5", "cf5", "c_tot5", 1), ("pp_xgf60", "pp_xgf", "pp_toi", 3600), ("pk_xga60", "pk_xga", "pk_toi", 3600),
    ("pen_drawn_pg", "pen_drawn", None, 1), ("pen_taken_pg", "pen_taken", None, 1),
    ("gf_pg", "gf", None, 1), ("ga_pg", "ga", None, 1),
]
FEATURES = ["own_xgf60_5", "own_cf_pct5", "own_pp_xgf60", "own_pen_drawn_pg", "own_finish", "own_gf_pg",
            "opp_xga60_5", "opp_cf_pct5", "opp_pk_xga60", "opp_pen_taken_pg", "opp_ga_pg", "opp_goalie_gsax", "opp_goalie_known",
            "is_home", "own_b2b", "opp_b2b"]


# ----------------------------------------------------------------------------------------------- helpers
def _recency_ratings(df: pd.DataFrame, key: str, specs, halflife, carry, prior_games, prior_den=None) -> pd.DataFrame:
    """Pre-game recency-weighted ratings per `key` (team or goalie), carrying a share over each summer.

    specs: list of (name, num_col, den_col|None, scale). The prior is `prior_games` average games of league-average
    play (or `prior_den` units of the denominator, e.g. expected goals faced). Returns one column per spec."""
    decay = 0.5 ** (1 / halflife)
    df = df.sort_values([key, "date", "game_id"])
    league, kprior = {}, {}
    for name, num, den, scale in specs:
        league[name] = df[num].sum() / (df[den].sum() if den else df[num].notna().sum())
        kprior[name] = prior_den if prior_den is not None else prior_games * (df[den].mean() if den else 1.0)
    out = {name: np.full(len(df), np.nan) for name, *_ in specs}
    idx = {k: np.array(v) for k, v in df.groupby(key, sort=False).indices.items()}
    seasons = df["season"].to_numpy()
    cols = {c: df[c].to_numpy(float) for _, n, dn, _ in specs for c in (n, dn) if c}
    for _, rows in idx.items():
        acc_n = {name: 0.0 for name, *_ in specs}
        acc_d = {name: 0.0 for name, *_ in specs}
        prev = None
        for i in rows:
            if prev is not None and seasons[i] != prev:
                for name in acc_n:
                    acc_n[name] *= carry
                    acc_d[name] *= carry
            for name, num, den, scale in specs:
                k = kprior[name]
                out[name][i] = (acc_n[name] + k * league[name]) / (acc_d[name] + k) * scale
                x_n, x_d = cols[num][i], (cols[den][i] if den else 1.0)
                if not (np.isnan(x_n) or np.isnan(x_d)):
                    acc_n[name] = acc_n[name] * decay + x_n
                    acc_d[name] = acc_d[name] * decay + x_d
            prev = seasons[i]
    res = pd.DataFrame(out, index=df.index)
    return res.reindex(df.index)


# ----------------------------------------------------------------------------------------------- data assembly
def load_games(refresh_current: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (schedule, long, goalie_games): games, one row per team per game with pre-game ratings, goalie log."""
    cur = nhl_data.current_season()
    sched = pd.concat([nhl_data.schedule(s, refresh=refresh_current and s == cur)
                       for s in range(nhl_data.FIRST_SEASON, cur + 1)], ignore_index=True)
    sched["date"] = pd.to_datetime(sched["date"])
    sched["played"] = sched["state"].isin(["OFF", "FINAL"]) & sched["home_score"].notna()
    tg = nhl_data.team_games(refresh_current=refresh_current)
    gg = pd.concat([nhl_data.goalie_games(s)
                    for s in range(nhl_data.FIRST_SEASON, cur + 1)], ignore_index=True)

    # long table: one row per team per game (scheduled games included, stats NaN until played)
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        x = sched[["game_id", "season", "game_type", "date", "played", side, opp]].rename(columns={side: "team", opp: "opp"})
        x["is_home"] = int(side == "home")
        rows.append(x)
    L = pd.concat(rows, ignore_index=True)
    L = L.merge(tg.drop(columns=["season", "home_or_away", "gameDate", "playoffGame", "is_home"]), on=["game_id", "team"], how="left")
    L["c_tot5"] = L["cf5"] + L["ca5"]
    # The model predicts REGULAR goals; empty-net goals depend on game state (a late 1-2 goal lead), not team
    # strength, so they are added afterwards by the empty-net step in game_probs. (en_goals comes with the team stats.)
    L["en_goals"] = L["en_goals"].fillna(0)
    L["goals"] = L["gf"] - L["en_goals"]
    L.loc[~L["played"], [c for c in tg.columns if c not in ("team", "season", "game_id", "home_or_away", "gameDate", "playoffGame", "is_home")]] = np.nan

    # team ratings (pre-game)
    R = _recency_ratings(L, "team", TEAM_STATS, TEAM_HALFLIFE, TEAM_CARRY, TEAM_PRIOR)
    L = L.join(R)
    fin = _recency_ratings(L.assign(gmx=L["gf"] - L["xgf"]), "team", [("finish", "gmx", "xgf", 1)], TEAM_HALFLIFE * 2, TEAM_CARRY, 40)
    L["finish"] = fin["finish"]         # goals above expected per xG, shrunk heavily toward league average

    # rest
    L = L.sort_values(["team", "date", "game_id"])
    L["rest_days"] = L.groupby("team")["date"].diff().dt.days
    L["b2b"] = (L["rest_days"] == 1).astype(int)

    # goalies: pre-game ratings per goalie, then attach the starter (actual for played games)
    G = gg.merge(sched[["game_id", "date"]], on="game_id")
    G["saved_above"] = G["xga"] - G["ga"]
    gr = _recency_ratings(G, "goalie_id", [("gsax_rate", "saved_above", "xga", 1)], GOALIE_HALFLIFE, GOALIE_CARRY, 0,
                          prior_den=GOALIE_PRIOR_XG)
    G = G.join(gr)
    G = G.sort_values(["goalie_id", "date", "game_id"])
    G["xga_exp"] = G.groupby("goalie_id")["xga"].transform(lambda s: s.shift().fillna(0).cumsum())
    starters = G[G["started"] == 1][["game_id", "team", "goalie_id", "goalie", "gsax_rate", "xga_exp"]]
    L = L.merge(starters, on=["game_id", "team"], how="left")
    return sched, L, G


def goalie_table(G: pd.DataFrame) -> pd.DataFrame:
    """Latest rating for every goalie (after all games so far), plus recent starts."""
    decay = 0.5 ** (1 / GOALIE_HALFLIFE)
    G = G.sort_values(["goalie_id", "date"])
    out = []
    for gid, g in G.groupby("goalie_id"):
        n = d = 0.0
        prev = None
        for s, sa, xg in zip(g["season"], g["saved_above"], g["xga"]):
            if prev is not None and s != prev:
                n, d = n * GOALIE_CARRY, d * GOALIE_CARRY
            n, d, prev = n * decay + sa, d * decay + xg, s
        out.append({"goalie_id": gid, "goalie": g["goalie"].iloc[-1], "team_last": g["team"].iloc[-1],
                    "gsax_rate": n / (d + GOALIE_PRIOR_XG), "last_date": g["date"].max(), "xga_faced": g["xga"].sum()})
    return pd.DataFrame(out)


def attach_matchup(L: pd.DataFrame) -> pd.DataFrame:
    """own_* from the team row, opp_* from the opponent's row of the same game."""
    base = ["game_id", "team"]
    own_cols = ["xgf60_5", "cf_pct5", "pp_xgf60", "pen_drawn_pg", "finish", "gf_pg", "b2b"]
    opp_cols = ["xga60_5", "cf_pct5", "pk_xga60", "pen_taken_pg", "ga_pg", "b2b", "gsax_rate"]
    X = L.copy()
    for c in own_cols:
        X[f"own_{c}"] = X[c]
    O = L[base + opp_cols + ["goalie_id"]].rename(columns={"team": "opp", **{c: f"opp_{c}" for c in opp_cols}})
    O = O.rename(columns={"goalie_id": "opp_goalie_id"})
    X = X.merge(O, on=["game_id", "opp"], how="left")
    # a goalie's rating counts once he has faced some shots; otherwise league average
    X["opp_goalie_gsax"] = X["opp_gsax_rate"].fillna(0.0)
    X["opp_goalie_known"] = X["opp_gsax_rate"].notna().astype(int)
    return X


def make_model():
    return make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=500))


# ----------------------------------------------------------------------------------------------- probabilities
MAXG = 15
_K = np.arange(MAXG + 1)


# Empty-net step, measured on 7,440 regular-season games (2020-26, NHL play-by-play): when a team leads by m REGULAR
# (non-empty-net) goals, the chance it adds 0 / 1 / 2 empty-net goals. Up 1 or 2, trailing teams pull their goalie and
# the leader often scores into the empty net (more often than a decade ago: coaches pull earlier now).
EN_TABLE = {1: (0.682, 0.265, 0.053), 2: (0.343, 0.642, 0.015), 3: (0.704, 0.295, 0.001), 4: (0.960, 0.039, 0.001)}
_KF = np.arange(MAXG + 3)


def _with_empty_net(joint: np.ndarray) -> np.ndarray:
    """Turn a [game, home, away] distribution of regular goals into one of final goals (empty-netters added)."""
    n, k = joint.shape[0], joint.shape[1]
    final = np.zeros((n, k + 2, k + 2))
    for h in range(k):
        for a in range(k):
            p = joint[:, h, a]
            m = h - a
            probs = EN_TABLE.get(abs(m), (1.0, 0.0, 0.0))
            for extra, q in enumerate(probs):
                if q == 0:
                    continue
                if m > 0:
                    final[:, h + extra, a] += p * q
                elif m < 0:
                    final[:, h, a + extra] += p * q
                else:
                    final[:, h, a] += p * q if extra == 0 else 0.0
    return final


def game_probs(lam_h, lam_a, total_line=None, home_pl=-1.5) -> dict:
    """Poisson REGULAR goals (regulation + OT), then the empty-net step. Ties go to OT/shootout, split by strength.

    Empty-net goals never change the winner, but they turn many 1-goal wins into 2-goal wins (puck line) and push totals up."""
    lam_h, lam_a = np.atleast_1d(lam_h).astype(float), np.atleast_1d(lam_a).astype(float)
    ph = poisson.pmf(_K[None, :], lam_h[:, None])
    pa = poisson.pmf(_K[None, :], lam_a[:, None])
    joint = _with_empty_net(ph[:, :, None] * pa[:, None, :])         # [game, home goals, away goals], final
    diff = _KF[:, None] - _KF[None, :]
    p_home_reg = (joint * (diff > 0)).sum((1, 2))
    p_tie = (joint * (diff == 0)).sum((1, 2))
    ot_home = 0.5 + 0.5 * (lam_h - lam_a) / (lam_h + lam_a)      # OT/SO: slight edge to the better team
    out = {"p_home_win": p_home_reg + p_tie * ot_home, "p_tie_reg": p_tie,
           "p_home_cover_m15": (joint * (diff >= 2)).sum((1, 2)),   # home -1.5 needs a 2+ goal win (OT wins are by 1)
           "exp_home": (joint.sum(2) * _KF).sum(1), "exp_away": (joint.sum(1) * _KF).sum(1)}
    out["p_away_cover_p15"] = 1 - out["p_home_cover_m15"]
    tot = _KF[:, None] + _KF[None, :]
    if total_line is not None:
        tl = np.asarray(total_line, float)[:, None, None]
        out["p_over"] = (joint * (tot[None] > tl)).sum((1, 2))
        out["p_under"] = (joint * (tot[None] < tl)).sum((1, 2))
    return out


def fit_platt(p, y):
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(C=1e4).fit(x[:, None], y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def platt(p, ab):
    if ab is None:
        return p
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(ab[0] * x + ab[1])))
