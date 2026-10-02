"""Honest walk-forward backtest of the NHL model.

The model is refit every 14 days using only earlier games; win-probability calibration for a season uses only
earlier seasons. Graded on regular-season games 2019-20 onward; compared with sportsbook moneylines for 2024-25+.

    python nhl_backtest.py     -> output/backtest_report.txt, backtest_summary.json, oos_predictions.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import nhl_model as m

OUT = Path("output")
FIRST_TEST, REFIT_DAYS = 2022, 14
LOG: list[str] = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))


def american_to_decimal(o):
    o = np.asarray(o, float)
    return np.where(o > 0, 1 + o / 100, 1 + 100 / np.abs(o))


def devig(h, a):
    ph, pa = 1 / american_to_decimal(h), 1 / american_to_decimal(a)
    return ph / (ph + pa)


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def walk_forward(sched, X) -> pd.DataFrame:
    X = X.sort_values("date")
    played = X[X["played"] & X["goals"].notna()]
    test_dates = sorted(X.loc[X["played"] & (X["season"] >= FIRST_TEST) & (X["game_type"] == "REG"), "date"].unique())
    parts, i = [], 0
    while i < len(test_dates):
        start = test_dates[i]
        end = start + pd.Timedelta(days=REFIT_DAYS)
        train = played[(played["date"] < start) & (played["season"] >= m.nhl_data.FIRST_SEASON + 1)]
        mdl = m.make_model().fit(train[m.FEATURES].fillna(train[m.FEATURES].median()), train["goals"])
        block = X[(X["date"] >= start) & (X["date"] < end) & X["played"] & (X["game_type"] == "REG")]
        block = block.assign(lam=mdl.predict(block[m.FEATURES].fillna(train[m.FEATURES].median())))
        parts.append(block[["game_id", "team", "is_home", "lam"]])
        while i < len(test_dates) and test_dates[i] < end:
            i += 1
    P = pd.concat(parts)
    g = sched[sched["played"] & (sched["game_type"] == "REG") & (sched["season"] >= FIRST_TEST)].copy()
    g = g.merge(P[P.is_home == 1][["game_id", "lam"]].rename(columns={"lam": "lam_h"}), on="game_id")
    g = g.merge(P[P.is_home == 0][["game_id", "lam"]].rename(columns={"lam": "lam_a"}), on="game_id")
    g["home_win"] = (g["home_score"] > g["away_score"]).astype(int)
    return g


def side_market_calibration(x: pd.DataFrame) -> dict:
    """Test the puck-line and total probabilities against closing prices, and fit corrections.

    Raw goal-count (Poisson) probabilities are a bit off: too cautious on favorites covering -1.5 (empty-net goals make
    2-goal wins common) and slightly overconfident on totals. A Platt correction per market fixes the scale.
    In testing neither market showed an edge at closing prices, so the site labels them leans."""
    out = {}
    y = x.dropna(subset=["close_home_pl"])
    if len(y) > 300:
        home_fav = (y["close_home_pl"] < 0).to_numpy()
        p_fav = np.where(home_fav, m.game_probs(y["lam_h"].to_numpy(), y["lam_a"].to_numpy())["p_home_cover_m15"],
                         m.game_probs(y["lam_a"].to_numpy(), y["lam_h"].to_numpy())["p_home_cover_m15"])
        margin = (y["home_score"] - y["away_score"]).to_numpy()
        fav_cov = np.where(home_fav, margin >= 2, margin <= -2).astype(int)
        out["pl_calib"] = m.fit_platt(p_fav, fav_cov)
        say(f"  puck line: favorite -1.5 predicted {p_fav.mean():.1%} vs covered {fav_cov.mean():.1%} (correction fitted)")
    y = x.dropna(subset=["close_total"])
    if len(y) > 300 and "last_period" in y:
        so_h = ((y["last_period"] == "SO") & (y["home_score"] > y["away_score"])).astype(int)
        so_a = ((y["last_period"] == "SO") & (y["away_score"] > y["home_score"])).astype(int)
        tot = (y["home_score"] - so_h + y["away_score"] - so_a).to_numpy()
        line = y["close_total"].to_numpy()
        pr = m.game_probs(y["lam_h"].to_numpy(), y["lam_a"].to_numpy(), total_line=line)
        share = pr["p_over"] / (pr["p_over"] + pr["p_under"])
        decided = tot != line
        out["ou_calib"] = m.fit_platt(share[decided], (tot > line)[decided].astype(int))
        say(f"  totals: over predicted {share[decided].mean():.1%} vs hit {(tot > line)[decided].mean():.1%} (correction fitted)")
    return out


def main():
    OUT.mkdir(exist_ok=True)
    LOG.clear()
    sched, L, G = m.load_games()
    X = m.attach_matchup(L)
    g = walk_forward(sched, X)
    pr = m.game_probs(g["lam_h"].to_numpy(), g["lam_a"].to_numpy())
    g["p_raw"] = pr["p_home_win"]
    g["p_home_win"] = np.nan
    for s in sorted(g.season.unique()):                                  # season-by-season calibration
        prev = g[g.season < s]
        ab = m.fit_platt(prev.p_raw.to_numpy(), prev.home_win.to_numpy()) if len(prev) > 500 else None
        g.loc[g.season == s, "p_home_win"] = m.platt(g.loc[g.season == s, "p_raw"].to_numpy(), ab)
    g["pick"] = np.where(g.p_home_win >= 0.5, g.home, g.away)
    g["pick_right"] = ((g.p_home_win >= 0.5) == (g.home_win == 1)).astype(int)

    say(f"NHL walk-forward backtest, regular season {g.season.min()}-{g.season.max() + 1}: {len(g)} games")
    say("\nA. PICKING WINNERS (every game)")
    rows = []
    for s, sub in g.groupby("season"):
        rows.append({"season": f"{s}-{str(s + 1)[2:]}", "games": len(sub), "model_right": sub.pick_right.mean(),
                     "home_team_right": sub.home_win.mean(), "log_loss": logloss(sub.p_home_win, sub.home_win),
                     "coinflip_log_loss": logloss(np.full(len(sub), sub.home_win.mean()), sub.home_win)})
    t = pd.DataFrame(rows)
    say(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    say(f"all: model right {g.pick_right.mean():.1%}, always-home {g.home_win.mean():.1%}")

    say("\nB. CALIBRATION (does 60% mean 60%?)")
    g["bin"] = pd.qcut(g.p_home_win, 8)
    say(g.groupby("bin", observed=True).agg(games=("home_win", "size"), predicted=("p_home_win", "mean"),
                                            actual=("home_win", "mean")).reset_index(drop=True).to_string(float_format=lambda x: f"{x:.3f}"))

    summary = {"games": int(len(g)), "winner_pct": float(g.pick_right.mean()), "log_loss": logloss(g.p_home_win, g.home_win)}
    # ------------------------------------------------------------------ vs the market
    odds = [pd.read_parquet(f) for f in sorted(Path("cache").glob("odds_*.parquet"))]
    if odds:
        o = pd.concat(odds, ignore_index=True)
        o["date"] = pd.to_datetime(o["date"])
        x = g.merge(o, on=["date", "home", "away"], how="inner").dropna(subset=["close_home_ml", "close_away_ml"])
        x["p_close"] = devig(x.close_home_ml, x.close_away_ml)
        has_open = x.open_home_ml.notna() & x.open_away_ml.notna()
        x["p_open"] = np.where(has_open, devig(x.open_home_ml.fillna(-110), x.open_away_ml.fillna(-110)), np.nan)
        say(f"\nC. VS THE SPORTSBOOK (ESPN lines, {len(x)} games, {int(has_open.sum())} with opening lines)")
        say(f"  log loss: model {logloss(x.p_home_win, x.home_win):.4f} | closing line {logloss(x.p_close, x.home_win):.4f} "
            f"| opening line {logloss(x.p_open[has_open], x.home_win[has_open]):.4f}  (lower is better)")
        say(f"  winners: model {x.pick_right.mean():.1%} | closing-line favorite {((x.p_close >= .5) == (x.home_win == 1)).mean():.1%}")
        # does the line move toward the model between open and close?
        xo = x[has_open]
        gap, move = xo.p_home_win - xo.p_open, xo.p_close - xo.p_open
        b = float((gap @ move) / (gap @ gap))
        resid = move - b * gap
        tstat = b / np.sqrt((resid @ resid) / (len(gap) - 1) / (gap @ gap))
        say(f"  line movement: the market moves {b:.0%} of the way from the open toward the model (t={tstat:.1f})")
        rows = []
        for when in ("open", "close"):
            sub = x if when == "close" else xo
            p_m = sub[f"p_{when}"]
            for thr in (0.02, 0.04, 0.06):
                edge_h, edge_a = sub.p_home_win - p_m, (1 - sub.p_home_win) - (1 - p_m)
                bet_h, bet_a = edge_h >= thr, edge_a >= thr
                odds_h, odds_a = sub[f"{when}_home_ml"], sub[f"{when}_away_ml"]
                prof = np.where(bet_h, np.where(sub.home_win == 1, american_to_decimal(odds_h) - 1, -1), 0) + \
                       np.where(bet_a, np.where(sub.home_win == 0, american_to_decimal(odds_a) - 1, -1), 0)
                n = int(bet_h.sum() + bet_a.sum())
                wins = int(((bet_h) & (sub.home_win == 1)).sum() + ((bet_a) & (sub.home_win == 0)).sum())
                clv = np.concatenate([(sub.p_close - sub.p_open)[bet_h], (sub.p_open - sub.p_close)[bet_a]]) if when == "open" else []
                rows.append({"bet at": when, "edge >=": f"{thr:.0%}", "bets": n, "won": wins, "units": prof.sum(),
                             "roi": prof.sum() / n if n else np.nan,
                             "beat_close": float((np.asarray(clv) > 0).mean()) if len(clv) else np.nan})
        say("\n  Moneyline bets when the model's win chance beats the sportsbook's (de-vigged) by the edge shown:")
        say(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        summary.update({"market_games": int(len(x)), "model_log_loss_mkt_games": logloss(x.p_home_win, x.home_win),
                        "close_log_loss": logloss(x.p_close, x.home_win), "move_share": b, "move_t": float(tstat),
                        "bets": rows})
        g = g.merge(x[["game_id", "p_open", "p_close", "open_home_ml", "open_away_ml", "close_home_ml", "close_away_ml",
                       "close_total"]], on="game_id", how="left")
        summary.update(side_market_calibration(x))
    (OUT / "backtest_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    g.drop(columns=["bin"]).to_csv(OUT / "oos_predictions.csv", index=False)
    (OUT / "backtest_report.txt").write_text("\n".join(LOG), encoding="utf-8")
    say("\nsaved output/backtest_report.txt, backtest_summary.json, oos_predictions.csv")


if __name__ == "__main__":
    main()
