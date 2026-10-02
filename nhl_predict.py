"""Daily NHL picks + pick history.

    python nhl_predict.py              # grade finished games, then picks for today and the next 2 days
    python nhl_predict.py --days 1     # just today

Writes
  output/picks_<date>.csv     today's board: projected score, win %, fair vs sportsbook moneyline, puck line, total
  output/ledger.csv           pick history: every pick logged BEFORE puck drop, graded after the game
  output/history.csv          the same history in a simple one-row-per-game layout for Excel / Google Sheets

Starting goalies are projected from current rosters and recent starts (backup on the 2nd night of a back-to-back).
Put confirmed starters in goalie_overrides.csv (date,team,goalie) and re-run before puck drop.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import nhl_data
import nhl_model as m
import nhl_odds

OUT = Path("output")
LEDGER = OUT / "ledger.csv"
OVERRIDES = Path("goalie_overrides.csv")
EDGE = 0.06                       # flag a value bet when the model's win chance beats the sportsbook's by 6+ points
                                  # (backtest 2024-26: edges this size at the OPENING line returned about +2.6% to +5%
                                  #  and beat the closing line ~70% of the time; the same bets at the closing line lost)


def american_to_decimal(o):
    o = np.asarray(o, float)
    return np.where(o > 0, 1 + o / 100, 1 + 100 / np.abs(o))


def fair_ml(p):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return np.where(p >= 0.5, -100 * p / (1 - p), 100 * (1 - p) / p).round(0)


def devig(h, a):
    ph, pa = 1 / american_to_decimal(h), 1 / american_to_decimal(a)
    return ph / (ph + pa)


def implied(odds) -> float:
    """Win chance implied by one American price (vig included)."""
    return float(1 / american_to_decimal(odds))


def lean_by_value(p_a, p_b, odds_a, odds_b, label_a, label_b) -> str:
    """The side with the better expected value at the listed prices (-110 if a price is missing).

    Puck lines and totals are leans only: the model's goal-count probabilities have not been checked against these markets."""
    if p_a is None or pd.isna(p_a):
        return ""
    oa = -110 if odds_a is None or pd.isna(odds_a) else odds_a
    ob = -110 if odds_b is None or pd.isna(odds_b) else odds_b
    ev_a = p_a * (american_to_decimal(oa) - 1) - (1 - p_a)
    ev_b = p_b * (american_to_decimal(ob) - 1) - (1 - p_b)
    return label_a if ev_a >= ev_b else label_b


# ------------------------------------------------------------------------------------------ starting goalies
def projected_starters(games: pd.DataFrame, G: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """{(game_id, team): (goalie_id, goalie, how)}: override > back-to-back backup > most recent regular starter."""
    ov = pd.read_csv(OVERRIDES) if OVERRIDES.exists() else pd.DataFrame(columns=["date", "team", "goalie"])
    out, rosters = {}, {}
    recent = G[G["started"] == 1].sort_values("date")
    for _, r in games.iterrows():
        for side in ("home", "away"):
            team = r[side]
            if team not in rosters:
                rosters[team] = nhl_data.current_goalies(team)
            roster = rosters[team]
            o = ov[(ov["team"] == team) & (pd.to_datetime(ov["date"]).dt.date == r["date"].date())]
            if len(o):
                name = str(o.iloc[0]["goalie"]).lower()
                match = [g for g in roster if name in g["goalie"].lower()] or \
                        gt[gt["goalie"].str.lower().str.contains(name, regex=False)].to_dict("records")
                if match:
                    out[(r["game_id"], team)] = (match[0]["goalie_id"], match[0]["goalie"], "confirmed (override)")
                    continue
            ids = [g["goalie_id"] for g in roster] or gt[gt["team_last"] == team]["goalie_id"].tolist()
            mine = recent[recent["goalie_id"].isin(ids)]
            # weight: starts for this team in its last 25 games count most, then starts anywhere last season
            team_recent = mine[mine["team"] == team].tail(25)["goalie_id"].value_counts()
            anywhere = mine[mine["date"] >= mine["date"].max() - pd.Timedelta(days=300)]["goalie_id"].value_counts()
            score = {gid: 3 * team_recent.get(gid, 0) + anywhere.get(gid, 0) for gid in ids}
            order = sorted(ids, key=lambda g: -score.get(g, 0))
            if not order:
                out[(r["game_id"], team)] = (np.nan, "unknown", "no goalie data")
                continue
            pick, how = order[0], "projected (usual starter)"
            if r.get(f"{side}_b2b", 0) == 1 and len(order) > 1:
                # second night of a back-to-back: teams usually start whoever did NOT play last night
                prev_team = recent[(recent["team"] == team) & (recent["date"] < r["date"])]
                last_night = prev_team["goalie_id"].iloc[-1] if len(prev_team) else None
                rested = [g for g in order if g != last_night]
                if rested:
                    pick, how = rested[0], "projected (rested goalie, 2nd of back-to-back)"
            name = next((g["goalie"] for g in roster if g["goalie_id"] == pick), gt.set_index("goalie_id")["goalie"].get(pick, str(pick)))
            out[(r["game_id"], team)] = (pick, name, how)
    return out


# ------------------------------------------------------------------------------------------ ledger
COLS = ["logged_at", "date", "game_id", "start_utc", "away", "home", "away_goalie", "home_goalie", "goalie_status",
        "proj_away", "proj_home", "p_home_win", "fair_home_ml", "fair_away_ml", "home_ml", "away_ml", "open_home_ml",
        "open_away_ml", "p_mkt_home", "pick", "pick_prob", "pick_ml", "value_side", "value_edge", "value_ml",
        "total_line", "proj_total", "p_over", "total_lean", "pl_home_line", "p_home_pl", "pl_lean",
        "away_score", "home_score", "last_period", "pick_result", "value_result", "value_units", "total_result",
        "pl_result", "close_home_ml", "close_away_ml", "value_clv", "value_logged_at"]


def load_ledger():
    return pd.read_csv(LEDGER).reindex(columns=COLS) if LEDGER.exists() else pd.DataFrame(columns=COLS)


def record(new: pd.DataFrame):
    old = load_ledger()
    now = pd.Timestamp.now(tz="UTC")
    locked = old["pick_result"].notna() | (pd.to_datetime(old["start_utc"], utc=True) <= now)
    keep = old[locked | ~old["game_id"].isin(new["game_id"])]
    fresh = new[~new["game_id"].isin(keep["game_id"]) & (pd.to_datetime(new["start_utc"], utc=True) > now)].copy()
    # A value bet is recorded the FIRST time it is flagged, at that price, and stays recorded on later runs (that is
    # when it would be bet). Re-running near puck drop would otherwise replace it with the closing price, and the
    # closing-line check would always read zero.
    fresh["value_side"] = fresh["value_side"].astype(object)
    fresh["value_logged_at"] = pd.Series(np.where(fresh["value_side"].notna(), fresh["logged_at"], None), index=fresh.index, dtype=object)
    prev = old[~locked].drop_duplicates("game_id", keep="last").set_index("game_id")
    for i in fresh.index:
        gid = fresh.at[i, "game_id"]
        if gid in prev.index and isinstance(prev.at[gid, "value_side"], str) and prev.at[gid, "value_side"]:
            for c in ("value_side", "value_edge", "value_ml"):
                fresh.at[i, c] = prev.at[gid, c]
            fresh.at[i, "value_logged_at"] = prev.at[gid, "value_logged_at"] if isinstance(prev.at[gid, "value_logged_at"], str)                 else prev.at[gid, "logged_at"]
    OUT.mkdir(exist_ok=True)
    pd.concat([keep, fresh.reindex(columns=COLS)], ignore_index=True).sort_values(["date", "start_utc", "game_id"]).to_csv(LEDGER, index=False)
    return len(fresh)


def grade(sched: pd.DataFrame) -> pd.DataFrame:
    led = load_ledger()
    if led.empty:
        return led
    for c in led.columns:
        led[c] = led[c].astype(object)
    s = sched.set_index("game_id")
    todo = led["pick_result"].isna() & led["game_id"].isin(s.index[s["played"]])
    for i in led.index[todo]:
        g = s.loc[led.at[i, "game_id"]]
        hs, as_ = float(g["home_score"]), float(g["away_score"])
        home_won = hs > as_
        winner = g["home"] if home_won else g["away"]
        led.at[i, "home_score"], led.at[i, "away_score"], led.at[i, "last_period"] = hs, as_, g["last_period"]
        led.at[i, "pick_result"] = "Win" if led.at[i, "pick"] == winner else "Loss"
        if isinstance(led.at[i, "value_side"], str) and led.at[i, "value_side"]:
            won = led.at[i, "value_side"] == winner
            led.at[i, "value_result"] = "Win" if won else "Loss"
            led.at[i, "value_units"] = float(american_to_decimal(led.at[i, "value_ml"]) - 1) if won else -1.0
        tl = pd.to_numeric(led.at[i, "total_line"], errors="coerce")
        # shootout winner gets +1 in the final score but the total counts only real goals
        tot = hs + as_ - (1 if g["last_period"] == "SO" else 0)
        if pd.notna(tl) and led.at[i, "total_lean"] in ("Over", "Under"):
            led.at[i, "total_result"] = "Push" if tot == tl else ("Win" if (tot > tl) == (led.at[i, "total_lean"] == "Over") else "Loss")
        margin = hs - as_
        pl = pd.to_numeric(led.at[i, "pl_home_line"], errors="coerce")
        if pd.notna(pl) and isinstance(led.at[i, "pl_lean"], str):
            home_cov = margin + pl > 0
            led.at[i, "pl_result"] = "Win" if home_cov == led.at[i, "pl_lean"].startswith(g["home"]) else "Loss"
    # closing lines + CLV for graded value bets (did the market move toward the pick after it was logged?)
    need = led["pick_result"].notna() & led["close_home_ml"].isna()
    for d in sorted(set(led.loc[need, "date"])):
        try:
            board = pd.DataFrame(nhl_odds.day_board(date.fromisoformat(str(d))))
        except Exception:
            continue
        if board.empty:
            continue
        cl = board.set_index(["home", "away"])
        for i in led.index[need & (led["date"] == d)]:
            key = (led.at[i, "home"], led.at[i, "away"])
            if key not in cl.index:
                continue
            ch, ca = cl.loc[key, "close_home_ml"], cl.loc[key, "close_away_ml"]
            led.at[i, "close_home_ml"], led.at[i, "close_away_ml"] = ch, ca
            side = led.at[i, "value_side"]
            bet = pd.to_numeric(led.at[i, "value_ml"], errors="coerce")
            if isinstance(side, str) and side and pd.notna(ch) and pd.notna(bet):
                # closing-line value: how much the price on our side shortened from when the bet was flagged
                close_side = ch if side == led.at[i, "home"] else ca
                led.at[i, "value_clv"] = round(float(implied(close_side) - implied(bet)), 4)
    led.to_csv(LEDGER, index=False)
    return led


def history_csv(led: pd.DataFrame):
    """One row per game, plain column names: easy to open in Excel or import to Google Sheets."""
    h = pd.DataFrame({
        "date": led["date"], "awayTeam": led["away"], "homeTeam": led["home"],
        "awayGoalie": led["away_goalie"], "homeGoalie": led["home_goalie"],
        "projAway": led["proj_away"], "projHome": led["proj_home"],
        "pick": led["pick"], "pickWinProb": (pd.to_numeric(led["pick_prob"]) * 100).round(1), "pickML": led["pick_ml"],
        "awayScore": led["away_score"], "homeScore": led["home_score"], "endedIn": led["last_period"],
        "pickResult": led["pick_result"],
        "valueBet": led["value_side"], "valueEdgePct": (pd.to_numeric(led["value_edge"]) * 100).round(1),
        "valueML": led["value_ml"], "valueResult": led["value_result"], "valueUnits": led["value_units"],
        "valueBeatClose": pd.to_numeric(led["value_clv"], errors="coerce").map(lambda v: "" if pd.isna(v) else ("Yes" if v > 0 else "No")),
        "total": led["total_line"], "totalLean": led["total_lean"], "totalResult": led["total_result"],
        "puckLineLean": led["pl_lean"], "puckLineResult": led["pl_result"],
    })
    h.to_csv(OUT / "history.csv", index=False)
    return h


def summary(led: pd.DataFrame):
    g = led[led["pick_result"].notna()]
    if g.empty:
        print("\nPick history: nothing graded yet.")
        return
    def rec(col):
        c = g[col].dropna()
        return f"{(c == 'Win').sum()}-{(c == 'Loss').sum()}" + (f"-{(c == 'Push').sum()}" if (c == "Push").any() else "")
    units = pd.to_numeric(g["value_units"], errors="coerce").sum()
    clv = pd.to_numeric(g["value_clv"], errors="coerce").dropna()
    clv_s = f" | value bets beat the close {(clv > 0).mean():.0%} of the time" if len(clv) else ""
    print(f"\nPick history: {len(g)} graded | winners {rec('pick_result')} | value bets {rec('value_result')} "
          f"({units:+.2f} units){clv_s} | totals {rec('total_result')} | puck line {rec('pl_result')}")


# ------------------------------------------------------------------------------------------ main
def run(days: int = 3):
    cur = nhl_data.current_season()
    # Everything refreshes from the NHL's official API (new games' play-by-play is fetched once and cached) and ESPN.
    sched, L, G = m.load_games(refresh_current=True)
    led = grade(sched)
    summary(led if not led.empty else load_ledger())

    X = m.attach_matchup(L)
    played = X[X["played"] & X["goals"].notna() & (X["season"] >= nhl_data.FIRST_SEASON + 1)]
    med = played[m.FEATURES].median()
    mdl = m.make_model().fit(played[m.FEATURES].fillna(med), played["goals"])
    oos = pd.read_csv(OUT / "oos_predictions.csv") if (OUT / "oos_predictions.csv").exists() else None
    calib = m.fit_platt(oos.p_raw.to_numpy(), oos.home_win.to_numpy()) if oos is not None else None
    bts = json.loads((OUT / "backtest_summary.json").read_text()) if (OUT / "backtest_summary.json").exists() else {}
    pl_calib, ou_calib = bts.get("pl_calib"), bts.get("ou_calib")      # tested corrections for puck line / totals

    today = date.today()
    days_list = [today + timedelta(days=k) for k in range(days)]
    up = sched[(~sched["played"]) & (sched["date"].dt.date.isin(days_list))].copy()
    if up.empty:
        print("No NHL games in the next", days, "days.")
        return
    b2b = L.set_index(["game_id", "team"])["b2b"]
    up["home_b2b"] = [b2b.get((g, t), 0) for g, t in zip(up.game_id, up.home)]
    up["away_b2b"] = [b2b.get((g, t), 0) for g, t in zip(up.game_id, up.away)]

    gt = m.goalie_table(G)
    starters = projected_starters(up, G, gt)
    grate = gt.set_index("goalie_id")["gsax_rate"]

    # features for the upcoming games: team rows already carry pre-game ratings; swap in projected goalies
    rows = X[X["game_id"].isin(up["game_id"])].copy()
    rows["opp_goalie_gsax"] = [grate.get(starters[(gid, opp)][0], 0.0) for gid, opp in zip(rows.game_id, rows.opp)]
    rows["opp_goalie_known"] = 1
    rows["lam"] = mdl.predict(rows[m.FEATURES].fillna(med))
    lam = rows.set_index(["game_id", "is_home"])["lam"]
    up["lam_h"] = [lam[(g, 1)] for g in up.game_id]
    up["lam_a"] = [lam[(g, 0)] for g in up.game_id]
    import nhl_stats
    team_stats = nhl_stats.build(up, L, grate.to_dict(), starters)

    # sportsbook lines (ESPN / DraftKings)
    board = pd.concat([pd.DataFrame(nhl_odds.day_board(d)) for d in days_list], ignore_index=True)
    if not board.empty:
        board["date"] = pd.to_datetime(board["date"])
        up = up.merge(board.drop(columns=["kickoff"], errors="ignore"), on=["date", "home", "away"], how="left")
    for c in ("close_home_ml", "close_away_ml", "open_home_ml", "open_away_ml", "close_total", "close_home_pl",
              "close_over_odds", "close_under_odds", "close_home_pl_odds", "close_away_pl_odds", "tv"):
        if c not in up:
            up[c] = np.nan

    tl = up["close_total"].fillna(6.0).to_numpy()
    pr = m.game_probs(up["lam_h"].to_numpy(), up["lam_a"].to_numpy(), total_line=tl)
    up["p_home_win"] = m.platt(pr["p_home_win"], calib)
    up["p_mkt_home"] = np.where(up.close_home_ml.notna() & up.close_away_ml.notna(),
                                devig(up.close_home_ml.fillna(-110), up.close_away_ml.fillna(-110)), np.nan)
    rows_out = []
    for i, r in up.reset_index(drop=True).iterrows():
        ph = r["p_home_win"]
        pick_home = ph >= 0.5
        edge_h, edge_a = ph - r["p_mkt_home"], (1 - ph) - (1 - r["p_mkt_home"])
        value_side, value_edge, value_ml = "", np.nan, np.nan
        if pd.notna(r["p_mkt_home"]):
            if edge_h >= EDGE:
                value_side, value_edge, value_ml = r["home"], edge_h, r["close_home_ml"]
            elif edge_a >= EDGE:
                value_side, value_edge, value_ml = r["away"], edge_a, r["close_away_ml"]
        p_over = pr["p_over"][i] if pd.notna(r["close_total"]) else np.nan
        p_under = pr["p_under"][i] if pd.notna(r["close_total"]) else np.nan
        if pd.notna(p_over) and ou_calib:                 # tested correction: rescale the over/under split
            push = 1 - p_over - p_under
            share = float(m.platt(np.array([p_over / (p_over + p_under)]), ou_calib)[0])
            p_over, p_under = share * (1 - push), (1 - share) * (1 - push)
        home_fav = (r["close_home_ml"] < 0) if pd.notna(r["close_home_ml"]) else (ph >= 0.5)
        pl_line = -1.5 if home_fav else 1.5              # the favourite lays 1.5 goals
        if home_fav:                                      # favourite must win by 2+
            p_fav_pl = pr["p_home_cover_m15"][i]
        else:
            p_fav_pl = m.game_probs([r["lam_a"]], [r["lam_h"]])["p_home_cover_m15"][0]
        p_fav_pl = float(m.platt(np.array([p_fav_pl]), pl_calib)[0]) if pl_calib else p_fav_pl
        p_home_pl = p_fav_pl if home_fav else 1 - p_fav_pl
        ag, hg = starters[(r["game_id"], r["away"])], starters[(r["game_id"], r["home"])]
        rows_out.append({
            "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "date": r["date"].date().isoformat(),
            "game_id": r["game_id"], "start_utc": r["start_utc"], "away": r["away"], "home": r["home"],
            "away_goalie": ag[1], "home_goalie": hg[1],
            "goalie_status": "confirmed" if ("confirmed" in ag[2] and "confirmed" in hg[2]) else "projected",
            "proj_away": round(float(pr["exp_away"][i]), 2), "proj_home": round(float(pr["exp_home"][i]), 2), "p_home_win": round(ph, 4),
            "fair_home_ml": fair_ml(ph).item(), "fair_away_ml": fair_ml(1 - ph).item(),
            "home_ml": r["close_home_ml"], "away_ml": r["close_away_ml"], "open_home_ml": r["open_home_ml"], "open_away_ml": r["open_away_ml"],
            "p_mkt_home": r["p_mkt_home"], "pick": r["home"] if pick_home else r["away"],
            "pick_prob": round(ph if pick_home else 1 - ph, 4), "pick_ml": r["close_home_ml"] if pick_home else r["close_away_ml"],
            "value_side": value_side, "value_edge": value_edge, "value_ml": value_ml,
            "total_line": r["close_total"], "proj_total": round(float(pr["exp_home"][i] + pr["exp_away"][i]), 2), "p_over": p_over,
            "total_lean": lean_by_value(p_over, p_under, r.get("close_over_odds"), r.get("close_under_odds"), "Over", "Under"),
            "pl_home_line": pl_line, "p_home_pl": round(p_home_pl, 4),
            "pl_lean": lean_by_value(p_home_pl, 1 - p_home_pl, r.get("close_home_pl_odds"), r.get("close_away_pl_odds"),
                                     f"{r['home']} {pl_line:+.1f}", f"{r['away']} {-pl_line:+.1f}"),
            # display-only details for the web page (not kept in the ledger)
            "p_under": p_under, "over_odds": r.get("close_over_odds"), "under_odds": r.get("close_under_odds"),
            "home_pl_odds": r.get("close_home_pl_odds"), "away_pl_odds": r.get("close_away_pl_odds"),
            "home_name": r.get("home_name"), "away_name": r.get("away_name"), "home_logo": r.get("home_logo"),
            "away_logo": r.get("away_logo"), "home_color": r.get("home_color"), "away_color": r.get("away_color"),
            "tv": r.get("tv"), "venue": r.get("venue"), "matchup_stats": team_stats.get(r["game_id"]),
            "p_fav_pl": p_fav_pl, "home_fav": bool(home_fav),
        })
    picks = pd.DataFrame(rows_out)
    OUT.mkdir(exist_ok=True)
    for d, sub in picks.groupby("date"):
        sub.to_csv(OUT / f"picks_{d}.csv", index=False)
    n = record(picks)
    # line snapshot: today's prices at the time of this run (builds our own open -> close record for testing timing)
    snap = picks[pd.to_datetime(picks["start_utc"], utc=True) > pd.Timestamp.now(tz="UTC")]
    snap = snap.assign(snapshot_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))[
        ["snapshot_at", "date", "game_id", "start_utc", "away", "home", "away_goalie", "home_goalie", "goalie_status",
         "home_ml", "away_ml", "open_home_ml", "open_away_ml", "total_line", "over_odds", "under_odds", "pl_home_line",
         "home_pl_odds", "away_pl_odds", "p_home_win", "proj_away", "proj_home"]]
    if len(snap):
        f = OUT / "line_snapshots.csv"
        snap.to_csv(f, mode="a", header=not f.exists(), index=False)
    led = load_ledger()
    history_csv(led)
    print_board(picks)
    print(f"\n{n} picks logged to output/ledger.csv (games that have started are locked). History: output/history.csv")
    import nhl_site
    print(f"web page: {nhl_site.build().resolve()}")


def print_board(p: pd.DataFrame):
    for d, sub in p.groupby("date"):
        print(f"\n{'=' * 118}\nNHL {d}\n{'=' * 118}")
        print(f"{'matchup':12s} {'goalies (away / home)':34s} {'proj':>9s} {'pick':>11s} {'fair ML':>8s} {'book ML':>8s} "
              f"{'value':>16s} {'total':>12s} {'puck line':>11s}")
        for _, r in sub.iterrows():
            book = "" if pd.isna(r["pick_ml"]) else f"{r['pick_ml']:+.0f}"
            fair = r["fair_home_ml"] if r["pick"] == r["home"] else r["fair_away_ml"]
            val = f"{r['value_side']} {r['value_ml']:+.0f} ({r['value_edge']:+.1%})" if r["value_side"] else ""
            tot = f"{r['total_lean']} {r['total_line']:.1f}" if r["total_lean"] else f"proj {r['proj_total']:.1f}"
            gl = f"{str(r['away_goalie'])[:16]} / {str(r['home_goalie'])[:16]}"
            print(f"{r['away']} @ {r['home']:6s} {gl:34s} {r['proj_away']:.1f}-{r['proj_home']:.1f}  "
                  f"{r['pick']:>4s} {r['pick_prob']:5.1%} {fair:+8.0f} {book:>8s} {val:>16s} {tot:>12s} {r['pl_lean']:>11s}")
        print(f"goalies: {sub['goalie_status'].iloc[0]} - put confirmed starters in goalie_overrides.csv and re-run before puck drop.")
        print(f"value = model's win chance beats the book's by {EDGE:.0%}+. In testing it only paid when bet EARLY (near the opening line).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    run(ap.parse_args().days)
