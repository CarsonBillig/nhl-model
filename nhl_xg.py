"""Our own expected-goals (xG) model, trained on NHL play-by-play (no third-party xG).

xG = the chance an unblocked shot becomes a goal, from distance, angle, shot type, rebound, rush and strength
(5-on-5, power play, ...). Empty-net shots are left out (those goals are modelled separately).

Walk-forward: each season's shots are valued by a model trained only on EARLIER seasons (recent seasons weighted
more), so the backtest never sees the future and the model keeps up with changes in how the NHL logs shots (close-range
shots and rebounds are recorded far more often since 2020-21). The first season (2020-21) is valued by a model fit on
itself; it is only ever used as training data, never tested.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import nhl_data

MODELS = nhl_data.CACHE / "xg"
MODELS.mkdir(exist_ok=True)
FIRST = nhl_data.FIRST_SEASON
RECENCY = 0.5                    # a season's weight halves for each season further back
SHOT_TYPES = ["wrist", "snap", "slap", "backhand", "tip-in", "deflected", "wrap-around", "bat", "poke", "between-legs",
              "cradle", "unknown"]
FEATURES = ["distance", "angle", "shot_type_code", "rebound", "rush", "own_sk", "opp_sk", "ot"]


def _prep(ev: pd.DataFrame) -> pd.DataFrame:
    x = ev[ev["kind"].isin(["sog", "miss", "goal"]) & (ev["empty_net"] == 0) & ev["distance"].notna()].copy()
    x["shot_type_code"] = x["shot_type"].map({s: i for i, s in enumerate(SHOT_TYPES)}).fillna(len(SHOT_TYPES) - 1).astype(int)
    x["ot"] = (x["period"] > 3).astype(int)
    x["is_goal"] = (x["kind"] == "goal").astype(int)
    return x


def _season_shots(season: int) -> pd.DataFrame:
    import nhl_pbp
    f = nhl_pbp.PBP / f"events_{season}.parquet"
    if not f.exists():
        return pd.DataFrame()
    ev = pd.read_parquet(f)
    return _prep(ev[(ev["game_id"] // 1_000_000) == season])


def train_for(season: int) -> HistGradientBoostingClassifier:
    """The model used to value `season`'s shots: fit on all earlier seasons (recent ones weighted more)."""
    past = list(range(FIRST, season)) or [FIRST]
    parts = [_season_shots(s).assign(w=RECENCY ** max(season - 1 - s, 0)) for s in past]
    x = pd.concat([p for p in parts if len(p)], ignore_index=True)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=80,
                                       l2_regularization=1.0, categorical_features=[FEATURES.index("shot_type_code")],
                                       random_state=7)
    m.fit(x[FEATURES], x["is_goal"], sample_weight=x["w"])
    joblib.dump(m, MODELS / f"xg_{season}.joblib")
    return m


def model_for(season: int) -> HistGradientBoostingClassifier:
    f = MODELS / f"xg_{season}.joblib"
    return joblib.load(f) if f.exists() else train_for(season)


def add_xg(events: pd.DataFrame) -> pd.DataFrame:
    """Adds column `xg` (0 for blocked shots, empty-net shots and penalties), each season valued by its own model."""
    out = events.reset_index(drop=True)
    out["xg"] = 0.0
    x = _prep(out)
    for s, idx in x.groupby("season").groups.items():
        out.loc[idx, "xg"] = model_for(int(s)).predict_proba(x.loc[idx, FEATURES])[:, 1]
    return out


def evaluate(seasons) -> pd.DataFrame:
    """Out-of-sample check by season: total xG vs goals and log loss vs a no-skill baseline."""
    rows = []
    for s in seasons:
        x = _season_shots(s)
        if not len(x):
            continue
        p = np.clip(model_for(s).predict_proba(x[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
        y = x["is_goal"].to_numpy()
        base = np.full(len(y), y.mean())
        ll = lambda q: float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())
        rows.append({"season": s, "shots": len(y), "goals": int(y.sum()), "xG": round(float(p.sum()), 1),
                     "log_loss": round(ll(p), 4), "no_skill_log_loss": round(ll(base), 4), "out_of_sample": s > FIRST})
    return pd.DataFrame(rows)
