"""E8. What do short histories leave available?

Short-history machines have 64 observed ticks: 40 consecutive ticks just before the cutoff
plus 24 older ticks sampled from earlier. This experiment:
  layout    : confirms that layout and counts how many ticks have all of lags 1..8 observed
              (usable self-supervised examples for lag models).
  estimators: compares ways to estimate a short-history machine's odds at the cutoff, scored
              exactly on validation targets (24 episodes) and on train labels (noisier):
              the prior strength, using the recent block only, down-weighting the older
              sampled ticks, and recency decay.
"""
import numpy as np
import pandas as pd

from .common import K, PAYOUTS, history_grid
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import Q_COLS, condition_errors  # noqa: E402

QUESTION = "How should short histories be used: how much to trust the 64 ticks vs the prior?"
SHOW = ["layout", "estimators"]
STRENGTHS = (5.0, 10.0, 20.0, 40.0, 80.0)
OLD_WEIGHTS = (0.0, 0.25, 0.5, 1.0)
HALF_LIVES = (8, 16, 32, 64, None)


def _forecast(Y, prior, cutoff, strength, old_weight=1.0, half_life=None, recent=40):
    M, T = Y.shape
    P = np.empty((M, K))
    for m in range(M):
        obs = np.flatnonzero(Y[m, : cutoff[m]] >= 0)
        age = cutoff[m] - 1 - obs
        w = np.where(age < recent, 1.0, old_weight)
        if half_life is not None:
            w = w * 2.0 ** (-age / half_life)
        c = np.zeros(K)
        np.add.at(c, Y[m, obs], w)
        P[m] = (strength * prior[m] + c) / (strength + c.sum())
    return P


def _score(pred, rows, truth):
    err = ((pred - truth) ** 2).sum(1)
    return condition_errors(rows, err).get("short_history", np.nan)


def run(ctx):
    Y, meta, prior = history_grid(ctx)
    short = meta["short_history"].to_numpy()
    Ys, ms, ps = Y[short], meta[short].reset_index(drop=True), prior[short]
    cutoff = ms["cutoff"].to_numpy()

    lay = []
    for m in range(len(ms)):
        obs = np.flatnonzero(Ys[m] >= 0)
        recent = obs[obs >= cutoff[m] - 40]
        old = obs[obs < cutoff[m] - 40]
        have = set(obs)
        full_lags = sum(all((t - k) in have for k in range(1, 9)) for t in obs)
        lay.append({"n_obs": len(obs), "n_recent": len(recent),
                    "recent_consecutive": len(recent) == 40 and recent.min() == cutoff[m] - 40,
                    "n_old": len(old), "oldest_tick": obs.min() if len(obs) else np.nan,
                    "ticks_with_lags_1to8": full_lags})
    layout = pd.DataFrame(lay).describe().T.reset_index().rename(columns={"index": "quantity"})

    # Truth: validation targets (exact) and train labels (one-hot outcome, noisy).
    vmap = ctx.vq.set_index(ctx.vq["machine_id"].astype(str))
    tmap = ctx.tq.set_index(ctx.tq["machine_id"].astype(str))
    ids = ms["machine_id"].to_numpy()
    in_v = np.isin(ids, vmap.index)
    in_t = np.isin(ids, tmap.index)
    q_v = vmap.loc[ids[in_v], Q_COLS].to_numpy()
    y_t = np.searchsorted(PAYOUTS, tmap.loc[ids[in_t], "outcome"].to_numpy().astype(int))
    oh_t = np.eye(K)[y_t]
    rows_v, rows_t = ms[in_v], ms[in_t]

    est = []

    def add(name, P, **params):
        est.append({"estimator": name, **params,
                    "exact_error_validation": _score(P[in_v], rows_v, q_v) if in_v.any() else np.nan,
                    "outcome_error_train": _score(P[in_t], rows_t, oh_t) if in_t.any() else np.nan})

    for s in STRENGTHS:
        add("all 64 ticks", _forecast(Ys, ps, cutoff, s), strength=s)
        add("recent 40 only", _forecast(Ys, ps, cutoff, s, old_weight=0.0), strength=s)
    for w in OLD_WEIGHTS:
        add("older ticks weighted", _forecast(Ys, ps, cutoff, 20.0, old_weight=w), strength=20.0, old_weight=w)
    for hl in HALF_LIVES:
        add("recency decay", _forecast(Ys, ps, cutoff, 20.0, half_life=hl), strength=20.0, half_life=hl)
    estimators = pd.DataFrame(est)
    ref = estimators.query("estimator == 'all 64 ticks' and strength == 20.0")
    for col in ("exact_error_validation", "outcome_error_train"):
        estimators[col.replace("error", "ratio_vs_ref")] = estimators[col] / ref[col].iloc[0]
    return {"layout": layout, "estimators": estimators}
