"""Shared machinery for the Phase 1 experiments.

Most experiments ask "does feature X help predict the next payout?" They answer it
the same way: a count model that conditions on X, compared against a baseline on the
same rows, both scored leave-one-out (each row's own outcome is removed from the
counts before predicting it). Leave-one-out keeps the comparison honest without
refitting per fold, and it is exact for count models, so it's fast.

Gains are reported in Brier units (sum of squared errors over the 5 payouts) on observed
outcomes: positive = the feature helps. On outcomes, a gain of g lowers the true
squared error by g in expectation, so `dR` = g / (reference's true error in that
condition) approximates how much the competition ratio R would drop.
"""
from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import CONDITIONS, Q_COLS, REF_COLS, score  # noqa: E402

PAYOUTS = np.array([0, 1, 2, 5, 20])
K = len(PAYOUTS)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE = ROOT / "analysis/artifacts/examples.pkl"

ENCODINGS = {
    "win": lambda x: np.where(np.isnan(x), np.nan, (x > 0).astype(float)),          # 0 vs any return
    "profit": lambda x: np.where(np.isnan(x), np.nan, (x > 1).astype(float)),       # 0/1 vs 2+
    "cat3": lambda x: np.where(np.isnan(x), np.nan, np.minimum(x, 2).astype(float)),  # 0 / 1 / 2+ (starter)
    "full": lambda x: np.where(np.isnan(x), np.nan, np.searchsorted(PAYOUTS, np.nan_to_num(x)).astype(float)),
    "big": lambda x: np.where(np.isnan(x), np.nan, (x >= 5).astype(float)),         # 5x or 20x
}


@dataclass
class Context:
    """Everything an experiment needs. Built once by load()."""
    hist: pd.DataFrame          # labeled history rows (train + validation), sorted
    vq: pd.DataFrame            # validation query rows (true probabilities available)
    tq: pd.DataFrame            # train query rows (observed labels only)
    y: np.ndarray               # payout index 0..4 for each hist row
    machine: np.ndarray         # integer machine code for each hist row
    prior: np.ndarray           # (n, 5) type x signal prior for each hist row
    ref_true_error: dict        # reference's exact error per condition (validation targets)
    smoke: bool = False
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))


def load(table=DEFAULT_TABLE, smoke=False, episodes_per_condition=2):
    df = pd.read_pickle(table)
    vq_full = df[(df["split"] == "validation") & df["is_query"]]
    ref_true = score(vq_full, vq_full[REF_COLS].to_numpy(), against="target")["reference_error"]
    labeled = df[df["split"].isin(["train", "validation"])].copy()
    if smoke:
        keep = (labeled.groupby(["split", "condition"], observed=True)["episode_id"]
                .apply(lambda s: list(pd.unique(s))[:episodes_per_condition]))
        labeled = labeled[labeled["episode_id"].isin([e for eps in keep for e in eps])]
    labeled["condition"] = labeled["condition"].astype(str)
    hist = labeled[~labeled["is_query"]].sort_values(["machine_id", "tick"]).reset_index(drop=True)
    vq = labeled[(labeled["split"] == "validation") & labeled["is_query"]].reset_index(drop=True)
    tq = labeled[(labeled["split"] == "train") & labeled["is_query"]].reset_index(drop=True)
    y = np.searchsorted(PAYOUTS, hist["outcome"].to_numpy().astype(int))
    machine = pd.factorize(hist["machine_id"].astype(str))[0]
    return Context(hist=hist, vq=vq, tq=tq, y=y, machine=machine,
                   prior=context_prior(hist, y, hist), ref_true_error=ref_true, smoke=smoke)


def onehot(y):
    out = np.zeros((len(y), K))
    out[np.arange(len(y)), y] = 1
    return out


def context_prior(fit_rows, fit_y, rows):
    """Payout distribution per (machine_type, signal), from fit_rows; mapped onto rows."""
    key_fit = fit_rows["machine_type"].astype(str) + "|" + fit_rows["signal"].astype(str)
    counts = pd.DataFrame(onehot(fit_y)).groupby(key_fit.to_numpy()).sum()
    probs = (counts + 0.5).div(counts.sum(1) + 0.5 * K, axis=0)
    key = rows["machine_type"].astype(str) + "|" + rows["signal"].astype(str)
    return probs.reindex(key.to_numpy()).to_numpy()


def group_codes(columns):
    """Integer group id for each row from a list of equal-length arrays (no NaNs)."""
    return pd.DataFrame({i: c for i, c in enumerate(columns)}).groupby(list(range(len(columns))), sort=False).ngroup().to_numpy()


def loo_conditional(y, keys, prior, strength, group=None):
    """Leave-one-out Dirichlet count model.

    For each row: p = (counts in its cell, minus the row itself, + strength * prior) / (n - 1 + strength).
    keys: list of arrays (NaN = unknown); the cell is (group, *keys). group is usually the machine.
    Rows with any unknown key get the prior. Returns (probs, valid_mask).
    """
    n = len(y)
    valid = np.ones(n, bool)
    for k in keys:
        valid &= ~np.isnan(k)
    cols = ([group[valid]] if group is not None else []) + [k[valid] for k in keys]
    out = np.array(prior, float, copy=True)
    if not valid.any():
        return out, valid
    yv = y[valid]
    g = group_codes(cols) if cols else np.zeros(valid.sum(), int)
    counts = np.zeros((g.max() + 1, K))
    np.add.at(counts, (g, yv), 1)
    own = counts[g] - onehot(yv)
    out[valid] = (own + strength * prior[valid]) / (own.sum(1, keepdims=True) + strength)
    return out, valid


def loo_machine_base(ctx, strength=20.0):
    """Each machine's own payout frequencies (leave-one-out), shrunk to its type x signal prior."""
    return loo_conditional(ctx.y, [], ctx.prior, strength, group=ctx.machine)[0]


def permute_within(values, group, rng):
    """Shuffle values within each group: keeps each machine's feature mix, breaks timing."""
    # ctx.hist is sorted by machine, so each group is one contiguous block in ascending code
    # order; sorting by (group, random) keeps the blocks in place and shuffles inside them.
    assert np.all(np.diff(group) >= 0), "rows must be grouped by machine"
    return values[np.lexsort((rng.random(len(values)), group))]


def brier(p, y):
    return ((p - onehot(y)) ** 2).sum(1)


def logloss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))


def gain_rows(ctx, base, model, mask=None, **labels):
    """Per-condition gain of model over base: mean over episodes of the episode-mean gain.

    Returns tidy rows: labels..., condition, gain (Brier), se, gain_logloss, dR, n_episodes, n_rows.
    'all' averages the four conditions equally, like the competition.
    """
    mask = np.ones(len(ctx.y), bool) if mask is None else mask
    d = pd.DataFrame({
        "episode_id": ctx.hist["episode_id"].to_numpy()[mask],
        "condition": ctx.hist["condition"].to_numpy()[mask],
        "g": (brier(base, ctx.y) - brier(model, ctx.y))[mask],
        "gl": (logloss(base, ctx.y) - logloss(model, ctx.y))[mask],
    })
    ep = d.groupby(["condition", "episode_id"]).agg(g=("g", "mean"), gl=("gl", "mean"), n=("g", "size"))
    rows = []
    for cond, e in ep.groupby(level=0):
        rows.append({**labels, "condition": cond, "gain": e["g"].mean(),
                     "se": e["g"].std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else np.nan,
                     "gain_logloss": e["gl"].mean(),
                     "dR": e["g"].mean() / ctx.ref_true_error.get(cond, np.nan),
                     "n_episodes": len(e), "n_rows": int(e["n"].sum())})
    if rows:
        conds = pd.DataFrame(rows)
        rows.append({**labels, "condition": "all", "gain": conds["gain"].mean(),
                     "se": np.sqrt((conds["se"] ** 2).sum()) / len(conds),
                     "gain_logloss": conds["gain_logloss"].mean(), "dR": conds["dR"].mean(),
                     "n_episodes": int(conds["n_episodes"].sum()), "n_rows": int(conds["n_rows"].sum())})
    return rows


def lag(ctx, k, enc="win", who="own"):
    col = f"own_lag{k}" if who == "own" else f"{who}_lag{k}"
    return ENCODINGS[enc](ctx.hist[col].to_numpy().astype(float))


# ---------- Online (causal) predictors over a machine x tick grid ----------

def history_grid(ctx):
    """Y[m, t] = payout index at tick t for machine m, or -1 if unobserved. Plus per-machine metadata."""
    h = ctx.hist
    meta = (h.groupby("machine_id", observed=True, sort=True)
            .agg(episode_id=("episode_id", "first"), condition=("condition", "first"),
                 machine_type=("machine_type", "first"), signal=("signal", "first"),
                 cutoff=("cutoff", "first"), split=("split", "first"),
                 short_history=("short_history", "first"))
            .reset_index())
    meta["machine_id"] = meta["machine_id"].astype(str)
    m = pd.Index(meta["machine_id"]).get_indexer(h["machine_id"].astype(str))
    T = int(meta["cutoff"].max()) + 1
    Y = np.full((len(meta), T), -1, int)
    Y[m, h["tick"].to_numpy()] = ctx.y
    first = h.groupby("machine_id", observed=True, sort=True).head(1).index
    meta_prior = ctx.prior[first]
    return Y, meta, meta_prior


def online_decay(Y, prior, strength, half_life):
    """Causal forecast at every tick: (strength * prior + decayed counts of earlier outcomes) / total.

    half_life=None means no decay (all past outcomes weigh the same). Returns P[m, t, 5].
    """
    M, T = Y.shape
    decay = 1.0 if half_life is None else 2.0 ** (-1.0 / half_life)
    c = np.zeros((M, K))
    P = np.empty((M, T, K))
    base = strength * prior
    for t in range(T):
        P[:, t] = (base + c) / (strength + c.sum(1, keepdims=True))
        c *= decay
        obs = Y[:, t] >= 0
        c[obs, Y[obs, t]] += 1
    return P


def grid_gain_rows(ctx, meta, Y, P_base, P_model, min_tick=0, **labels):
    """gain_rows() for grid predictions: scores every observed cell with tick >= min_tick."""
    mm, tt = np.nonzero(Y >= 0)
    keep = tt >= min_tick
    mm, tt = mm[keep], tt[keep]
    y = Y[mm, tt]
    fake = type("G", (), {})()
    fake.y = y
    fake.hist = pd.DataFrame({"episode_id": meta["episode_id"].to_numpy()[mm],
                              "condition": meta["condition"].to_numpy()[mm]})
    fake.ref_true_error = ctx.ref_true_error
    return gain_rows(fake, P_base[mm, tt], P_model[mm, tt], **labels)


def cutoff_predictions(meta, P):
    """Forecast at each machine's cutoff, from a grid predictor."""
    return P[np.arange(len(meta)), meta["cutoff"].to_numpy()]


def exact_score(ctx, meta, pred_by_machine, **labels):
    """Exact competition score on validation queries for per-machine cutoff forecasts."""
    vq = ctx.vq
    if len(vq) == 0:
        return []
    idx = pd.Index(meta["machine_id"]).get_indexer(vq["machine_id"].astype(str))
    if (idx < 0).any():
        return []
    res = score(vq, pred_by_machine[idx], against="target")
    out = [{**labels, "condition": c, "ratio": r} for c, r in res["ratio"].items()]
    out.append({**labels, "condition": "all", "ratio": res["R"], "points": res["points"]})
    return out
