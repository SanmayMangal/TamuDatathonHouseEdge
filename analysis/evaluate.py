"""Competition-style scoring for the example table.

Mirrors the kit's contest score: squared probability error per row, averaged
within each episode, then across episodes in each condition, then divided by
the frozen pooled reference's error for that condition. R is the mean of the
four ratios; points = max_points / (1 + R).

Two ways to score:
  against="target"  : validation query rows vs the true probabilities (q_*). This is the
                      exact competition number for validation.
  against="outcome" : any rows with an observed outcome (e.g. all history rows), vs the
                      one-hot outcome. E[outcome error] = target error + noise, where the
                      noise term (1 - sum q^2) doesn't depend on the forecast, so rankings
                      within a condition match the competition. Raw ratios are pulled
                      toward 1 by that noise; pass noise= to subtract a per-condition estimate.
"""
import numpy as np
import pandas as pd

PAYOUTS = (0, 1, 2, 5, 20)
P_COLS = [f"p_{p}" for p in PAYOUTS]
REF_COLS = ["ref_" + c for c in P_COLS]
Q_COLS = ["q" + c[1:] for c in P_COLS]
CONDITIONS = ("stable", "short_history", "drift", "shared_change")


def onehot(outcome):
    out = np.zeros((len(outcome), len(PAYOUTS)))
    idx = pd.Series(outcome).map({p: i for i, p in enumerate(PAYOUTS)}).to_numpy()
    out[np.arange(len(outcome)), idx.astype(int)] = 1
    return out


def row_errors(pred, truth):
    return ((np.asarray(pred, float) - np.asarray(truth, float)) ** 2).sum(1)


def condition_errors(df, err):
    """Mean of per-episode mean errors, per condition."""
    per_ep = pd.DataFrame({"episode_id": df["episode_id"].to_numpy(), "condition": df["condition"].astype(str).to_numpy(),
                           "err": err}).groupby(["condition", "episode_id"])["err"].mean()
    return per_ep.groupby(level=0).mean()


def score(df, pred, against="outcome", noise=None, max_points=60.0):
    """df: rows of the example table; pred: (n, 5) array aligned with df.

    Returns a dict with per-condition errors, reference errors, ratios, R and points.
    """
    pred = np.asarray(pred, float)
    assert pred.shape == (len(df), 5), pred.shape
    assert np.allclose(pred.sum(1), 1, atol=1e-6), "rows must sum to 1"
    truth = df[Q_COLS].to_numpy() if against == "target" else onehot(df["outcome"].to_numpy())
    if np.isnan(truth).any():
        raise ValueError("Missing truth: use target scoring only on validation query rows.")
    e = condition_errors(df, row_errors(pred, truth))
    b = condition_errors(df, row_errors(df[REF_COLS].to_numpy(), truth))
    if noise is not None:
        nz = pd.Series(noise).reindex(e.index)
        e, b = e - nz, b - nz
    ratio = e / np.maximum(b, 1e-9)
    R = float(ratio.mean())
    return {"error": e.to_dict(), "reference_error": b.to_dict(), "ratio": ratio.to_dict(),
            "R": R, "points": max_points / (1 + R), "n_rows": len(df)}


def noise_by_condition(df):
    """Per-condition mean of 1 - sum(q^2) on validation query rows: the irreducible part of
    outcome-based error. Use as noise= when scoring against outcomes."""
    v = df[df["is_query"] & df[Q_COLS[0]].notna()]
    nz = 1 - (v[Q_COLS].to_numpy() ** 2).sum(1)
    return condition_errors(v, nz).to_dict()


def report(result, name=""):
    parts = " | ".join(f"{c} {result['ratio'][c]:.3f}" for c in CONDITIONS if c in result["ratio"])
    return f"{name:24s} points {result['points']:6.2f}  R {result['R']:.4f}  [{parts}]"


def folds(df, k):
    """Yield (train_mask, test_mask) over train+validation rows, whole episodes per fold."""
    labeled = df["fold"] >= 0
    for f in range(k):
        yield labeled & (df["fold"] != f), labeled & (df["fold"] == f)
