"""E5. How much payout detail matters at a lag?

For lags 1..8, condition on the lagged payout encoded five ways:
  win (0 vs >0), profit (0-1 vs 2+), cat3 (0 / 1 / 2+, the starter's), big (5x-20x vs rest),
  full (all five values).
Gains are split into the part that comes from the chance of a zero (binary Brier on
P(0)) and the part from the size mix among nonzero payouts (Brier on the renormalized
distribution over 1, 2, 5, 20, scored on nonzero rows). This says whether a lag mainly
moves win chance, payout size, or both.
"""
import numpy as np
import pandas as pd

from .common import ENCODINGS, gain_rows, lag, loo_conditional, loo_machine_base

QUESTION = "Which encoding of a lagged payout carries the signal, and does it move win chance or payout size?"
SHOW = ["summary"]
STRENGTH = 10.0


def _parts(p, y):
    zero = 2 * (p[:, 0] - (y == 0)) ** 2
    pos = p[:, 1:] / np.clip(p[:, 1:].sum(1, keepdims=True), 1e-12, None)
    oh = np.zeros_like(pos)
    nz = y > 0
    oh[np.arange(len(y))[nz], y[nz] - 1] = 1
    size = np.where(nz, ((pos - oh) ** 2).sum(1), np.nan)
    return zero, size


def run(ctx):
    base = loo_machine_base(ctx)
    bz, bs = _parts(base, ctx.y)
    rows, parts = [], []
    for k in range(1, 9):
        for enc in ENCODINGS:
            p, valid = loo_conditional(ctx.y, [lag(ctx, k, enc)], base, STRENGTH, group=ctx.machine)
            rows += gain_rows(ctx, base, p, valid, lag=k, enc=enc)
            z, s = _parts(p, ctx.y)
            parts.append({"lag": k, "enc": enc, "gain_zero_part": np.nanmean((bz - z)[valid]),
                          "gain_size_part": np.nanmean((bs - s)[valid])})
    detail = pd.DataFrame(rows)
    summary = (detail[detail["condition"] == "all"][["lag", "enc", "dR", "gain", "gain_logloss"]]
               .merge(pd.DataFrame(parts), on=["lag", "enc"]))
    return {"summary": summary, "detail": detail}
