"""E4. Are the effects combinations of lags?

For every pair of lags (i, j) with 1 <= i < j <= 8, per machine (leave-one-out):
  single_i, single_j : condition on one lag
  naive              : combine the two single-lag effects as if independent
                       (p proportional to base * (p_i / base) * (p_j / base))
  joint              : condition on the exact pair of values
  null               : joint, after shuffling both lags within the machine
interaction = joint gain - max(naive, best single). A clearly positive interaction means
the model needs the pair itself, not just each lag on its own.
"""
from itertools import combinations

import numpy as np
import pandas as pd

from .common import gain_rows, lag, loo_conditional, loo_machine_base, permute_within

QUESTION = "Do pairs of lags carry signal beyond each lag on its own?"
SHOW = ["top_pairs", "interaction_by_condition"]
ENCS = ("win", "cat3")
STRENGTH = 10.0


def run(ctx):
    base = loo_machine_base(ctx)
    rows = []
    for enc in ENCS:
        X = {k: lag(ctx, k, enc) for k in range(1, 9)}
        single = {k: loo_conditional(ctx.y, [X[k]], base, STRENGTH, group=ctx.machine)[0] for k in X}
        for i, j in combinations(range(1, 9), 2):
            pj, valid = loo_conditional(ctx.y, [X[i], X[j]], base, STRENGTH, group=ctx.machine)
            naive = base * (single[i] / base) * (single[j] / base)
            naive /= naive.sum(1, keepdims=True)
            xi, xj = X[i].copy(), X[j].copy()
            xi[valid] = permute_within(X[i][valid], ctx.machine[valid], ctx.rng)
            xj[valid] = permute_within(X[j][valid], ctx.machine[valid], ctx.rng)
            pn, _ = loo_conditional(ctx.y, [xi, xj], base, STRENGTH, group=ctx.machine)
            for name, p in (("single_i", single[i]), ("single_j", single[j]), ("naive", naive),
                            ("joint", pj), ("null", pn)):
                rows += gain_rows(ctx, base, p, valid, enc=enc, lag_i=i, lag_j=j, variant=name)
    detail = pd.DataFrame(rows)
    wide = detail.pivot_table(index=["enc", "lag_i", "lag_j", "condition"], columns="variant",
                              values="dR").reset_index()
    wide.columns.name = None
    wide["best_single"] = wide[["single_i", "single_j"]].max(axis=1)
    wide["interaction"] = wide["joint"] - np.maximum(wide["naive"], wide["best_single"])
    top = (wide[wide["condition"] == "all"].sort_values("joint", ascending=False)
           .groupby("enc").head(10).reset_index(drop=True))
    inter = (wide.pivot_table(index=["enc", "lag_i", "lag_j"], columns="condition", values="interaction")
             .reset_index())
    inter.columns.name = None
    inter = inter.sort_values("all", ascending=False).groupby("enc").head(10).reset_index(drop=True)
    return {"top_pairs": top, "interaction_by_condition": inter, "pairs_dR": wide, "detail": detail}
