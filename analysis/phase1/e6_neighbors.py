"""E6. Do the other machines in the episode help?

Three views, all leave-one-out per machine, against the machine's overall rates:
  pairwise : condition on one neighbor's payout (win/loss) at lag k, for each of the 7
             neighbors and k = 1..8; plus a shuffled-within-machine null for each.
  aggregate: condition on how many neighbors paid out at lag k (0-2, 3-4, 5-7), and on the
             share of same-type neighbors that paid out.
  coupling : for each machine, its single best (neighbor, lag) gain vs the best of its 56
             shuffled nulls. If real best gains clear the null for many machines, specific
             machine pairs are coupled (sparse links) rather than the whole episode moving together.
"""
import numpy as np
import pandas as pd

from .common import gain_rows, lag, loo_conditional, loo_machine_base, permute_within, brier

QUESTION = "Do neighbors' earlier payouts help, through specific pairs or the whole episode?"
SHOW = ["pairwise_by_lag", "aggregate", "coupling"]
STRENGTH = 10.0


def run(ctx):
    base = loo_machine_base(ctx)
    bb = brier(base, ctx.y)
    h = ctx.hist
    machine_ids = h["machine_id"].astype(str).to_numpy()
    rows, per_machine = [], []
    for s in range(1, 8):
        for k in range(1, 9):
            x = lag(ctx, k, "win", who=f"nb{s}")
            for version in ("real", "null"):
                xx = x.copy()
                ok = ~np.isnan(x)
                if version == "null":
                    xx[ok] = permute_within(x[ok], ctx.machine[ok], ctx.rng)
                p, valid = loo_conditional(ctx.y, [xx], base, STRENGTH, group=ctx.machine)
                rows += gain_rows(ctx, base, p, valid, slot=s, lag=k, version=version)
                g = pd.Series((bb - brier(p, ctx.y))[valid]).groupby(machine_ids[valid]).mean()
                per_machine.append(pd.DataFrame({"machine_id": g.index, "slot": s, "lag": k,
                                                 "version": version, "gain": g.values}))
    detail = pd.DataFrame(rows)
    by_lag = (detail.groupby(["version", "lag", "condition"])["dR"].mean().unstack("condition").reset_index())
    by_lag.columns.name = None

    # Aggregates over the 7 neighbors at each lag.
    agg_rows = []
    same_type = np.stack([(h[f"nb{s}_type"].astype(str) == h["machine_type"].astype(str)).to_numpy()
                          for s in range(1, 8)], 1)
    for k in range(1, 9):
        W = np.stack([lag(ctx, k, "win", who=f"nb{s}") for s in range(1, 8)], 1)
        known = ~np.isnan(W).any(1)
        wins = np.where(known, np.nansum(W, 1), np.nan)
        binned = np.where(np.isnan(wins), np.nan, np.digitize(wins, [3, 5]))
        p, valid = loo_conditional(ctx.y, [binned], base, STRENGTH, group=ctx.machine)
        agg_rows += gain_rows(ctx, base, p, valid, lag=k, feature="neighbor_wins_binned")
        st = np.where(known & same_type.any(1),
                      np.round(np.nansum(np.where(same_type, W, 0), 1) / np.maximum(same_type.sum(1), 1) * 2) / 2,
                      np.nan)
        p, valid = loo_conditional(ctx.y, [st], base, STRENGTH, group=ctx.machine)
        agg_rows += gain_rows(ctx, base, p, valid, lag=k, feature="same_type_win_share")
    agg = pd.DataFrame(agg_rows)
    agg = agg.pivot_table(index=["feature", "lag"], columns="condition", values="dR").reset_index()
    agg.columns.name = None

    pm = pd.concat(per_machine, ignore_index=True)
    best = pm.groupby(["machine_id", "version"])["gain"].max().unstack("version")
    cond = h.groupby("machine_id", observed=True)["condition"].first()
    cond.index = cond.index.astype(str)
    best["condition"] = cond.reindex(best.index).to_numpy()
    threshold = best["null"].quantile(0.95)
    coupling = best.groupby("condition").agg(
        machines=("real", "size"), median_best_real=("real", "median"), median_best_null=("null", "median"),
        frac_real_above_null95=("real", lambda r: (r > threshold).mean())).reset_index()
    return {"pairwise_by_lag": by_lag, "aggregate": agg, "coupling": coupling,
            "pairwise_detail": detail, "per_machine_pairs": pm}
