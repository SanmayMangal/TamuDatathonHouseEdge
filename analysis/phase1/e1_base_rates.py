"""E1. How do machine type and signal shift the odds?

Splits each (type, signal) cell's payouts into the chance of a zero, the average return,
and the size mix given a nonzero payout. Then tests whether machines inside a cell differ
by more than chance (overdispersion), which says whether type x signal is enough of a prior
or each machine needs its own rates.
"""
import numpy as np
import pandas as pd

from .common import PAYOUTS

QUESTION = "How do type and signal shift the odds, and do machines differ beyond type x signal?"
SHOW = ["by_type_signal", "overdispersion", "by_condition_type"]


def _rates(g):
    out = g["outcome"]
    n = len(out)
    win = out[out > 0]
    row = {"n_rows": n, "n_machines": g["machine_id"].nunique()}
    row.update({f"P({p})": (out == p).mean() for p in PAYOUTS})
    row["mean_return"] = out.mean()
    row.update({f"P({p}|>0)": (win == p).mean() if len(win) else np.nan for p in PAYOUTS[1:]})
    return pd.Series(row)


def run(ctx):
    h = ctx.hist[["machine_id", "machine_type", "signal", "condition", "outcome"]].copy()
    h["machine_type"] = h["machine_type"].astype(str)
    by_cell = h.groupby(["machine_type", "signal"]).apply(_rates, include_groups=False).reset_index()
    by_cond = h.groupby(["condition", "machine_type"]).apply(_rates, include_groups=False).reset_index()

    # Overdispersion: variance of machine-level rates vs what binomial sampling alone gives.
    m = h.groupby(["machine_type", "signal", "machine_id"], observed=True)["outcome"].agg(
        n="size", p0=lambda x: (x == 0).mean(), p20=lambda x: (x == 20).mean(), ret="mean")
    rows = []
    for (t, s), g in m.groupby(level=[0, 1]):
        for col in ("p0", "p20"):
            p = (g[col] * g["n"]).sum() / g["n"].sum()
            expected = (p * (1 - p) / g["n"]).mean()
            rows.append({"machine_type": t, "signal": s, "rate": col, "pooled": p,
                         "sd_across_machines": g[col].std(ddof=1),
                         "sd_if_identical": np.sqrt(expected),
                         "dispersion_ratio": g[col].var(ddof=1) / expected if expected > 0 else np.nan,
                         "n_machines": len(g)})
    disp = pd.DataFrame(rows)
    return {"by_type_signal": by_cell, "overdispersion": disp, "by_condition_type": by_cond,
            "per_machine_rates": m.reset_index()}
