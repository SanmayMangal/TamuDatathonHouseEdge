"""E2. Which single lags matter?

For each lag k = 1..12, condition each machine's next payout on its own payout k ticks earlier
(leave-one-out count model, shrunk toward the machine's overall rates), and measure the gain
over the machine's overall rates alone. Three versions per lag:
  machine : a separate table per machine (allows machine-specific and opposite effects)
  type    : one table per machine type (assumes machines of a type share the effect)
  null    : the machine version after shuffling the lag within each machine (no real signal)
Lags 9-12 are a second control: the docs say effects stop at lag 8.
"""
from .common import gain_rows, lag, loo_conditional, loo_machine_base, permute_within
import numpy as np
import pandas as pd

QUESTION = "Which single lags carry signal, and is it machine-specific?"
SHOW = ["summary", "machine_dR_by_condition"]
LAGS = range(1, 13)
ENCS = ("win", "full")
STRENGTH = 10.0


def run(ctx):
    base = loo_machine_base(ctx)
    type_code = pd.factorize(ctx.hist["machine_type"].astype(str))[0]
    rows = []
    for enc in ENCS:
        for k in LAGS:
            x = lag(ctx, k, enc)
            pm, valid = loo_conditional(ctx.y, [x], base, STRENGTH, group=ctx.machine)
            rows += gain_rows(ctx, base, pm, valid, lag=k, enc=enc, version="machine")
            pt, _ = loo_conditional(ctx.y, [x], base, STRENGTH, group=type_code)
            rows += gain_rows(ctx, base, pt, valid, lag=k, enc=enc, version="type")
            xs = x.copy()
            xs[valid] = permute_within(x[valid], ctx.machine[valid], ctx.rng)
            pn, _ = loo_conditional(ctx.y, [xs], base, STRENGTH, group=ctx.machine)
            rows += gain_rows(ctx, base, pn, valid, lag=k, enc=enc, version="null")
    detail = pd.DataFrame(rows)
    summary = (detail[detail["condition"] == "all"]
               .pivot_table(index=["enc", "lag"], columns="version", values="dR")
               .reset_index())
    summary.columns.name = None
    summary = summary.rename(columns={"machine": "dR_machine", "type": "dR_type", "null": "dR_null"})
    by_condition = (detail[detail["version"] == "machine"]
                    .pivot_table(index=["enc", "lag"], columns="condition", values="dR").reset_index())
    by_condition.columns.name = None
    return {"summary": summary, "machine_dR_by_condition": by_condition, "detail": detail}
