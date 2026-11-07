"""E9. Where does the reference lose the most?

Uses the 768 validation queries, where the true probabilities are known:
  by_group      : reference error by condition x machine type, and by signal.
  by_payout     : which payout's probability carries the error.
  truth_shape   : how peaked the true odds are (max probability, entropy), and how far they
                  sit from the machine's long-run and recent frequencies. True odds far from
                  long-run frequencies mean something time-varying (lags, regimes) is moving them.
  ceilings      : error of in-sample "oracle" forecasts that know the average true odds per
                  (type, signal) or per (condition, type, signal). Not achievable; it shows how
                  much of the truth is explained by those labels versus by each machine's state.
"""
import numpy as np
import pandas as pd

from .common import K, history_grid
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import P_COLS, Q_COLS, REF_COLS, condition_errors  # noqa: E402

QUESTION = "Where is the reference's error concentrated, and how much is explainable?"
SHOW = ["by_group", "by_payout", "truth_shape", "ceilings"]


def run(ctx):
    v = ctx.vq.copy()
    if v.empty:
        return {}
    v["machine_type"] = v["machine_type"].astype(str)
    q, r = v[Q_COLS].to_numpy(), v[REF_COLS].to_numpy()
    v["ref_error"] = ((r - q) ** 2).sum(1)
    by_group = pd.concat([
        v.groupby(["condition", "machine_type"])["ref_error"].agg(["mean", "size"]).reset_index(),
        v.groupby(["condition", "signal"])["ref_error"].agg(["mean", "size"]).reset_index()
         .rename(columns={"signal": "machine_type"}).assign(machine_type=lambda d: "signal " + d["machine_type"].astype(str)),
    ], ignore_index=True)

    by_payout = pd.DataFrame((r - q) ** 2, columns=P_COLS).assign(condition=v["condition"].to_numpy())
    by_payout = by_payout.groupby("condition").mean().reset_index()

    # Distance of the truth from the machine's long-run and last-32-tick frequencies.
    Y, meta, _ = history_grid(ctx)
    idx = pd.Index(meta["machine_id"]).get_indexer(v["machine_id"].astype(str))
    longrun, recent = np.zeros((len(v), K)), np.zeros((len(v), K))
    for i, m in enumerate(idx):
        obs = Y[m][Y[m] >= 0]
        longrun[i] = np.bincount(obs, minlength=K) / max(len(obs), 1)
        tail = Y[m, max(0, meta.loc[m, "cutoff"] - 32): meta.loc[m, "cutoff"]]
        tail = tail[tail >= 0]
        recent[i] = np.bincount(tail, minlength=K) / max(len(tail), 1)
    ent = -(q * np.log(np.clip(q, 1e-12, None))).sum(1)
    shape = pd.DataFrame({"condition": v["condition"].to_numpy(), "max_q": q.max(1), "entropy": ent,
                          "dist_longrun": ((q - longrun) ** 2).sum(1), "dist_recent32": ((q - recent) ** 2).sum(1),
                          "frac_max_q_over_0.9": q.max(1) > 0.9})
    truth_shape = shape.groupby("condition").mean().reset_index()

    ceil = []
    for name, keys in (("type x signal", ["machine_type", "signal"]),
                       ("condition x type x signal", ["condition", "machine_type", "signal"])):
        mean_q = v.groupby(keys)[Q_COLS].transform("mean").to_numpy()
        err = ((mean_q - q) ** 2).sum(1)
        e = condition_errors(v, err)
        b = condition_errors(v, v["ref_error"].to_numpy())
        ceil += [{"oracle": name, "condition": c, "error": e[c], "ref_error": b[c], "ratio": e[c] / b[c]} for c in e.index]
    ceilings = pd.DataFrame(ceil).pivot_table(index="oracle", columns="condition", values="ratio").reset_index()
    ceilings.columns.name = None
    return {"by_group": by_group, "by_payout": by_payout, "truth_shape": truth_shape, "ceilings": ceilings}
