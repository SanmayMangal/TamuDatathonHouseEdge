"""Scratch helpers: per-machine grids for every split, including test."""
import numpy as np, pandas as pd
from pathlib import Path
PUB = Path(__file__).resolve().parents[2] / "main-participants-60-40/public"
PAY = np.array([0, 1, 2, 5, 20])
Q = ["p_0", "p_1", "p_2", "p_5", "p_20"]

def load_split(split):
    q = pd.read_csv(PUB / split / "queries.csv")
    h = pd.read_csv(PUB / split / "history.csv")
    if (PUB / split / "labels.csv").exists():
        q = q.merge(pd.read_csv(PUB / split / "labels.csv"), on="query_id")
    if (PUB / split / "conditions.csv").exists():
        q = q.merge(pd.read_csv(PUB / split / "conditions.csv"), on="query_id")
    if (PUB / split / "targets.csv").exists():
        t = pd.read_csv(PUB / split / "targets.csv").drop(columns=["episode_id", "outcome"])
        q = q.merge(t, on="query_id")
    if (PUB / split / "reference.csv").exists():
        r = pd.read_csv(PUB / split / "reference.csv").rename(columns={c: "ref_" + c for c in Q})
        q = q.merge(r, on="query_id")
    q = q.sort_values(["episode_id", "machine_id"]).reset_index(drop=True)
    T = 257
    Y = np.full((len(q), T), -1)
    idx = pd.Index(q["machine_id"]).get_indexer(h["machine_id"])
    Y[idx, h["tick"].to_numpy()] = np.searchsorted(PAY, h["payout"].to_numpy())
    q["short"] = [(Y[m, :c] < 0).any() for m, c in enumerate(q["cutoff"])]
    return q, Y

def win_grid(Y):
    return np.where(Y >= 0, (Y > 0).astype(float), np.nan)
