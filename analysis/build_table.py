"""Build the shared example table for House Edge experiments.

One row per observed (split, episode, machine, tick) in history.csv, plus one
"query" row per machine at its cutoff. Every feature on a row at tick t is
built only from ticks strictly below t, so history rows are valid
self-supervised training examples and query rows are the real predictions.

Usage (from the repo root):
    python analysis/build_table.py
    python analysis/build_table.py --data path/to/public --out analysis/artifacts/examples.pkl
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "main-participants-60-40"

PAYOUTS = (0, 1, 2, 5, 20)
P_COLS = [f"p_{p}" for p in PAYOUTS]
OWN_LAGS = 12          # 9-12 kept as a no-signal control (docs: effects up to lag 8)
NB_LAGS = 8
N_FOLDS = 5
REF_STRENGTH = 20.0    # metadata.json: "Pooled, strength=20"


def read_split(public, split):
    d = public / split
    q = pd.read_csv(d / "queries.csv")
    h = pd.read_csv(d / "history.csv")
    if (d / "labels.csv").exists():
        q = q.merge(pd.read_csv(d / "labels.csv"), on="query_id", how="left")
    else:
        q["outcome"] = np.nan
    if (d / "conditions.csv").exists():
        q = q.merge(pd.read_csv(d / "conditions.csv"), on="query_id", how="left")
    else:
        q["condition"] = pd.NA
    if (d / "targets.csv").exists():
        t = pd.read_csv(d / "targets.csv")[["query_id"] + P_COLS]
        q = q.merge(t.rename(columns={c: "q" + c[1:] for c in P_COLS}), on="query_id", how="left")
    return q, h


def reference_priors(public):
    """Fit the kit's frozen pooled baseline on this round's train split."""
    sys.path.insert(0, str(KIT))
    from slot_forecasting.baseline import FrequencyForecaster
    from slot_forecasting.io import load_inputs, read_labels

    queries, history = load_inputs(public / "train")
    labels = read_labels(public / "train/labels.csv", queries)
    model = FrequencyForecaster("pooled", prior_strength=REF_STRENGTH).fit(queries, history, labels)
    return model.global_prior, model.context_priors


def episode_folds(q):
    """Episode-level folds, stratified by condition, deterministic by hash."""
    folds = {}
    ep = q.groupby("episode_id")["condition"].first()
    for cond, eps in ep.groupby(ep.fillna("unknown")):
        ordered = sorted(eps.index, key=lambda e: hashlib.sha256(e.encode()).hexdigest())
        for i, e in enumerate(ordered):
            folds[e] = i % N_FOLDS
    return folds


def build_split(split, q, h, priors):
    global_prior, context_priors = priors
    folds = episode_folds(q) if split != "test" else {}
    out = []
    hist_by_ep = {e: g for e, g in h.groupby("episode_id")}
    for ep_id, eq in q.groupby("episode_id", sort=True):
        eq = eq.sort_values("machine_id").reset_index(drop=True)
        cutoff = int(eq["cutoff"].iloc[0])
        m_ids = list(eq["machine_id"])
        n = len(m_ids)
        slot = {m: i for i, m in enumerate(m_ids)}
        eh = hist_by_ep[ep_id]
        # grid[i, t] = payout of machine i at tick t, NaN if unobserved; extra column for the cutoff.
        grid = np.full((n, cutoff + 1), np.nan)
        grid[eh["machine_id"].map(slot).to_numpy(), eh["tick"].to_numpy()] = eh["payout"].to_numpy()
        assert eh["tick"].max() < cutoff, "history at/after cutoff"
        short = bool(np.isnan(grid[:, :cutoff]).any())

        for i, row in eq.iterrows():
            obs_ticks = np.flatnonzero(~np.isnan(grid[i, :cutoff]))
            ticks = np.append(obs_ticks, cutoff)
            k = len(ticks)
            rec = {
                "split": split, "episode_id": ep_id, "machine_id": row["machine_id"],
                "machine_slot": i, "tick": ticks, "cutoff": cutoff,
                "ticks_to_cutoff": cutoff - ticks,
                "is_query": np.r_[np.zeros(k - 1, bool), True],
                "outcome": np.r_[grid[i, obs_ticks], row["outcome"]],
                "machine_type": row["machine_type"], "signal": int(row["signal"]),
                "condition": row["condition"], "short_history": short,
                "fold": folds.get(ep_id, -1),
                "n_obs_before": np.arange(k),
                "gap_since_prev": np.r_[np.nan, np.diff(ticks)],
            }
            for lag in range(1, OWN_LAGS + 1):
                src = ticks - lag
                rec[f"own_lag{lag}"] = np.where(src >= 0, grid[i, np.clip(src, 0, None)], np.nan)
            others = [j for j in range(n) if j != i]
            for s, j in enumerate(others, 1):
                rec[f"nb{s}_type"] = eq.loc[j, "machine_type"]
                rec[f"nb{s}_signal"] = int(eq.loc[j, "signal"])
                for lag in range(1, NB_LAGS + 1):
                    src = ticks - lag
                    rec[f"nb{s}_lag{lag}"] = np.where(src >= 0, grid[j, np.clip(src, 0, None)], np.nan)

            # Frozen reference: pooled counts of own payouts strictly before each tick.
            prior = np.array(context_priors.get(f"{row['machine_type']}|{int(row['signal'])}", global_prior))
            onehot = np.zeros((k, len(PAYOUTS)))
            obs_out = grid[i, obs_ticks].astype(int)
            onehot[np.arange(k - 1), [PAYOUTS.index(v) for v in obs_out]] = 1
            before = np.vstack([np.zeros(len(PAYOUTS)), np.cumsum(onehot[:-1], axis=0)])
            ref = (REF_STRENGTH * prior + before) / (REF_STRENGTH + before.sum(1, keepdims=True))
            for c, col in enumerate(P_COLS):
                rec["ref_" + col] = ref[:, c]
            for col in ("q_0", "q_1", "q_2", "q_5", "q_20"):
                rec[col] = np.r_[np.full(k - 1, np.nan), row[col]] if col in row else np.nan
            out.append(pd.DataFrame(rec))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=KIT / "public")
    ap.add_argument("--out", type=Path, default=ROOT / "analysis/artifacts/examples.pkl")
    args = ap.parse_args()
    public = args.data.resolve()
    priors = reference_priors(public)
    frames = [build_split(s, *read_split(public, s), priors) for s in ("train", "validation", "test")]
    df = pd.concat(frames, ignore_index=True)
    for c in ("split", "machine_type", "condition") + tuple(f"nb{s}_type" for s in range(1, 8)):
        df[c] = df[c].astype("category")
    lag_cols = [c for c in df.columns if "_lag" in c] + ["outcome"]
    df[lag_cols] = df[lag_cols].astype("float32")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.out)
    summary = df.groupby(["split", "is_query"], observed=True).size().unstack()
    print(summary.to_string())
    print(f"{len(df):,} rows x {df.shape[1]} columns -> {args.out}")


if __name__ == "__main__":
    main()
