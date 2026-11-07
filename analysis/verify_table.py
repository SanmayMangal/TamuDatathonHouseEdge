"""Sanity checks for the example table and evaluator. Run after build_table.py.

    python analysis/verify_table.py [--comparison path/to/run/comparison]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import P_COLS, REF_COLS, noise_by_condition, report, score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "main-participants-60-40"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=ROOT / "analysis/artifacts/examples.pkl")
    ap.add_argument("--data", type=Path, default=KIT / "public")
    ap.add_argument("--comparison", type=Path, help="run_starter.py output/comparison folder")
    args = ap.parse_args()
    df = pd.read_pickle(args.table)
    ok = True

    # 1. Row counts match the raw files.
    for split in ("train", "validation", "test"):
        nh = len(pd.read_csv(args.data / split / "history.csv"))
        nq = len(pd.read_csv(args.data / split / "queries.csv"))
        s = df[df["split"] == split]
        good = (~s["is_query"]).sum() == nh and s["is_query"].sum() == nq
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] {split}: {nh} history rows, {nq} query rows")

    # 2. Reference columns reproduce the kit's frozen reference forecasts.
    vq = df[(df["split"] == "validation") & df["is_query"]].copy()
    ref = pd.read_csv(args.data / "validation/reference.csv").set_index("query_id")
    vq["query_id"] = vq["machine_id"].astype(str) + "-next"
    diff = np.abs(vq[REF_COLS].to_numpy() - ref.loc[vq["query_id"], P_COLS].to_numpy()).max()
    ok &= diff < 1e-12
    print(f"[{'ok' if diff < 1e-12 else 'FAIL'}] reference columns vs reference.csv: max diff {diff:.2e}")

    # 3. Lag features equal raw history lookups, and never read the row's own tick or later.
    rng = np.random.default_rng(0)
    raw = {}
    for split in ("train", "validation", "test"):
        h = pd.read_csv(args.data / split / "history.csv")
        raw.update({(m, t): p for m, t, p in zip(h["machine_id"], h["tick"], h["payout"])})
    sample = df.iloc[rng.choice(len(df), 3000, replace=False)]
    machines = df.groupby("episode_id", observed=True)["machine_id"].unique()
    bad = 0
    for _, r in sample.iterrows():
        t = r["tick"]
        for lag in (1, 3, 8, 12):
            exp = raw.get((r["machine_id"], t - lag), np.nan)
            got = r[f"own_lag{lag}"]
            bad += not ((np.isnan(exp) and np.isnan(got)) or exp == got)
        others = sorted(m for m in machines[r["episode_id"]] if m != r["machine_id"])
        for s, m in enumerate(others, 1):
            for lag in (1, 8):
                exp = raw.get((m, t - lag), np.nan)
                got = r[f"nb{s}_lag{lag}"]
                bad += not ((np.isnan(exp) and np.isnan(got)) or exp == got)
        if not r["is_query"]:
            bad += raw[(r["machine_id"], t)] != r["outcome"]
    ok &= bad == 0
    print(f"[{'ok' if bad == 0 else 'FAIL'}] 3000 random rows: lag features and outcomes match raw history ({bad} mismatches)")

    # 4. Evaluator reproduces the official checker on validation.
    res = score(vq, vq[REF_COLS].to_numpy(), against="target")
    good = abs(res["points"] - 30.0) < 1e-9
    ok &= good
    print(f"[{'ok' if good else 'FAIL'}] {report(res, 'reference (target)')}")
    if args.comparison:
        for csv in sorted(args.comparison.glob("*.csv")):
            p = pd.read_csv(csv).set_index("query_id").loc[vq["query_id"], P_COLS].to_numpy()
            mine = score(vq, p, against="target")["points"]
            out = subprocess.run([sys.executable, "check_submission.py", str(csv), "--split", "validation"],
                                 cwd=KIT, capture_output=True, text=True).stdout
            theirs = float(next(l for l in out.splitlines() if l.startswith("Validation points")).split()[2])
            good = abs(mine - theirs) < 1e-6
            ok &= good
            print(f"[{'ok' if good else 'FAIL'}] {csv.stem:14s} evaluator {mine:.6f} vs checker {theirs:.6f}")

    # 5. Outcome-based proxy on history rows (informational).
    noise = noise_by_condition(df)
    lab = df[(df["split"] == "validation") & ~df["is_query"]]
    flat = np.full((len(lab), 5), 0.2)
    print("     noise (1 - sum q^2) by condition:", {k: round(v, 4) for k, v in noise.items()})
    print("     " + report(score(lab, flat, "outcome"), "uniform, raw proxy"))
    print("     " + report(score(lab, flat, "outcome", noise=noise), "uniform, noise-corrected"))
    print("     " + report(score(vq, np.full((len(vq), 5), 0.2), "target"), "uniform, exact (queries)"))
    print("     reference error by condition (exact):", {k: round(v, 4) for k, v in res["reference_error"].items()})

    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
