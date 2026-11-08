"""Generate a synthetic "changed" kit to rehearse round two (testing only; never used for a submission).

Mimics the kit's file formats and the main-round simulator as we understand it, but changes the structure:
rules are new lag pairs, three-lag parities and single lags; flips happen 10-60 ticks before the cutoff.

    python round2/synth_kit.py --kit main-participants-60-40 --out /tmp/synth_kit [--seed 7]
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

PAY = np.array([0, 1, 2, 5, 20])
TYPES = ("steady", "balanced", "jackpot")
BASE = {"steady": 0.7, "balanced": -0.6, "jackpot": -2.4}
SHARE1 = {"steady": 0.62, "balanced": 0.35, "jackpot": 0.21}
HI = {"steady": (0.79, 0.205, 0.005), "balanced": (0.69, 0.265, 0.045), "jackpot": (0.376, 0.461, 0.163)}
RULES = [((3, 6), 0.2), ((2, 8), 0.2), ((1, 4, 7), 0.2), ((2, 3, 6), 0.1), ((5,), 0.3)]
CONDS = ("stable", "short_history", "drift", "shared_change")
COUNTS = {"train": 24, "validation": 24, "test": 48}
lg = lambda p: np.log(p / (1 - p))
sg = lambda z: 1 / (1 + np.exp(-z))


def episode(rng, cond):
    c = int(rng.integers(192, 257))
    types = rng.choice(TYPES, 8, p=[0.45, 0.3, 0.25]); signals = rng.integers(-1, 2, 8)
    rules = [RULES[i][0] for i in rng.choice(len(RULES), 8, p=[r[1] for r in RULES])]
    dirs = rng.choice([1, -1], 8)
    flip = np.full(8, -1)
    if cond == "drift":
        flip = c - rng.integers(10, 61, 8)
    elif cond == "shared_change":
        flip[:] = c - rng.integers(10, 61)
    m_off = rng.normal(0, 0.3, 8)
    s_lat = lg(np.array([SHARE1[t] for t in types])) + rng.normal(0, 0.6, 8)
    e = np.zeros(c + 1); e[0] = rng.normal(0, 0.4)
    for t in range(1, c + 1):
        e[t] = 0.95 * e[t - 1] + rng.normal(0, 0.4 * np.sqrt(1 - 0.95 ** 2))
    Y = np.zeros((8, c), int); Q = np.zeros((8, 5))
    for t in range(c + 1):
        for m in range(8):
            if t >= 8:
                state = (sum(Y[m, t - k] > 0 for k in rules[m]) % 2 == 0)
                d = dirs[m] * (-1 if (flip[m] >= 0 and t < flip[m]) else 1)
                amp = 2.2 * (1 if state else -1) * d
            else:
                amp = 0.0
            pw = sg(BASE[types[m]] + 0.35 * signals[m] + m_off[m] + e[t] + amp)
            s1 = sg(s_lat[m] + rng.normal(0, 0.5))
            p = np.r_[1 - pw, pw * s1, pw * (1 - s1) * np.array(HI[types[m]])]
            if t == c:
                Q[m] = p
            else:
                Y[m, t] = rng.choice(5, p=p)
    return c, types, signals, Y, Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    if a.out.exists():
        shutil.rmtree(a.out)
    for f in ("check_submission.py", "build_entry.py"):
        shutil.copy(a.kit / f, a.out / f) if (a.out).exists() else None
    shutil.copytree(a.kit / "slot_forecasting", a.out / "slot_forecasting", ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("check_submission.py", "build_entry.py"):
        shutil.copy(a.kit / f, a.out / f)
    pub = a.out / "public"
    for split, n in COUNTS.items():
        (pub / split).mkdir(parents=True)
        Qr, H, L, C, T = [], [], [], [], []
        for i, cond in enumerate([c for c in CONDS for _ in range(n)]):
            ep = f"adaptation-{split}-e{i:05d}"
            c, types, signals, Y, Q = episode(rng, cond)
            short = cond == "short_history"
            for m in range(8):
                mid = f"{ep}-m{m:03d}"; qid = f"{mid}-next"
                ticks = np.arange(c)
                if short:
                    ticks = np.r_[np.sort(rng.choice(c - 40, 24, replace=False)), np.arange(c - 40, c)]
                Qr.append({"query_id": qid, "episode_id": ep, "machine_id": mid, "machine_type": types[m],
                           "signal": int(signals[m]), "cutoff": c})
                H += [{"episode_id": ep, "machine_id": mid, "tick": int(t), "payout": int(PAY[Y[m, t]])} for t in ticks]
                y = int(PAY[rng.choice(5, p=Q[m])])
                L.append({"query_id": qid, "outcome": y}); C.append({"query_id": qid, "condition": cond})
                T.append({"query_id": qid, "episode_id": ep, **{f"p_{p}": Q[m, k] for k, p in enumerate(PAY)}, "outcome": y})
        pd.DataFrame(Qr).to_csv(pub / split / "queries.csv", index=False)
        pd.DataFrame(H).to_csv(pub / split / "history.csv", index=False)
        pd.DataFrame({"query_id": [q["query_id"] for q in Qr], **{f"p_{p}": 0.2 for p in PAY}}).to_csv(pub / split / "sample_submission.csv", index=False)
        if split != "test":
            pd.DataFrame(L).to_csv(pub / split / "labels.csv", index=False)
            pd.DataFrame(C).to_csv(pub / split / "conditions.csv", index=False)
        if split == "validation":
            pd.DataFrame(T).to_csv(pub / split / "targets.csv", index=False)
    meta = json.loads((a.kit / "public/metadata.json").read_text())
    meta.update({"round": "adaptation", "max_points": 40, "season_id": "0123456789abcdef01234567",
                 "counts_per_condition": COUNTS, "warning": "SYNTHETIC rehearsal kit"})
    (pub / "metadata.json").write_text(json.dumps(meta, indent=2))
    # Reference: the kit's pooled baseline, fitted on synthetic train.
    sys.path.insert(0, str(a.out))
    from slot_forecasting.baseline import FrequencyForecaster
    from slot_forecasting.io import load_inputs, read_labels, write_submission
    q, h = load_inputs(pub / "train"); lab = read_labels(pub / "train/labels.csv", q)
    model = FrequencyForecaster("pooled", prior_strength=20).fit(q, h, lab)
    vq, vh = load_inputs(pub / "validation")
    write_submission(pub / "validation/reference.csv", model.predict(vq, vh))
    print(f"wrote synthetic kit to {a.out}")


if __name__ == "__main__":
    main()
