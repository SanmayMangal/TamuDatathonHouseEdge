"""Hybrids: blend the Bayesian model with the evidence-feature GBM, and a learned calibration layer
on the Bayesian outputs. Weights/calibration are chosen with 4-fold episode cross-validation on the
validation queries (true probabilities), so the reported score is out-of-fold.

    python analysis/model_choice/hybrid.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "discovery"))
from features import build  # noqa: E402
import prototype_model as pm  # noqa: E402

lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
sg = lambda z: 1 / (1 + np.exp(-z))


def err(pred, truth):
    return ((pred - truth) ** 2).sum(1)


def main():
    hp = pm.fit_hyper()
    q, PB = pm.predict("validation", hp)
    Xtr, ytr, mtr, _ = build("train"); Xva, _, mva, qva = build("validation")
    train = ~mtr["is_query"].to_numpy(); vq = mva["is_query"].to_numpy()
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Xtr[train], ytr[train])
    PG = np.zeros((vq.sum(), 5)); PG[:, gbm.classes_.astype(int)] = gbm.predict_proba(Xva[vq])
    PG = PG[pd.Index(mva.loc[vq, "machine_id"]).get_indexer(q["machine_id"])]
    truth = q[pm.Q].to_numpy()
    eps = q["episode_id"].unique(); rng = np.random.default_rng(0); rng.shuffle(eps)
    fold = pd.Series(np.arange(len(eps)) % 4, index=eps).reindex(q["episode_id"]).to_numpy()

    # 1. Linear blend of probabilities, weight chosen in-fold.
    grid = np.linspace(0, 1, 21)
    oof = np.zeros_like(PB); chosen = []
    for f in range(4):
        tr, te = fold != f, fold == f
        w = grid[np.argmin([err(g * PG[tr] + (1 - g) * PB[tr], truth[tr]).mean() for g in grid])]
        chosen.append(w); oof[te] = w * PG[te] + (1 - w) * PB[te]
    pm.exact_score(q, PB, "Bayesian alone")
    pm.exact_score(q, PG, "GBM (evidence features) alone")
    pm.exact_score(q, oof, f"blend, out-of-fold (weights on GBM {np.round(chosen, 2)})")

    # 2. Calibration of the Bayesian win log-odds per type: logit' = a_type * logit + b_type, in-fold.
    pw = 1 - PB[:, 0]; z = lg(pw); s = PB[:, 1] / PB[:, 1:].sum(1); hi = PB[:, 2:] / PB[:, 2:].sum(1, keepdims=True)
    oof2 = np.zeros_like(PB); params = []
    A = np.linspace(0.8, 1.3, 11); Bs = np.linspace(-0.4, 0.4, 17)
    for f in range(4):
        tr, te = fold != f, fold == f
        for typ in pm.TYPES:
            mt = (q["machine_type"] == typ).to_numpy()
            best = min(((a, b) for a in A for b in Bs), key=lambda ab: err(
                np.column_stack([1 - sg(ab[0] * z + ab[1]), sg(ab[0] * z + ab[1]) * s, (sg(ab[0] * z + ab[1]) * (1 - s))[:, None] * hi])[tr & mt],
                truth[tr & mt]).mean())
            params.append((f, typ) + best)
            p2 = sg(best[0] * z + best[1])
            oof2[te & mt] = np.column_stack([1 - p2, p2 * s, (p2 * (1 - s))[:, None] * hi])[te & mt]
    pm.exact_score(q, oof2, "Bayesian + per-type log-odds calibration (out-of-fold)")
    print("calibration params (fold, type, slope, shift):", [(f, t, round(a, 2), round(b, 2)) for f, t, a, b in params])


if __name__ == "__main__":
    main()
