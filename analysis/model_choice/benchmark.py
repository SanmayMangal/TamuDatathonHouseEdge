"""Learned models vs the Bayesian prototype, scored exactly on validation targets.

    python analysis/model_choice/benchmark.py
"""
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "discovery"))
from features import build  # noqa: E402
import prototype_model as pm  # noqa: E402

GENERIC = lambda cols: [c for c in cols if not c.startswith(("ev_", "now_", "best_", "second_"))]


def to_pred(model, X):
    P = np.zeros((len(X), 5))
    P[:, model.classes_.astype(int)] = model.predict_proba(X)
    return P


def main():
    t0 = time.time()
    Xtr, ytr, mtr, _ = build("train")
    Xva, _, mva, qva = build("validation")
    print(f"features built in {time.time() - t0:.0f}s: train rows {len(Xtr):,} x {Xtr.shape[1]} features", flush=True)
    train = ~mtr["is_query"].to_numpy()
    vq = mva["is_query"].to_numpy()
    order = pd.Index(mva.loc[vq, "machine_id"]).get_indexer(qva["machine_id"])  # align to qva order
    results = []

    def run(name, model, cols, scale=False):
        t = time.time()
        A, B = Xtr.loc[train, cols].to_numpy(), Xva.loc[vq, cols].to_numpy()
        if scale:
            A, B = np.nan_to_num(A, nan=0.0), np.nan_to_num(B, nan=0.0)
            sc = StandardScaler().fit(A); A, B = sc.transform(A), sc.transform(B)
        model.fit(A, ytr[train])
        P = to_pred(model, B)[order]
        R = pm.exact_score(qva, P, name)
        results.append({"model": name, "R": R, "points": 60 / (1 + R), "seconds": time.time() - t})

    cols_all = list(Xtr.columns)
    run("GBM, generic lag features", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0), GENERIC(cols_all))
    run("GBM, + rule-evidence features", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0), cols_all)
    run("MLP (64x64), + rule-evidence features",
        MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=60, early_stopping=True, random_state=0), cols_all, scale=True)
    hp = pm.fit_hyper()
    q, P = pm.predict("validation", hp)
    R = pm.exact_score(q, P, "Bayesian rule model (prototype)")
    results.append({"model": "Bayesian rule model (prototype)", "R": R, "points": 60 / (1 + R), "seconds": np.nan})
    out = HERE.parent / "results/model_choice"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "benchmark.csv", index=False)


if __name__ == "__main__":
    main()
