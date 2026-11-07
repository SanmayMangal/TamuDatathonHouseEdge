"""Causal per-tick features for learned models (every feature at tick t uses ticks < t only).

Two feature sets:
  generic : type, signal, short flag, own win/loss at lags 1-8, recency-weighted win rates,
            the 28 pair-match indicators at t.
  evidence: generic + per-pair rule evidence accumulated causally from the machine's own past
            (rate after a match vs after a mismatch, full-history and recency-weighted, and the
            rate implied for the current state), plus "best pair" summaries.
Rows: every observed history tick with all lags 1-8 known (training targets) and every query row.
"""
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/discovery"))
from prototype_model import load_split  # noqa: E402

PAIRS = list(combinations(range(1, 9), 2))
TYPES = ("steady", "balanced", "jackpot")


def build(split, evidence=True, half_life=24.0, alpha=2.0):
    q, Y, W = load_split(split)
    M, T = W.shape
    cut = q["cutoff"].to_numpy()
    # Lagged win/loss: L[k][m, t] = W[m, t-k]
    lagged = {k: np.concatenate([np.full((M, k), np.nan), W[:, :-k]], 1) for k in range(1, 13)}
    match = {p: np.where(np.isnan(lagged[p[0]]) | np.isnan(lagged[p[1]]), np.nan,
                         (lagged[p[0]] == lagged[p[1]]).astype(float)) for p in PAIRS}
    # Recency-weighted causal win rates
    def causal_rate(hl):
        d = 1.0 if hl is None else 2.0 ** (-1.0 / hl)
        num = np.zeros(M); den = np.zeros(M); out = np.full((M, T), np.nan)
        for t in range(T):
            out[:, t] = (num + 0.5) / (den + 1.0)
            num *= d; den *= d
            ok = ~np.isnan(W[:, t]); num[ok] += W[ok, t]; den[ok] += 1
        return out
    rates = {f"rate_hl{h}": causal_rate(h) for h in (8, 32)} | {"rate_all": causal_rate(None)}
    feats = {}
    if evidence:
        # Per pair: causal stats of outcome after match / mismatch, full and decayed.
        for p in PAIRS:
            mt = match[p]
            for tag, hl in (("all", None), ("hl", half_life)):
                d = 1.0 if hl is None else 2.0 ** (-1.0 / hl)
                a1 = np.zeros(M); n1 = np.zeros(M); a0 = np.zeros(M); n0 = np.zeros(M)
                r1 = np.full((M, T), np.nan); r0 = np.full((M, T), np.nan); ev = np.full((M, T), np.nan)
                for t in range(T):
                    base = rates["rate_all"][:, t]
                    r1[:, t] = (a1 + alpha * base) / (n1 + alpha); r0[:, t] = (a0 + alpha * base) / (n0 + alpha)
                    # evidence: log-likelihood ratio of a 2-rate model vs 1-rate model (approx: n * squared gap)
                    ev[:, t] = (r1[:, t] - r0[:, t]) * np.sqrt(n1 * n0 / np.maximum(n1 + n0, 1))
                    a1 *= d; n1 *= d; a0 *= d; n0 *= d
                    ok = ~np.isnan(mt[:, t]) & ~np.isnan(W[:, t])
                    m1 = ok & (mt[:, t] == 1); m0 = ok & (mt[:, t] == 0)
                    a1[m1] += W[m1, t]; n1[m1] += 1; a0[m0] += W[m0, t]; n0[m0] += 1
                feats[f"ev_{tag}_{p[0]}{p[1]}"] = ev
                feats[f"now_{tag}_{p[0]}{p[1]}"] = np.where(mt == 1, r1, np.where(mt == 0, r0, np.nan))
        for tag in ("all", "hl"):
            E = np.stack([np.abs(feats[f"ev_{tag}_{p[0]}{p[1]}"]) for p in PAIRS])
            N = np.stack([feats[f"now_{tag}_{p[0]}{p[1]}"] for p in PAIRS])
            E = np.where(np.isnan(N), -1, np.nan_to_num(E, nan=-1))
            best = np.argmax(E, 0)
            feats[f"best_now_{tag}"] = np.take_along_axis(N, best[None], 0)[0]
            feats[f"best_ev_{tag}"] = np.take_along_axis(E, best[None], 0)[0]
            srt = np.sort(E, 0)
            feats[f"second_ev_{tag}"] = srt[-2]
    rows = []
    for m in range(M):
        c = int(cut[m])
        ticks = [t for t in range(8, c) if not np.isnan(W[m, t]) and all(not np.isnan(lagged[k][m, t]) for k in range(1, 9))]
        ticks.append(c)
        rows.append(pd.DataFrame({"m": m, "tick": ticks, "is_query": [False] * (len(ticks) - 1) + [True]}))
    R = pd.concat(rows, ignore_index=True)
    mi, ti = R["m"].to_numpy(), R["tick"].to_numpy()
    X = pd.DataFrame({
        "type_" + t: (q["machine_type"].to_numpy()[mi] == t).astype(float) for t in TYPES})
    X["signal"] = q["signal"].to_numpy()[mi]
    X["short"] = q["short"].to_numpy()[mi].astype(float)
    for k in range(1, 9):
        X[f"lag{k}"] = lagged[k][mi, ti]
    for name, arr in rates.items():
        X[name] = arr[mi, ti]
    for p in PAIRS:
        X[f"match_{p[0]}{p[1]}"] = match[p][mi, ti]
    for name, arr in feats.items():
        X[name] = arr[mi, ti]
    y = np.where(R["is_query"], -1, Y[mi, np.minimum(ti, Y.shape[1] - 1)])
    meta = q.iloc[mi].reset_index(drop=True)[["query_id", "episode_id", "machine_id", "machine_type"]
                                              + [c for c in ("condition",) if c in q]]
    meta["is_query"] = R["is_query"].to_numpy(); meta["tick"] = ti
    return X, y, meta, q
