"""Round-two fitter: learns the rule structure from a kit's labeled histories (numpy, offline).

Generalizes the main-round model so nothing about the main round's structure is assumed:
  * Rule family: a rule is a set of lags L (single lag, pair, or triple, all lags <= MAX_LAG).
    State at tick t = parity of the encoded outcomes at t-k for k in L (even = "state on").
    For a pair, even parity = the two outcomes match, i.e. the main-round rule.
  * Encoding of lagged outcomes: "win" (payout > 0) or "profit" (payout >= 2), chosen by evidence.
  * Flips: zero or one direction flip anywhere in the history, with the flip-timing distribution and
    flip probability (long vs short histories) learned by EM.
  * Rule-family prior learned by EM, so families that don't occur get ~zero weight.
Then state win rates per type x signal (with between-machine Beta strengths), the 1x share, and the
high-prize mix (with a learned per-machine shrinkage strength in case the mix varies by machine).

    python round2/fit.py --data path/to/public --splits train --out hp.json
"""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betaln, logsumexp

PAY = np.array([0, 1, 2, 5, 20])
TYPES = ("steady", "balanced", "jackpot")
MAX_LAG = 8
RULES = [(k,) for k in range(1, MAX_LAG + 1)] + list(combinations(range(1, MAX_LAG + 1), 2)) \
    + list(combinations(range(1, MAX_LAG + 1), 3))
DMAX = 400
SETTINGS = {"ep_half_life": 6.0, "ep_var": 0.2, "machine_var": 0.05, "share_strength": 60.0}
ENCODINGS = {"win": 1, "profit": 2}   # outcome index threshold: payout index >= threshold


def load(public, split):
    d = Path(public) / split
    q = pd.read_csv(d / "queries.csv")
    h = pd.read_csv(d / "history.csv")
    for name in ("labels.csv", "conditions.csv"):
        if (d / name).exists():
            q = q.merge(pd.read_csv(d / name), on="query_id")
    if (d / "targets.csv").exists():
        q = q.merge(pd.read_csv(d / "targets.csv").drop(columns=["episode_id", "outcome"], errors="ignore"), on="query_id")
    q = q.sort_values(["episode_id", "machine_id"]).reset_index(drop=True)
    T = int(q["cutoff"].max()) + 1
    Y = np.full((len(q), T), -1)
    Y[pd.Index(q["machine_id"]).get_indexer(h["machine_id"]), h["tick"].to_numpy()] = np.searchsorted(PAY, h["payout"].to_numpy())
    q["short"] = [(Y[m, :c] < 0).any() for m, c in enumerate(q["cutoff"])]
    return q, Y


def rule_table(Yrow, c, rule, thr):
    """Valid ticks, state (even parity), win outcome, and per-split log marginal (Beta(1,1)) for k=0..n."""
    t = np.arange(MAX_LAG, c)
    y = Yrow[t]
    ok = y >= 0
    par = np.zeros(len(t), int)
    for k in rule:
        v = Yrow[t - k]
        ok &= v >= 0
        par += (v >= thr)
    t, s, w = t[ok], (par[ok] % 2 == 0), (y[ok] >= 1).astype(float)
    n = len(t)
    A1 = np.r_[0, np.cumsum(s * w)]; N1 = np.r_[0, np.cumsum(s)]; Yc = np.r_[0, np.cumsum(w)]
    a1p, n1p = A1, N1
    a0p, n0p = Yc - A1, np.arange(n + 1) - N1
    A1T, N1T, YT = A1[-1], N1[-1], Yc[-1]
    A0T, N0T = YT - A1T, n - N1T
    # direction-free split: cell P = {after: s, before: not s}, cell Q = complement
    kP, nP = a0p + (A1T - a1p), n0p + (N1T - n1p)
    kQ, nQ = a1p + (A0T - a0p), n1p + (N0T - n0p)
    ll = betaln(1 + kP, 1 + nP - kP) + betaln(1 + kQ, 1 + nQ - kQ)
    return t, ll, (kP, nP, kQ, nQ)


def flip_mass(t, c, h_cum, p_flip):
    """Prior mass of each split k=0..n given the flip-distance histogram (cumulative) and p_flip."""
    n = len(t)
    if n == 0:
        return np.array([1.0])
    def H(a, b):  # mass for distances a..b inclusive (1-based), clipped
        a = np.clip(a, 1, DMAX); b = np.clip(b, 0, DMAX)
        return np.where(b >= a, h_cum[b] - h_cum[a - 1], 0.0)
    hi = np.r_[DMAX, c - t[:-1] - 1]            # k: tau in (t[k-1], t[k]] -> distance c-t[k] .. c-t[k-1]-1
    lo = c - t
    mass = np.empty(n + 1)
    mass[:n] = p_flip * H(lo, hi)
    mass[n] = p_flip * H(np.array([1]), np.array([c - t[-1] - 1]))[0]
    mass[0] += 1 - p_flip
    return np.maximum(mass, 1e-300)


def fit_structure(q, Y, thr, iters=14, log=print):
    M = len(q)
    tabs = [[rule_table(Y[m], int(q.loc[m, "cutoff"]), r, thr) for r in RULES] for m in range(M)]
    logpi = np.full(len(RULES), -np.log(len(RULES)))
    h = np.ones(DMAX + 1); h[0] = 0; h /= h.sum()
    pf = {True: 0.3, False: 0.3}  # keyed by short
    for it in range(iters + 1):
        h_cum = np.cumsum(h)
        post_rule = np.zeros(len(RULES)); flip_hist = np.zeros(DMAX + 1); flip_p = {True: [], False: []}
        total = 0.0; map_choice = []
        for m in range(M):
            c = int(q.loc[m, "cutoff"]); sh = bool(q.loc[m, "short"])
            joint = []
            for r, (t, ll, _) in enumerate(tabs[m]):
                mass = flip_mass(t, c, h_cum, pf[sh])
                joint.append(logpi[r] + np.log(mass) + ll)
            flat = np.concatenate(joint); Z = logsumexp(flat); total += Z
            pr = np.exp(flat - Z)
            off = 0; pflip_m = 0.0
            best = (-np.inf, None)
            for r, (t, ll, _) in enumerate(tabs[m]):
                p = pr[off: off + len(ll)]; off += len(ll)
                post_rule[r] += p.sum()
                if len(t):
                    mass0 = flip_mass(t, c, h_cum, pf[sh])[0]
                    noflip = p[0] * (1 - pf[sh]) / mass0
                    pflip_m += p.sum() - noflip
                    d = np.clip(c - np.r_[t, t[-1] + 1], 1, DMAX)   # split k -> first post tick (k=n: just after last)
                    np.add.at(flip_hist, d[1:], p[1:])
                    j = int(np.argmax(joint[r]))
                    if joint[r][j] > best[0]:
                        best = (joint[r][j], (r, j))
            flip_p[sh].append(pflip_m)
            map_choice.append(best[1])
        if it == iters:
            break
        logpi = np.log((post_rule + 0.5) / (post_rule + 0.5).sum())
        sm = np.convolve(flip_hist, np.ones(5) / 5, mode="same") + 0.02
        sm[0] = 0; h = sm / sm.sum()
        for sh in (True, False):
            if flip_p[sh]:
                pf[sh] = float(np.clip(np.mean(flip_p[sh]), 1e-3, 1 - 1e-3))
        log(f"   EM iter {it}: log evidence {total:.1f}, p_flip long {pf[False]:.3f} short {pf[True]:.3f}")
    return {"logpi": logpi, "h": h, "p_flip": pf, "evidence": total, "map": map_choice, "tabs": tabs}


def fit(public, splits=("train",), log=print):
    parts = [load(public, s) for s in splits]
    q = pd.concat([p[0] for p in parts], ignore_index=True)
    T = max(p[1].shape[1] for p in parts)
    Y = np.vstack([np.pad(p[1], ((0, 0), (0, T - p[1].shape[1])), constant_values=-1) for p in parts])
    results = {}
    for enc, thr in ENCODINGS.items():
        log(f"structure EM, encoding={enc}")
        results[enc] = fit_structure(q, Y, thr, log=log)
    enc = max(results, key=lambda e: results[e]["evidence"])
    S = results[enc]
    log(f"chosen encoding: {enc} (log evidence " + ", ".join(f"{e} {r['evidence']:.1f}" for e, r in results.items()) + ")")
    rows = []
    for m, (r, j) in enumerate(S["map"]):
        t, ll, (kP, nP, kQ, nQ) = S["tabs"][m][r]
        c = int(q.loc[m, "cutoff"])
        a = (kP[j], nP[j], kQ[j], nQ[j])
        pP, pQ = (a[0] + 1) / (a[1] + 2), (a[2] + 1) / (a[3] + 2)
        kf, nf, ku, nu = a if pP >= pQ else (a[2], a[3], a[0], a[1])
        rows.append({"machine_id": q.loc[m, "machine_id"], "episode_id": q.loc[m, "episode_id"],
                     "condition": q.loc[m, "condition"] if "condition" in q else None, "cutoff": c,
                     "type": q.loc[m, "machine_type"], "signal": int(q.loc[m, "signal"]), "short": bool(q.loc[m, "short"]),
                     "rule": RULES[r], "rule_family": len(RULES[r]), "split": j, "n_valid": len(t),
                     "flip_d": (c - t[j]) if 0 < j < len(t) else (np.nan if j == 0 else 0),
                     "kf": kf, "nf": nf, "ku": ku, "nu": nu})
    r = pd.DataFrame(rows)
    g = r.groupby(["type", "signal"])[["kf", "nf", "ku", "nu"]].sum()
    gt = r.groupby("type")[["kf", "nf", "ku", "nu"]].sum()
    rates = [[t, int(s), float(v.kf / v.nf), float(v.ku / v.nu)] for (t, s), v in g.iterrows() if v.nf > 0 and v.nu > 0]
    type_rates = {t: [float(v.kf / v.nf), float(v.ku / v.nu)] for t, v in gt.iterrows()}
    strength = {}
    for typ in type_rates:
        d = r[(r["type"] == typ) & ~r["short"]]
        d = d if len(d) > 5 else r[r["type"] == typ]
        s = []
        for k, n in (("kf", "nf"), ("ku", "nu")):
            dd = d[d[n] > 0]
            p = dd[k] / dd[n]; pbar = dd[k].sum() / dd[n].sum()
            var_true = max(p.var() - (pbar * (1 - pbar) / dd[n]).mean(), 1e-6)
            s.append(float(np.clip(pbar * (1 - pbar) / var_true - 1, 10, 300)))
        strength[typ] = s
    # Size mix: 1x share and high-prize split per type; learned per-machine shrinkage for the high split.
    share1, hi_mix, hi_strength = {}, {}, {}
    for typ in TYPES:
        idx = np.flatnonzero(q["machine_type"] == typ)
        if not len(idx):
            continue
        v = np.concatenate([Y[m, :q.loc[m, "cutoff"]] for m in idx])
        share1[typ] = float((v[v > 0] == 1).mean())
        mix = np.array([(v == k).sum() for k in (2, 3, 4)], float) + 0.5
        hi_mix[typ] = (mix / mix.sum()).tolist()
        # dispersion of the machine-level share of 2x within high prizes
        p2, n = [], []
        for m in idx:
            u = Y[m, :q.loc[m, "cutoff"]]; u = u[u >= 2]
            if len(u) >= 5:
                p2.append((u == 2).mean()); n.append(len(u))
        p2, n = np.array(p2), np.array(n)
        pb = hi_mix[typ][0]
        var_true = p2.var() - (pb * (1 - pb) / n).mean() if len(p2) > 5 else 0
        hi_strength[typ] = float(np.clip(pb * (1 - pb) / var_true - 1, 10, 1e6)) if var_true > 1e-6 else 1e6
    fam = {"single": 0.0, "pair": 0.0, "triple": 0.0}
    pi = np.exp(S["logpi"])
    for rr, p in zip(RULES, pi):
        fam[{1: "single", 2: "pair", 3: "triple"}[len(rr)]] += float(p)
    return {
        "model": "house-edge-bayes-rule-v2",
        "fitted_on": {"splits": list(splits), "n_machines": len(q)},
        "encoding": enc, "encoding_threshold": ENCODINGS[enc],
        "encoding_evidence": {e: float(v["evidence"]) for e, v in results.items()},
        "settings": SETTINGS,
        "rules": [[list(rr), float(lp)] for rr, lp in zip(RULES, S["logpi"])],
        "family_prior": fam,
        "flip_hist": S["h"].tolist(), "p_flip_long": S["p_flip"][False], "p_flip_short": S["p_flip"][True],
        "rates": rates, "type_rates": type_rates, "strength": strength,
        "share1": share1, "hi_mix": hi_mix, "hi_strength": hi_strength,
    }, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", default="train")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    hp, r = fit(args.data, tuple(args.splits.split(",")))
    Path(args.out).write_text(json.dumps(hp, indent=1) + "\n")
    r.to_csv(Path(args.out).with_suffix(".map.csv"), index=False)
    top = sorted(hp["rules"], key=lambda x: -x[1])[:6]
    print("family prior:", {k: round(v, 3) for k, v in hp["family_prior"].items()})
    print("top rules:", [(tuple(rr), round(float(np.exp(lp)), 3)) for rr, lp in top])
    print(f"p_flip long {hp['p_flip_long']:.3f}, short {hp['p_flip_short']:.3f}; wrote {args.out}")


if __name__ == "__main__":
    main()
