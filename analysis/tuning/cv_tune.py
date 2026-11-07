"""Cross-validated fine-tuning of the Bayesian rule model.

Questions:
  B. Do priors fitted on train + validation histories beat train-only priors?
  C. Do tuned post-processing settings (episode offset half-life/variance, machine offset variance,
     1x-share strength) and a rate-prior strength scale beat the current settings?
  D. Both.
Protocol: validation episodes split into 4 folds, stratified by condition (6 episodes per condition per
fold). Every number reported is out-of-fold: a fold's queries are scored with priors and settings that
never saw that fold's histories or answers. Settings are chosen on the other 3 folds (nested).

    python analysis/tuning/cv_tune.py        # ~5-10 min
"""
from itertools import product
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/discovery"))
import prototype_model as pm  # noqa: E402
from evaluate import condition_errors  # noqa: E402

OUT = ROOT / "analysis/results/tuning"
CURRENT = {"ep_half_life": 6.0, "ep_var": 0.2, "machine_var": 0.05, "share_strength": 60.0, "rate_scale": 1.0}
GRID = {"ep_half_life": [3.0, 6.0, 12.0], "ep_var": [0.1, 0.2, 0.4], "machine_var": [0.0, 0.02, 0.05, 0.1],
        "share_strength": [30.0, 60.0, 120.0], "rate_scale": [0.5, 1.0, 2.0]}
K = 4


# ---------- priors from an arbitrary set of machines ----------

def map_rows(split):
    q, Y, W = pm.load_split(split)
    rows = []
    for m in range(len(q)):
        c = int(q.loc[m, "cutoff"]); b = pm.map_fit(W[m], c)
        rows.append({"machine_id": q.loc[m, "machine_id"], "episode_id": q.loc[m, "episode_id"],
                     "type": q.loc[m, "machine_type"], "signal": q.loc[m, "signal"], "short": q.loc[m, "short"],
                     "pair": b[1], "flip_before_cutoff": (c - b[2]) if b[2] > 0 else np.nan,
                     "kf": b[4], "nf": b[5], "ku": b[6], "nu": b[7]})
    r = pd.DataFrame(rows)
    ys = {mid: Y[m, :q.loc[m, "cutoff"]] for m, mid in enumerate(q["machine_id"])}
    return r, ys


def hp_from(r, ys):
    """Same estimators as prototype_model.fit_hyper, on any subset of machines."""
    counts = r["pair"].value_counts().reindex(pm.PAIRS).fillna(0) + 0.5
    hp = {"pair_logprior": np.log(counts / counts.sum()).to_dict()}
    g = r.groupby(["type", "signal"])[["kf", "nf", "ku", "nu"]].sum()
    hp["rates"] = {k: (v.kf / v.nf, v.ku / v.nu) for k, v in g.iterrows()}
    hp["strength"] = {}
    for typ in pm.TYPES:
        d = r[(r["type"] == typ) & ~r["short"]]
        s = []
        for k, n in (("kf", "nf"), ("ku", "nu")):
            p = d[k] / d[n]; pbar = d[k].sum() / d[n].sum()
            var_true = max(p.var() - (pbar * (1 - pbar) / d[n]).mean(), 1e-6)
            s.append(float(np.clip(pbar * (1 - pbar) / var_true - 1, 10, 300)))
        hp["strength"][typ] = tuple(s)
    flips = r["flip_before_cutoff"].dropna()
    hist = pd.Series(np.histogram(flips, bins=np.arange(pm.FLIP_MIN, pm.FLIP_MAX + 1))[0],
                     index=np.arange(pm.FLIP_MIN, pm.FLIP_MAX)) + 0.5
    hp["flip_prior"] = hist / hist.sum()
    hp["p_flip_long"] = float(r[~r["short"]]["flip_before_cutoff"].notna().mean())
    hp["p_flip_short"] = float(r[r["short"]]["flip_before_cutoff"].notna().mean())
    vals = {typ: np.concatenate([ys[mid] for mid, t in zip(r["machine_id"], r["type"]) if t == typ]) for typ in pm.TYPES}
    hp["share1"] = {t: float((v[v > 0] == 1).mean()) for t, v in vals.items()}
    hp["hi_mix"] = {t: np.array([(v == k).sum() for k in (2, 3, 4)]) / (v >= 2).sum() for t, v in vals.items()}
    return hp


# ---------- cached posterior + fast post-processing ----------

def posterior_cache(hp, q, Y, W, machines, rate_scale):
    """Per machine: base log-odds, residual summaries for offsets, win counts for the 1x share."""
    out = {}
    for m in machines:
        typ, sg = q.loc[m, "machine_type"], q.loc[m, "signal"]
        pf, pu = hp["rates"][(typ, sg)]; sf, su = hp["strength"][typ]
        sf, su = sf * rate_scale, su * rate_scale
        rp = (pf * sf, (1 - pf) * sf, pu * su, (1 - pu) * su)
        p_flip = hp["p_flip_short"] if q.loc[m, "short"] else hp["p_flip_long"]
        c = int(q.loc[m, "cutoff"])
        L, P, H = pm.hypotheses(W[m], c, hp, rp, p_flip)
        w = np.exp(L - L.max()); pw = (w @ P) / w.sum()
        tt, yy, pp = pm.map_states(W[m], c, H[int(np.argmax(L))])
        y = Y[m, :c]; wins = y[y > 0]
        out[m] = {"z": np.log(pw / (1 - pw)), "age": c - 1 - tt, "r": yy - pp, "info": pp * (1 - pp),
                  "ones": int((wins == 1).sum()), "nwins": len(wins), "type": typ}
    return out


def apply(cache, hp, q, machines, s):
    machines = list(machines)
    z = {m: cache[m]["z"] for m in machines}
    eps = q.loc[machines].groupby("episode_id").groups
    for _, idx in eps.items():
        num = den = 0.0
        for m in idx:
            wt = 2.0 ** (-cache[m]["age"] / s["ep_half_life"])
            num += (wt * cache[m]["r"]).sum(); den += (wt * cache[m]["info"]).sum()
        shift = num / (den + 1 / s["ep_var"])
        for m in idx:
            z[m] += shift
            if s["machine_var"] > 0:
                z[m] += cache[m]["r"].sum() / (cache[m]["info"].sum() + 1 / s["machine_var"])
    pred = np.zeros((len(machines), 5))
    for k, m in enumerate(machines):
        c = cache[m]; pw = 1 / (1 + np.exp(-z[m]))
        sh = (c["ones"] + s["share_strength"] * hp["share1"][c["type"]]) / (c["nwins"] + s["share_strength"])
        pred[k] = np.r_[1 - pw, pw * sh, pw * (1 - sh) * hp["hi_mix"][c["type"]]]
    return pred


def R_of(q, machines, pred):
    sub = q.loc[list(machines)]
    truth = sub[pm.Q].to_numpy(); ref = sub[["ref_" + c for c in pm.Q]].to_numpy()
    e = condition_errors(sub, ((pred - truth) ** 2).sum(1)); b = condition_errors(sub, ((ref - truth) ** 2).sum(1))
    return float((e / b).mean())


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    q, Y, W = pm.load_split("validation")
    r_tr, ys_tr = map_rows("train")
    r_va, ys_va = map_rows("validation")
    print(f"MAP fits done in {time.time() - t0:.0f}s", flush=True)
    hp_train = hp_from(r_tr, ys_tr)
    # Sanity: identical to the shipped estimator
    ref_hp = pm.fit_hyper()
    assert all(np.allclose(hp_train["rates"][k], ref_hp["rates"][k]) for k in ref_hp["rates"])
    assert np.allclose(hp_train["flip_prior"], ref_hp["flip_prior"])

    # Stratified episode folds
    ep_cond = q.groupby("episode_id")["condition"].first()
    fold_of = {}
    for cond, eps in ep_cond.groupby(ep_cond):
        for k, e in enumerate(sorted(eps.index, key=lambda e: hashlib.sha256(e.encode()).hexdigest())):
            fold_of[e] = k % K
    q["fold"] = q["episode_id"].map(fold_of)
    folds = [np.flatnonzero(q["fold"] == f) for f in range(K)]

    combos = [dict(zip(GRID, v)) for v in product(*GRID.values())]
    scales = GRID["rate_scale"]
    # Train-only priors: one cache per rate scale over all validation machines.
    cache_A = {sc: posterior_cache(hp_train, q, Y, W, range(len(q)), sc) for sc in scales}
    print(f"train-prior posteriors cached at {time.time() - t0:.0f}s", flush=True)
    # Train + other-folds priors: per fold, cache for that fold's machines.
    ys_all = {**ys_tr, **ys_va}
    cache_B = {}
    for f in range(K):
        keep = r_va["episode_id"].map(fold_of) != f
        hp_f = hp_from(pd.concat([r_tr, r_va[keep]], ignore_index=True), ys_all)
        cache_B[f] = (hp_f, {sc: posterior_cache(hp_f, q, Y, W, folds[f], sc) for sc in scales})
    print(f"train+val-prior posteriors cached at {time.time() - t0:.0f}s", flush=True)

    # Settings score on train-only priors, per fold (used for nested selection).
    table = []
    for ci, s in enumerate(combos):
        for f in range(K):
            pred = apply(cache_A[s["rate_scale"]], hp_train, q, folds[f], s)
            table.append({"combo": ci, "fold": f, "R": R_of(q, folds[f], pred)})
    tab = pd.DataFrame(table)
    print(f"settings grid scored at {time.time() - t0:.0f}s", flush=True)

    def oof(variant):
        preds = np.zeros((len(q), 5)); per_fold = []; chosen = []
        for f in range(K):
            if variant in ("C", "D"):
                inner = tab[tab["fold"] != f].groupby("combo")["R"].mean()
                s = combos[int(inner.idxmin())]
            else:
                s = CURRENT
            chosen.append(s)
            if variant in ("A", "C"):
                p = apply(cache_A[s["rate_scale"]], hp_train, q, folds[f], s)
            else:
                hp_f, cb = cache_B[f]
                p = apply(cb[s["rate_scale"]], hp_f, q, folds[f], s)
            preds[folds[f]] = p
            per_fold.append(R_of(q, folds[f], p))
        return preds, per_fold, chosen

    res = {}
    for v in "ABCD":
        preds, per_fold, chosen = oof(v)
        R = R_of(q, range(len(q)), preds)
        res[v] = {"R": R, "points": 60 / (1 + R), "per_fold_R": per_fold, "chosen": chosen}
    names = {"A": "current (train priors, current settings)", "B": "train+val priors, current settings",
             "C": "train priors, tuned settings", "D": "train+val priors, tuned settings"}
    print()
    for v in "ABCD":
        d = np.array(res[v]["per_fold_R"]) - np.array(res["A"]["per_fold_R"])
        print(f"{v}. {names[v]:44s} points {res[v]['points']:.3f}  R {res[v]['R']:.4f}  "
              f"fold dR vs A {np.round(d, 4)}  folds better {int((d < 0).sum())}/4")
    for v in "CD":
        print(f"settings chosen per fold ({v}):")
        for s in res[v]["chosen"]:
            print("   ", s)
    best_all = tab.groupby("combo")["R"].mean().sort_values()
    print("\ntop settings on all 4 folds (train priors; in-sample, for reference):")
    for ci in best_all.index[:5]:
        print(f"   R {best_all[ci]:.4f}  {combos[ci]}")
    cur = [ci for ci, s in enumerate(combos) if s == CURRENT][0]
    print(f"   current settings: R {best_all[cur]:.4f}, rank {list(best_all.index).index(cur) + 1}/{len(combos)}")
    json.dump({v: {k: x for k, x in res[v].items()} for v in res}, open(OUT / "cv_results.json", "w"), indent=1, default=float)
    tab.assign(**{k: tab["combo"].map(lambda c: combos[c][k]) for k in GRID}).to_csv(OUT / "settings_grid.csv", index=False)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
