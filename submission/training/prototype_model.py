"""Training-side copy of analysis/discovery/prototype_model.py (used only to fit hyperparameters).

Prototype of the House Edge model built from the discovered structure (numpy/scipy; not the submission).

Generative structure found in the data (see analysis/discovery/README.md):
  * Each machine's win chance (payout > 0) is set by a "matching pair" rule: whether its win/loss
    outcomes at two lags (i, j) were equal. One side of the rule is favorable (high win chance),
    the other unfavorable. Pairs seen: (1,3), (2,5), (4,7); direction is about 50/50 per machine.
  * Drift / shared change: the rule's direction flips once, abruptly, 40-104 ticks before the cutoff
    (shared change: all machines in the episode on the same tick; drift: each machine on its own tick).
    Rates in each state don't change at the flip.
  * State rates depend on machine type and signal (about +0.35 log-odds per signal unit), with mild
    per-machine variation and a slowly wandering episode-wide offset.
  * Given a win, the 2x:5x:20x split is fixed per type; only the 1x share varies, as a persistent
    per-machine level plus unpredictable tick-level noise.

The model: for each machine, a Bayesian posterior over hypotheses (pair, direction, flip tick or none)
with Beta-Bernoulli marginal likelihoods; predictive win chance at the cutoff; then an episode-level
and a machine-level log-odds offset; then the size mix. Every hyperparameter is estimated from the
train split; validation is scored exactly against its true probabilities.

    python analysis/discovery/prototype_model.py            # ~1 min
"""
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import betaln

ROOT = Path(__file__).resolve().parents[2]
try:  # only needed for exact_score() when run inside the repo
    sys.path.insert(0, str(ROOT / "analysis"))
    from evaluate import condition_errors  # noqa: E402
except ImportError:  # pragma: no cover
    condition_errors = None

PUB = ROOT / "main-participants-60-40/public"
PAY = np.array([0, 1, 2, 5, 20])
Q = ["p_0", "p_1", "p_2", "p_5", "p_20"]
PAIRS = list(combinations(range(1, 9), 2))
TYPES = ("steady", "balanced", "jackpot")

# Tunables (light tuning on validation so far; re-tune with episode folds in Phase 2).
FLIP_MIN, FLIP_MAX = 36, 115      # flip-tick search window, ticks before cutoff (train range 40-104)
EP_HALF_LIFE, EP_VAR = 6.0, 0.2   # episode offset: recency half-life, prior variance (log-odds^2)
MACHINE_VAR = 0.05                # per-machine offset prior variance
SHARE_STRENGTH = 60.0             # 1x-share shrinkage toward the type mean (pseudo-wins)


def load_split(split, public=PUB):
    d = public / split
    q = pd.read_csv(d / "queries.csv")
    h = pd.read_csv(d / "history.csv")
    for name, key in (("labels.csv", None), ("conditions.csv", None)):
        if (d / name).exists():
            q = q.merge(pd.read_csv(d / name), on="query_id")
    if (d / "targets.csv").exists():
        q = q.merge(pd.read_csv(d / "targets.csv").drop(columns=["episode_id", "outcome"]), on="query_id")
    if (d / "reference.csv").exists():
        q = q.merge(pd.read_csv(d / "reference.csv").rename(columns={c: "ref_" + c for c in Q}), on="query_id")
    q = q.sort_values(["episode_id", "machine_id"]).reset_index(drop=True)
    Y = np.full((len(q), int(q["cutoff"].max()) + 1), -1)
    Y[pd.Index(q["machine_id"]).get_indexer(h["machine_id"]), h["tick"].to_numpy()] = np.searchsorted(PAY, h["payout"].to_numpy())
    q["short"] = [(Y[m, :c] < 0).any() for m, c in enumerate(q["cutoff"])]
    W = np.where(Y >= 0, (Y > 0).astype(float), np.nan)
    return q, Y, W


def pair_series(w, c, i, j):
    t = np.arange(8, c)
    a, b, y = w[t - i], w[t - j], w[t]
    ok = ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(y)
    xa, xb = w[c - i], w[c - j]
    now = None if (np.isnan(xa) or np.isnan(xb)) else bool(xa == xb)
    return t[ok], (a == b)[ok], y[ok], now


# ---------------- Hyperparameters from the training split ----------------

def map_fit(w, c, flip_penalty=3.0):
    """Maximum a posteriori (pair, flip tick, direction) with flat Beta(1,1) rates."""
    best = None
    for i, j in PAIRS:
        tt, mm, yy, _ = pair_series(w, c, i, j)
        for tau in [-1] + list(range(c - FLIP_MAX, c - FLIP_MIN + 1)):
            pre = tt < tau
            for d in (True, False):
                fav = np.where(pre, mm != d, mm == d)
                kf, nf, ku, nu = yy[fav].sum(), fav.sum(), yy[~fav].sum(), (~fav).sum()
                if (kf + 1) / (nf + 2) < (ku + 1) / (nu + 2):
                    continue  # "favorable" must be the higher-rate side
                ll = betaln(1 + kf, 1 + nf - kf) + betaln(1 + ku, 1 + nu - ku) - (0 if tau == -1 else flip_penalty)
                if best is None or ll > best[0]:
                    best = (ll, (i, j), tau, d, kf, nf, ku, nu)
    return best


def fit_hyper(public=PUB):
    q, Y, W = load_split("train", public)
    rows = []
    for m in range(len(q)):
        c = int(q.loc[m, "cutoff"])
        b = map_fit(W[m], c)
        rows.append({"type": q.loc[m, "machine_type"], "signal": q.loc[m, "signal"], "short": q.loc[m, "short"],
                     "pair": b[1], "flip_before_cutoff": (c - b[2]) if b[2] > 0 else np.nan,
                     "kf": b[4], "nf": b[5], "ku": b[6], "nu": b[7]})
    r = pd.DataFrame(rows)
    counts = r["pair"].value_counts().reindex(PAIRS).fillna(0) + 0.5
    hp = {"pair_logprior": np.log(counts / counts.sum()).to_dict()}
    g = r.groupby(["type", "signal"])[["kf", "nf", "ku", "nu"]].sum()
    hp["rates"] = {k: (v.kf / v.nf, v.ku / v.nu) for k, v in g.iterrows()}
    # Between-machine dispersion of state rates -> Beta prior strengths (method of moments, clipped).
    hp["strength"] = {}
    for typ in TYPES:
        d = r[(r["type"] == typ) & ~r["short"]]
        s = []
        for k, n in (("kf", "nf"), ("ku", "nu")):
            p = d[k] / d[n]; pbar = d[k].sum() / d[n].sum()
            var_true = max(p.var() - (pbar * (1 - pbar) / d[n]).mean(), 1e-6)
            s.append(float(np.clip(pbar * (1 - pbar) / var_true - 1, 10, 300)))
        hp["strength"][typ] = tuple(s)
    flips = r["flip_before_cutoff"].dropna()
    hist = pd.Series(np.histogram(flips, bins=np.arange(FLIP_MIN, FLIP_MAX + 1))[0],
                     index=np.arange(FLIP_MIN, FLIP_MAX)) + 0.5
    hp["flip_prior"] = hist / hist.sum()
    hp["p_flip_long"] = float(r[~r["short"]]["flip_before_cutoff"].notna().mean())
    hp["p_flip_short"] = float(r[r["short"]]["flip_before_cutoff"].notna().mean())
    ys = {typ: np.concatenate([Y[m, :q.loc[m, "cutoff"]] for m in np.flatnonzero(q["machine_type"] == typ)]) for typ in TYPES}
    hp["share1"] = {t: float((v[v > 0] == 1).mean()) for t, v in ys.items()}
    hp["hi_mix"] = {t: np.array([(v == k).sum() for k in (2, 3, 4)]) / (v >= 2).sum() for t, v in ys.items()}
    hp["map_table"] = r
    return hp


# ---------------- Prediction ----------------

def hypotheses(w, c, hp, rate_prior, p_flip):
    af, bf, au, bu = rate_prior
    fp = hp["flip_prior"]
    taus = np.r_[-1, c - fp.index.to_numpy()]
    ltau = np.r_[np.log(max(1 - p_flip, 1e-12)), np.log(max(p_flip, 1e-12)) + np.log(fp.to_numpy())]
    L, P, H = [], [], []
    for (i, j), lp in hp["pair_logprior"].items():
        tt, mm, yy, now = pair_series(w, c, i, j)
        pre = tt[None, :] < taus[:, None]
        for d in (True, False):
            fav = np.where(pre, mm[None] != d, mm[None] == d)
            kf = (fav * yy).sum(1); nf = fav.sum(1); ku = (~fav * yy).sum(1); nu = (~fav).sum(1)
            ll = betaln(af + kf, bf + nf - kf) - betaln(af, bf) + betaln(au + ku, bu + nu - ku) - betaln(au, bu)
            pf, pu = (af + kf) / (af + bf + nf), (au + ku) / (au + bu + nu)
            pw = 0.5 * (pf + pu) if now is None else (pf if now == d else pu)
            L.append(ll + ltau + lp + np.log(0.5)); P.append(pw)
            H += [((i, j), d, tau, pf_, pu_) for tau, pf_, pu_ in zip(taus, pf, pu)]
    return np.concatenate(L), np.concatenate(P), H


def map_states(w, c, hyp):
    (i, j), d, tau, pf, pu = hyp
    tt, mm, yy, _ = pair_series(w, c, i, j)
    fav = np.where(tt < tau, mm != d, mm == d)
    return tt, yy, np.where(fav, pf, pu)


def predict(split, hp, public=PUB, ep_var=EP_VAR, machine_var=MACHINE_VAR, share_strength=SHARE_STRENGTH):
    q, Y, W = load_split(split, public)
    n = len(q)
    logit = np.zeros(n); series = {}
    for m in range(n):
        typ, sg = q.loc[m, "machine_type"], q.loc[m, "signal"]
        pf, pu = hp["rates"][(typ, sg)]; sf, su = hp["strength"][typ]
        rp = (pf * sf, (1 - pf) * sf, pu * su, (1 - pu) * su)
        p_flip = hp["p_flip_short"] if q.loc[m, "short"] else hp["p_flip_long"]
        c = int(q.loc[m, "cutoff"])
        L, P, H = hypotheses(W[m], c, hp, rp, p_flip)
        wgt = np.exp(L - L.max()); pw = (wgt @ P) / wgt.sum()
        logit[m] = np.log(pw / (1 - pw))
        series[m] = map_states(W[m], c, H[int(np.argmax(L))])
    for _, idx in q.groupby("episode_id").groups.items():
        idx = list(idx)
        num = den = 0.0
        for m in idx:
            tt, yy, pp = series[m]
            wt = 2.0 ** (-(int(q.loc[m, "cutoff"]) - 1 - tt) / EP_HALF_LIFE)
            num += (wt * (yy - pp)).sum(); den += (wt * pp * (1 - pp)).sum()
        if ep_var > 0:
            logit[idx] += num / (den + 1 / ep_var)
        if machine_var > 0:
            for m in idx:
                tt, yy, pp = series[m]
                logit[m] += (yy - pp).sum() / ((pp * (1 - pp)).sum() + 1 / machine_var)
    pw = 1 / (1 + np.exp(-logit))
    S = np.empty(n)
    for m in range(n):
        y = Y[m, :q.loc[m, "cutoff"]]; y = y[y > 0]; typ = q.loc[m, "machine_type"]
        S[m] = ((y == 1).sum() + share_strength * hp["share1"][typ]) / (len(y) + share_strength)
    hi = np.vstack([hp["hi_mix"][t] for t in q["machine_type"]])
    return q, np.column_stack([1 - pw, pw * S, (pw * (1 - S))[:, None] * hi])


def exact_score(q, pred, label=""):
    truth = q[Q].to_numpy(); ref = q[["ref_" + x for x in Q]].to_numpy()
    e = condition_errors(q, ((pred - truth) ** 2).sum(1)); b = condition_errors(q, ((ref - truth) ** 2).sum(1))
    r = e / b; R = float(r.mean())
    print(f"{label:40s} points {60 / (1 + R):6.2f}  R {R:.4f}  " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
    return R


if __name__ == "__main__":
    hp = fit_hyper()
    q, pred = predict("validation", hp)
    exact_score(q, pred, "prototype (train hyperparameters)")
    q, pred = predict("validation", hp, ep_var=0, machine_var=0)
    exact_score(q, pred, "  without episode/machine offsets")
