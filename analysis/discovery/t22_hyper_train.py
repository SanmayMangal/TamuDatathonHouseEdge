import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
"""Estimate state rates by type x signal from TRAIN histories via MAP rule assignment."""
from itertools import combinations
from scipy.special import betaln
from discovery_lib import *
PAIRS = list(combinations(range(1, 9), 2))
def map_fit(w, c):
    t = np.arange(8, c); best = None
    for i, j in PAIRS:
        a, b, y = w[t - i], w[t - j], w[t]; ok = ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(y)
        tt, mm, yy = t[ok], (a == b)[ok], y[ok]
        for tau in [-1] + list(range(c - 110, c - 39)):
            pre = tt < tau
            for d in (1, -1):
                fav = np.where(pre, mm != (d == 1), mm == (d == 1))
                kf, nf, ku, nu = yy[fav].sum(), fav.sum(), yy[~fav].sum(), (~fav).sum()
                ll = betaln(1 + kf, 1 + nf - kf) + betaln(1 + ku, 1 + nu - ku) - (0 if tau == -1 else 3.0)
                if (kf + 1) / (nf + 2) < (ku + 1) / (nu + 2): ll -= 50  # favorable must be the higher-rate side
                if best is None or ll > best[0]: best = (ll, (i, j), tau, d, kf, nf, ku, nu)
    return best
q, Y = load_split("train"); W = win_grid(Y)
rows = []
for m in range(len(q)):
    b = map_fit(W[m], int(q.loc[m, "cutoff"]))
    rows.append({"type": q.loc[m, "machine_type"], "signal": q.loc[m, "signal"], "cond": q.loc[m, "condition"], "short": q.loc[m, "short"],
                 "pair": b[1], "tau_before_cutoff": (q.loc[m, "cutoff"] - b[2]) if b[2] > 0 else np.nan, "dir": b[3],
                 "kf": b[4], "nf": b[5], "ku": b[6], "nu": b[7]})
r = pd.DataFrame(rows); r.to_pickle("train_map.pkl")
print("pair freq:", r["pair"].value_counts(normalize=True).head(6).round(3).to_dict())
print("direction share (+1 = match favorable):", r["dir"].eq(1).mean().round(3))
print("change found share by cond:", r.assign(ch=r["tau_before_cutoff"].notna()).groupby("cond")["ch"].mean().round(3).to_dict())
print("tau before cutoff quantiles:", r["tau_before_cutoff"].quantile([0, .05, .5, .95, 1]).to_dict())
g = r.groupby(["type", "signal"])[["kf", "nf", "ku", "nu"]].sum()
g["pf"] = g["kf"] / g["nf"]; g["pu"] = g["ku"] / g["nu"]
lg = lambda p: np.log(p / (1 - p))
g["lf"], g["lu"] = lg(g["pf"]), lg(g["pu"])
print(g[["pf", "pu", "lf", "lu"]].round(3).to_string())
# per-machine dispersion of rates: method of moments for beta strength
for typ in ["steady", "balanced", "jackpot"]:
    d = r[(r["type"] == typ) & (~r["short"])]
    for kcol, ncol, name in (("kf", "nf", "fav"), ("ku", "nu", "unf")):
        p = d[kcol] / d[ncol]; pbar = d[kcol].sum() / d[ncol].sum()
        var_obs = p.var(); var_bin = (pbar * (1 - pbar) / d[ncol]).mean()
        var_true = max(var_obs - var_bin, 1e-6)
        strength = pbar * (1 - pbar) / var_true - 1
        print(f"{typ:9s} {name}: mean {pbar:.3f}, between-machine sd {np.sqrt(var_true):.3f}, implied beta strength {strength:.1f}")
