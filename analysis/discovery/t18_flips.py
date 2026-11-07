import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from discovery_lib import *
PAIRS = [(1, 3), (2, 5), (4, 7)]
def ll_bern(y, p): p = np.clip(p, 1e-4, 1 - 1e-4); return (y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
def fit_machine(w, c):
    t = np.arange(8, c); y = w[t]
    best = None
    for i, j in PAIRS:
        x = 2.0 * (w[t - i] == w[t - j]) - 1  # +1 match
        h = len(t) // 2
        zs = []
        for seg in (slice(0, h), slice(h, None)):
            yy, xx = y[seg], x[seg]
            d = yy[xx == 1].mean() - yy[xx == -1].mean(); zs.append(abs(d))
        if best is None or max(zs) > best[1]: best = ((i, j), max(zs), x)
    (i, j), _, x = best
    n = len(t)
    # model A: no change: p = sig(a + g x)
    def fit_static(yy, xx):
        p1 = (yy[xx == 1].sum() + .5) / ((xx == 1).sum() + 1); p0 = (yy[xx == -1].sum() + .5) / ((xx == -1).sum() + 1)
        return ll_bern(yy, np.where(xx == 1, p1, p0)), p1, p0
    llA, p1, p0 = fit_static(y, x)
    # model B: abrupt change at tau (both state rates refit per segment)
    bestB = (-np.inf, None)
    for tau in range(16, n - 16):
        l = fit_static(y[:tau], x[:tau])[0] + fit_static(y[tau:], x[tau:])[0]
        if l > bestB[0]: bestB = (l, tau)
    # model C: linear ramp of amplitude: logit p = a + (g0 + g1 * s) x, s in [0,1]; grid over g0, g1 with fixed a
    s = np.arange(n) / (n - 1)
    a = np.log(y.mean() / (1 - y.mean()))
    bestC = -np.inf
    for g0 in np.linspace(-3, 3, 25):
        for g1 in np.linspace(-6, 6, 49):
            for da in (-1, -0.5, 0, 0.5, 1):
                l = ll_bern(y, 1 / (1 + np.exp(-(a + da + (g0 + g1 * s) * x))))
                if l > bestC: bestC = l; parC = (g0, g1)
    # model B with same 2-param logistic form for fairness: sign flip abrupt at tau, fixed |g|
    return {"pair": f"{i}-{j}", "llA": llA, "llB": bestB[0], "tauB": t[bestB[1]] if bestB[1] else np.nan, "llC": bestC,
            "g0": parC[0], "g1": parC[1], "n": n}
rows = []
for split in ("train", "validation"):
    q, Y = load_split(split); W = win_grid(Y)
    for m in range(len(q)):
        if q.loc[m, "short"]: continue
        r = fit_machine(W[m], q.loc[m, "cutoff"])
        r.update(split=split, ep=q.loc[m, "episode_id"], cond=q.loc[m, "condition"], cutoff=q.loc[m, "cutoff"], mid=q.loc[m, "machine_id"])
        rows.append(r)
r = pd.DataFrame(rows)
r.to_pickle("flips.pkl")
r["B_vs_A"] = r["llB"] - r["llA"]; r["C_vs_A"] = r["llC"] - r["llA"]
r["flip"] = (r["g0"] * (r["g0"] + r["g1"]) < 0)
print(r.groupby("cond")[["B_vs_A", "C_vs_A"]].median().round(2).to_string())
print("\nshare where abrupt (B, 4 extra params) beats ramp (C) by >2:", r.assign(x=(r["llB"] - r["llC"]) > 2).groupby("cond")["x"].mean().round(3).to_dict())
print("share where ramp beats abrupt by >2:", r.assign(x=(r["llC"] - r["llB"]) > 2).groupby("cond")["x"].mean().round(3).to_dict())
print("share with sign change under ramp:", r.groupby("cond")["flip"].mean().round(3).to_dict())
