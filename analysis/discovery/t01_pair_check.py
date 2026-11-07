import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from itertools import combinations
import numpy as np, pandas as pd
from phase1.common import load, lag
ctx = load()
h = ctx.hist; w = (ctx.y > 0).astype(float)
X = {k: lag(ctx, k, "win") for k in range(1, 9)}
mid = h["machine_id"].astype(str).to_numpy()
rows = []
for i, j in combinations(range(1, 9), 2):
    ok = ~np.isnan(X[i]) & ~np.isnan(X[j])
    same = (X[i] == X[j])[ok]
    d = pd.DataFrame({"m": mid[ok], "same": same, "w": w[ok]})
    g = d.groupby(["m", "same"])["w"].agg(["mean", "size"]).unstack("same")
    p1, p0 = g[("mean", True)], g[("mean", False)]
    n1, n0 = g[("size", True)], g[("size", False)]
    pp = (p1*n1 + p0*n0)/(n1+n0); se = np.sqrt(pp*(1-pp)*(1/n1+1/n0))
    rows.append(pd.DataFrame({"m": g.index, "pair": f"{i}-{j}", "d_same_minus_diff": (p1-p0).values, "z": ((p1-p0)/se).values,
                              "p_win_same": p1.values, "p_win_diff": p0.values}))
r = pd.concat(rows)
meta = h.groupby("machine_id", observed=True).agg(cond=("condition","first"), typ=("machine_type","first"), sig=("signal","first"), short=("short_history","first"))
meta.index = meta.index.astype(str)
best = r.loc[r["z"].abs().groupby(r["m"]).idxmax().dropna()] if False else r.dropna(subset=["z"]).sort_values("z", key=abs, ascending=False).groupby("m").head(1).set_index("m")
second = r.dropna(subset=["z"]).sort_values("z", key=abs, ascending=False).groupby("m").nth(1).set_index("m")["z"].abs()
best = best.join(meta); best["second_abs_z"] = second
# also lag-7 single
x7 = X[7]; ok = ~np.isnan(x7)
d7 = pd.DataFrame({"m": mid[ok], "x": x7[ok], "w": w[ok]}).groupby(["m","x"])["w"].mean().unstack()
best["d_lag7"] = (d7[1.0]-d7[0.0]).reindex(best.index)
long = best[~best["short"]]
print("LONG-HISTORY machines:", len(long))
print("share whose best pair |z|>4:", (long["z"].abs()>4).mean().round(3))
print("best pair counts (|z|>4):"); print(long[long["z"].abs()>4]["pair"].value_counts().head(10).to_string())
print("sign of best pair effect (|z|>4): same-better share", (long[long["z"].abs()>4]["z"]>0).mean().round(3))
s = long[long["z"].abs()>4]
print("median |z| best vs second:", s["z"].abs().median().round(1), s["second_abs_z"].median().round(1))
print("median P(win|same), P(win|diff) by sign:")
print(s.assign(pos=s["z"]>0).groupby("pos")[["p_win_same","p_win_diff"]].median().round(3).to_string())
print("best pair by condition (|z|>4 share):"); print(long.assign(strong=long["z"].abs()>4).groupby("cond")["strong"].mean().round(3).to_string())
print("does pair-sign relate to signal or type?"); print(s.assign(pos=s["z"]>0).groupby(["typ"], observed=True)["pos"].mean().round(2).to_string()); print(s.assign(pos=s["z"]>0).groupby("sig")["pos"].mean().round(2).to_string())
print("pair distribution by type:"); print(pd.crosstab(s["pair"], s["typ"]).loc[lambda t: t.sum(1)>5].to_string())
# lag-7 among machines whose best pair includes 7 vs not
s2 = s.assign(has7=s["pair"].str.contains("7"))
print("median lag-7 single effect, best pair has 7 / not:", s2.groupby("has7")["d_lag7"].median().round(3).to_dict())
