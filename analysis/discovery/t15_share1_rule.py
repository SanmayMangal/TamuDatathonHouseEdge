import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from itertools import combinations
from discovery_lib import *
PAIRS = list(combinations(range(1, 9), 2))
out = []
for split in ("train", "validation"):
    q, Y = load_split(split); W = win_grid(Y)
    for m in range(len(q)):
        if q.loc[m, "short"]: continue
        c = q.loc[m, "cutoff"]; t = np.arange(8, c)
        y = Y[m, t]; wins = y > 0
        z = (y == 1).astype(float)
        best = (0, None, 0)
        singles = {}
        for k in range(1, 9):
            a = W[m, t - k][wins]; zz = z[wins]
            if a.std() > 0:
                d = zz[a == 1].mean() - zz[a == 0].mean(); se = np.sqrt(zz.var() * (1/(a==1).sum() + 1/(a==0).sum()))
                singles[k] = d / se if se > 0 else 0
        for i, j in PAIRS:
            same = (W[m, t - i] == W[m, t - j])[wins]; zz = z[wins]
            if 0 < same.sum() < len(same):
                d = zz[same].mean() - zz[~same].mean(); se = np.sqrt(zz.var() * (1/same.sum() + 1/(~same).sum()))
                zz_ = d / se if se > 0 else 0
                if abs(zz_) > abs(best[0]): best = (zz_, (i, j), d)
        out.append({"split": split, "type": q.loc[m, "machine_type"], "cond": q.loc[m, "condition"],
                    "best_z": best[0], "pair": best[1], "d": best[2],
                    "best_single_z": max(singles.values(), key=abs) if singles else np.nan,
                    "nwins": int(wins.sum())})
r = pd.DataFrame(out)
print("1x-share rule search (per machine, wins only), |best pair z| quantiles:", np.round(np.quantile(r["best_z"].abs(), [.5, .75, .9, .95]), 2))
print("share |z|>4:", (r["best_z"].abs() > 4).mean().round(3), " (28 pairs searched: chance rate of |z|>4 ~ 0.002)")
print("top pairs among |z|>4:", r[r["best_z"].abs() > 4]["pair"].value_counts().head(8).to_dict())
print("single-lag |z| quantiles:", np.round(np.quantile(r["best_single_z"].abs(), [.5, .9, .95]), 2))
print("by type share |z|>4:", r.assign(s=r["best_z"].abs() > 4).groupby("type")["s"].mean().round(3).to_dict())
