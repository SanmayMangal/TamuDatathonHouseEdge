import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from discovery_lib import *
q = pd.read_pickle("val_state3.pkl")
_, Y = load_split("validation")
t = q[Q].to_numpy(); s_true = t[:, 1] / t[:, 1:].sum(1)
c = q["cutoff"].to_numpy()
hs, nw = [], []
for m in range(len(q)):
    y = Y[m, :c[m]]; w = y[y > 0]; hs.append(((w == 1).sum() + .5) / (len(w) + 1)); nw.append(len(w))
hs = np.array(hs); nw = np.array(nw)
q["s_true"] = s_true; q["s_hist"] = hs; q["nw"] = nw
lg = lambda p: np.log(p / (1 - p))
for (cond, typ), g in q.groupby(["condition", "machine_type"]):
    print(f"{cond:14s} {typ:9s} corr(logit) {np.corrcoef(lg(g['s_true']), lg(g['s_hist']))[0,1]:.3f}  mean true {g['s_true'].mean():.3f} hist {g['s_hist'].mean():.3f}  median wins {g['nw'].median():.0f}")
# by state
print(q.groupby(["machine_type", "fav"])[["s_true", "s_hist"]].mean().round(3).to_string())
