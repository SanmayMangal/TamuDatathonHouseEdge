import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from itertools import combinations
from discovery_lib import *
q = pd.read_pickle("val_state3.pkl")
_, Y = load_split("validation")
c = q["cutoff"].to_numpy(); n = len(q)
truth = q[Q].to_numpy(); mt = truth[:, 1:] / truth[:, 1:].sum(1, keepdims=True)
q["ls1"] = np.log(mt[:, 0] / (1 - mt[:, 0]))
q["ls1r"] = q["ls1"] - q.groupby("machine_type")["ls1"].transform("mean")
print("sd of logit P(1x|win) residual:", q["ls1r"].std().round(3))
encs = {"win": lambda y: (y > 0), "one": lambda y: (y == 1), "big2": lambda y: (y >= 2), "zero_or_one": lambda y: (y <= 1), "ge5": lambda y: (y >= 3)}
res = []
for en, f in encs.items():
    V = {}
    for k in range(1, 9):
        y = Y[np.arange(n), c - k]
        V[k] = np.where(y < 0, np.nan, 2 * f(y).astype(float) - 1)
        r = q["ls1r"].to_numpy(); ok = ~np.isnan(V[k])
        res.append((abs(np.corrcoef(r[ok], V[k][ok])[0, 1]), en, f"lag{k}"))
    for i, j in combinations(range(1, 9), 2):
        p = V[i] * V[j]; ok = ~np.isnan(p)
        if p[ok].std() > 0:
            res.append((abs(np.corrcoef(r[ok], p[ok])[0, 1]), en, f"pair{i}-{j}"))
res.sort(reverse=True)
print("top |corr| with logit P(1x|win):", [(e, f, round(v, 3)) for v, e, f in res[:12]])
# persistence: history-based share1 by window, correlation with truth
for last in (16, 32, 64, 128, 256):
    est = []
    for m in range(n):
        y = Y[m, max(0, c[m] - last):c[m]]; w = y[y > 0]
        est.append(((w == 1).sum() + 1) / (len(w) + 2))
    est = np.log(np.array(est) / (1 - np.array(est)))
    est = est - pd.Series(est).groupby(q["machine_type"].to_numpy()).transform("mean").to_numpy()
    print(f"history last {last:3d}: corr with truth residual {np.corrcoef(est, q['ls1r'])[0,1]:.3f}")
