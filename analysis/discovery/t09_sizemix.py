import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from discovery_lib import *
q = pd.read_pickle("val_state2.pkl")
_, Y = load_split("validation"); W = win_grid(Y)
c = q["cutoff"].to_numpy(); n = len(q)
truth = q[Q].to_numpy(); mt = truth[:, 1:] / truth[:, 1:].sum(1, keepdims=True)
def hist_mix(m, last=None, hl=None):
    y = Y[m, :c[m]]; t = np.flatnonzero(y > 0)
    if last: t = t[t >= c[m] - last]
    w = np.ones(len(t)) if hl is None else 2.0 ** (-(c[m] - 1 - t) / hl)
    v = np.zeros(4); np.add.at(v, y[t] - 1, w); return v, w.sum()
for typ in ["steady", "balanced", "jackpot"]:
    ms = np.flatnonzero((q["machine_type"] == typ).to_numpy())
    tm = mt[ms].mean(0)
    for label, kw in [("all", {}), ("last64", {"last": 64}), ("hl32", {"hl": 32}), ("hl16", {"hl": 16})]:
        H = []; N = []
        for m in ms:
            v, s = hist_mix(m, **kw); H.append(v); N.append(s)
        H = np.array(H); N = np.array(N)
        for strength in (5, 20):
            est = (H + strength * tm) / (N[:, None] + strength)
            err = ((est - mt[ms]) ** 2).sum(1).mean()
            print(f"{typ:9s} {label:7s} s={strength:2d} mix err {err:.4f}   (type-mean err {((tm - mt[ms])**2).sum(1).mean():.4f})")
    # correlation of truth m1 with history m1 (all)
    H = np.array([hist_mix(m)[0] for m in ms]); hm = H / np.maximum(H.sum(1, keepdims=True), 1)
    print(f"   corr(true share of smallest prize, history share) = {np.corrcoef(mt[ms,0], hm[:,0])[0,1]:.3f}")
