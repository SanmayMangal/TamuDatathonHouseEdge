import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from discovery_lib import *
R = np.load("train_R.npy")
q, _ = load_split("train")
eps = q.groupby("episode_id").indices
long_eps = [e for e, ix in eps.items() if not q.loc[ix[0], "short"]]
# episode-level per-tick mean working residual (clip extreme working residuals)
Rc = np.clip(R, -10, 10)
S = np.array([np.nanmean(Rc[eps[e]], 0) for e in long_eps])  # (E, T)
S = S - np.nanmean(S, 1, keepdims=True)
ac = []
for k in range(1, 49):
    a, b = S[:, :-k].ravel(), S[:, k:].ravel(); ok = ~np.isnan(a) & ~np.isnan(b)
    ac.append(np.corrcoef(a[ok], b[ok])[0, 1])
print("tick-level autocorr of episode-mean residual:", " ".join(f"{k}:{v:.3f}" for k, v in zip(range(1, 49), ac) if k in (1,2,3,4,6,8,12,16,24,32,40,48)))
# cross-machine: corr between machine A residual at t and machine B residual at t (same tick) vs t-1
same, lag1 = [], []
for e in long_eps:
    X = Rc[eps[e]]
    for a in range(8):
        for b in range(8):
            if a == b: continue
            x, y = X[a], X[b]; ok = ~np.isnan(x) & ~np.isnan(y)
            same.append(np.corrcoef(x[ok], y[ok])[0, 1])
            ok2 = ~np.isnan(x[1:]) & ~np.isnan(y[:-1]); lag1.append(np.corrcoef(x[1:][ok2], y[:-1][ok2])[0, 1])
print(f"cross-machine residual corr, same tick: {np.nanmean(same):.4f}; one tick apart: {np.nanmean(lag1):.4f}")
