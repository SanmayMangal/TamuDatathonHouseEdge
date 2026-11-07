import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from discovery_lib import *
tr = pd.read_pickle("train_map.pkl")
q, Y = load_split("train"); W = win_grid(Y)
rng = np.random.default_rng(0)
zs, zn, zm = [], [], []
for m in range(len(q)):
    if q.loc[m, "short"]: continue
    c = int(q.loc[m, "cutoff"]); i, j = tr.loc[m, "pair"]; d = tr.loc[m, "dir"] == 1
    tau = c - tr.loc[m, "tau_before_cutoff"] if not np.isnan(tr.loc[m, "tau_before_cutoff"]) else -1
    t = np.arange(8, c); y = Y[m, t]; a, b = W[m, t - i], W[m, t - j]
    match = (a == b); fav = np.where(t < tau, match != d, match == d)
    win = y > 0; one = (y == 1)[win]
    for flag, store in ((fav[win], zs), (match[win], zm)):
        if 3 < flag.sum() < len(flag) - 3:
            p = one.mean(); dd = one[flag].mean() - one[~flag].mean()
            store.append(dd / np.sqrt(p * (1 - p) * (1 / flag.sum() + 1 / (~flag).sum())))
    perm = rng.permutation(fav[win])
    if 3 < perm.sum() < len(perm) - 3:
        p = one.mean(); dd = one[perm].mean() - one[~perm].mean(); zn.append(dd / np.sqrt(p * (1 - p) * (1 / perm.sum() + 1 / (~perm).sum())))
for name, z in (("fav state", zs), ("match", zm), ("shuffled", zn)):
    z = np.array(z); print(f"1x share vs {name:9s}: mean z {z.mean():+.3f}, sd z {z.std():.3f} (null sd = 1), share |z|>3 {np.mean(np.abs(z) > 3):.3f}")
