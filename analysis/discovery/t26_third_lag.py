import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from scipy.special import betaln
from discovery_lib import *
tr = pd.read_pickle("train_map.pkl")
q, Y = load_split("train"); W = win_grid(Y)
rng = np.random.default_rng(1)
z3, zn, enc_gain = [], [], []
def ll_cells(y, groups):
    s = 0.0
    for g in np.unique(groups):
        k = y[groups == g].sum(); n = (groups == g).sum(); s += betaln(1 + k, 1 + n - k)
    return s
for m in range(len(q)):
    if q.loc[m, "short"]: continue
    c = int(q.loc[m, "cutoff"]); i, j = tr.loc[m, "pair"]; d = tr.loc[m, "dir"] == 1
    tau = c - tr.loc[m, "tau_before_cutoff"] if not np.isnan(tr.loc[m, "tau_before_cutoff"]) else -1
    t = np.arange(8, c); y = W[m, t]; a, b = W[m, t - i], W[m, t - j]
    fav = np.where(t < tau, (a == b) != d, (a == b) == d)
    base = ll_cells(y, fav.astype(int))
    for k in range(1, 9):
        if k in (i, j): continue
        x = W[m, t - k]
        # third lag as a modifier within each state: 4 cells
        z3.append(ll_cells(y, fav * 2 + x) - base)
        zn.append(ll_cells(y, fav * 2 + rng.permutation(x)) - base)
    # encoding of the match: exact payout equality vs win equality
    ya, yb = Y[m, t - i], Y[m, t - j]
    fav_exact = np.where(t < tau, (ya == yb) != d, (ya == yb) == d)
    enc_gain.append(ll_cells(y, fav_exact.astype(int)) - base)
z3, zn = np.array(z3), np.array(zn)
print(f"third lag within state: mean log-evidence gain {z3.mean():+.3f} vs shuffled {zn.mean():+.3f}; share >+3: {np.mean(z3 > 3):.4f} vs {np.mean(zn > 3):.4f}")
print(f"match on exact payout instead of win/loss: mean log-evidence change {np.mean(enc_gain):+.2f}, share better: {np.mean(np.array(enc_gain) > 0):.3f}")
