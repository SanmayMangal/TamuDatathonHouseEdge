import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
"""Is there an episode-level offset in win log-odds, and is it constant or time-varying?
Uses train histories + the MAP rule assignment (train_map.pkl has per-machine pair/dir/tau)."""
from discovery_lib import *
tr = pd.read_pickle("train_map.pkl")
q, Y = load_split("train"); W = win_grid(Y)
G = tr.groupby(["type", "signal"])[["kf", "nf", "ku", "nu"]].sum()
lg = lambda p: np.log(p / (1 - p))
BASE = {k: (lg(r.kf / r.nf), lg(r.ku / r.nu)) for k, r in G.iterrows()}
# per machine per tick: expected logit under MAP rule; residual = observed win - predicted prob (in prob space), plus fav flag
R = np.full(Y.shape, np.nan); FAV = np.full(Y.shape, np.nan)
for m in range(len(q)):
    c = int(q.loc[m, "cutoff"]); i, j = tr.loc[m, "pair"]; d = tr.loc[m, "dir"] == 1
    tau = c - tr.loc[m, "tau_before_cutoff"] if not np.isnan(tr.loc[m, "tau_before_cutoff"]) else -1
    lf, lu = BASE[(tr.loc[m, "type"], tr.loc[m, "signal"])]
    for t in range(8, c):
        a, b, y = W[m, t - i], W[m, t - j], W[m, t]
        if np.isnan(a) or np.isnan(b) or np.isnan(y): continue
        fav = (a == b) == d
        if t < tau: fav = not fav
        p = 1 / (1 + np.exp(-(lf if fav else lu)))
        R[m, t] = (y - p) / (p * (1 - p))   # working residual on logit scale (score / info)
        FAV[m, t] = fav
eps = q.groupby("episode_id").indices
long_eps = [e for e, ix in eps.items() if not q.loc[ix[0], "short"]]
# 1) between-episode variance of mean residual vs within
ep_mean = np.array([np.nanmean(R[eps[e]]) for e in long_eps])
m_mean = np.array([np.nanmean(R[m]) for e in long_eps for m in eps[e]])
m_ep = np.repeat(ep_mean, 8)
print("machine mean working residual sd:", np.nanstd(m_mean).round(3), " episode-mean sd:", np.nanstd(ep_mean).round(3),
      " corr(machine, rest-of-episode):", np.corrcoef(m_mean, (m_ep * 8 - m_mean) / 7)[0, 1].round(3))
# 2) time variation: episode-mean residual in 32-tick windows; correlation between windows by distance
wins = 32
E = np.array([[np.nanmean(R[eps[e], s:s + wins]) for s in range(8, 200, wins)] for e in long_eps])
print("episode offset by window (rows=episode); correlation between windows k apart:")
for k in range(1, 6):
    a, b = E[:, :-k].ravel(), E[:, k:].ravel(); ok = ~np.isnan(a) & ~np.isnan(b)
    print(f"  {k * wins:3d} ticks apart: corr {np.corrcoef(a[ok], b[ok])[0,1]:.3f}")
# 3) same for machine-specific part (machine residual minus episode residual)
Mres = []
for e in long_eps:
    ix = eps[e]
    for m in ix:
        Mres.append([np.nanmean(R[m, s:s + wins]) - np.nanmean(R[ix, s:s + wins]) for s in range(8, 200, wins)])
Mres = np.array(Mres)
for k in (1, 3, 5):
    a, b = Mres[:, :-k].ravel(), Mres[:, k:].ravel(); ok = ~np.isnan(a) & ~np.isnan(b)
    print(f"  machine-specific part, {k * wins} ticks apart: corr {np.corrcoef(a[ok], b[ok])[0,1]:.3f}")
# 4) does the episode offset differ by condition?
cond = np.array([q.loc[eps[e][0], "condition"] for e in long_eps])
print("episode offset sd by condition:", pd.Series(ep_mean).groupby(cond).std().round(3).to_dict())
np.save("train_R.npy", R)
