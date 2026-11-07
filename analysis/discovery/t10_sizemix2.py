import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from itertools import combinations
from discovery_lib import *
q = pd.read_pickle("val_state2.pkl")
_, Y = load_split("validation"); W = win_grid(Y)
c = q["cutoff"].to_numpy(); n = len(q)
truth = q[Q].to_numpy(); mt = truth[:, 1:] / truth[:, 1:].sum(1, keepdims=True)
# log-ratio of big vs small prizes, per type
q["lr"] = np.log(mt[:, 1] / mt[:, 0])  # log(P2/P1)
q["lr5"] = np.log(mt[:, 2] / mt[:, 0])
q["fav1"] = q["fav"].astype(float)
for k in range(1, 9):
    q[f"x{k}"] = 2 * W[np.arange(n), c - k] - 1
    q[f"big{k}"] = (Y[np.arange(n), c - k] >= 2).astype(float)
for typ in ["steady", "balanced", "jackpot"]:
    g = q[q["machine_type"] == typ]
    print(f"\n{typ}: sd log(P2/P1) {g['lr'].std():.3f}")
    print("  corr with fav:", round(np.corrcoef(g['lr'], g['fav1'])[0,1], 3), " qwin:", round(np.corrcoef(g['lr'], g['qwin'])[0,1], 3), " signal:", round(np.corrcoef(g['lr'], g['signal'])[0,1], 3))
    cx = {k: round(np.corrcoef(g['lr'], g[f'x{k}'])[0,1], 2) for k in range(1, 9)}
    cb = {k: round(np.nanmean(g[g[f'big{k}']==1]['lr']) - np.nanmean(g[g[f'big{k}']==0]['lr']), 2) for k in range(1, 9)}
    print("  corr with win at lag:", cx)
    print("  diff when lag payout >=2:", cb)
    # pair products for size
    best = sorted(((abs(np.corrcoef(g['lr'], g[f'x{i}'] * g[f'x{j}'])[0,1]), (i, j)) for i, j in combinations(range(1, 9), 2) if (g[f'x{i}'] * g[f'x{j}']).std() > 0), reverse=True)[:4]
    print("  top |corr| with lag-pair products:", [(p, round(v, 2)) for v, p in best])
