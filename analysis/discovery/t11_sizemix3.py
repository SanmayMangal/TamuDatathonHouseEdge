import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from discovery_lib import *
q = pd.read_pickle("val_state2.pkl")
truth = q[Q].to_numpy(); mt = truth[:, 1:] / truth[:, 1:].sum(1, keepdims=True)
L = np.log(mt)
L = L - L.mean(1, keepdims=True)  # centered log-ratios (clr)
for typ in ["steady", "balanced", "jackpot"]:
    g = (q["machine_type"] == typ).to_numpy()
    X = L[g] - L[g].mean(0)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    print(f"{typ}: variance share per component {np.round(s**2 / (s**2).sum(), 3)}  first loading {np.round(vt[0], 3)}")
    # implied: clr shift proportional to log(payout)? compare with log payout centered
    lp = np.log([1, 2, 5, 20]); lp = lp - lp.mean()
    print(f"   corr(loading, log payout) = {np.corrcoef(vt[0], lp)[0,1]:.3f}; loading/logpayout ratio {np.round(vt[0] / lp, 3)}")
# tilt score per machine: projection on centered log payout
lp = np.log([1, 2, 5, 20]); lp = lp - lp.mean()
q["tilt"] = (L @ lp) / (lp @ lp)
q["tilt_res"] = q["tilt"] - q.groupby("machine_type")["tilt"].transform("mean")
print("\ntilt sd by type:", q.groupby("machine_type")["tilt_res"].std().round(3).to_dict())
# shared within episode?
ep = q.groupby("episode_id")["tilt_res"]
within = q["tilt_res"] - ep.transform("mean")
print("share of tilt variance between episodes:", round(1 - within.var() / q["tilt_res"].var(), 3), " (if random: ~1/8 = 0.125)")
print("by condition sd:", q.groupby("condition")["tilt_res"].std().round(3).to_dict())
print("corr tilt with signal:", round(np.corrcoef(q["tilt_res"], q["signal"])[0,1], 3), " with qwin:", round(np.corrcoef(q["tilt_res"], q["qwin"])[0,1], 3))
q.to_pickle("val_state3.pkl")
