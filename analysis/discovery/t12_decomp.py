import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from discovery_lib import *
q = pd.read_pickle("val_state3.pkl")
t = q[Q].to_numpy()
L = np.log(t); L = L - L.mean(1, keepdims=True)
for typ in ["steady", "balanced", "jackpot"]:
    g = (q["machine_type"] == typ).to_numpy()
    X = L[g] - L[g].mean(0)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    print(f"{typ}: var share {np.round(s**2/(s**2).sum(),3)}")
    for k in range(3): print(f"     comp{k} loading on (0,1,2,5,20): {np.round(vt[k],3)}")
    # within-group ratios
    r01 = t[g, 1] / (t[g, 0] + t[g, 1]); r_hi = t[g, 2:] / t[g, 2:].sum(1, keepdims=True)
    print(f"   P1/(P0+P1): mean {r01.mean():.3f} sd {r01.std():.3f};  mix of 2/5/20 sd {np.round(r_hi.std(0),4)}")
    print(f"   corr(P1/(P0+P1), qwin) {np.corrcoef(r01, q.loc[g,'qwin'])[0,1]:.3f}")
