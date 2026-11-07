import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from discovery_lib import *
q = pd.read_pickle("val_state.pkl")
_, Y = load_split("validation"); W = win_grid(Y)
c = q["cutoff"].to_numpy()
for k in range(1, 9):
    q[f"w{k}"] = W[np.arange(len(q)), c - k]
    q[f"y{k}"] = Y[np.arange(len(q)), c - k]
q["lq"] = np.log(q["qwin"] / (1 - q["qwin"]))
# Only long, confident machines
L = q[(~q["short"]) & (q["rz"].abs() > 4)].copy()
print("n confident long machines:", len(L))
for typ in ["steady", "balanced", "jackpot"]:
    for fav in [True, False]:
        g = L[(L["machine_type"] == typ) & (L["fav"] == fav)]
        print(f"\n{typ} fav={fav} n={len(g)} logit-qwin mean {g['lq'].mean():.2f} sd {g['lq'].std():.2f}")
        print("  by signal:", g.groupby("signal")["lq"].mean().round(2).to_dict())
        eff = {k: round(g[g[f"w{k}"] == 1]["lq"].mean() - g[g[f"w{k}"] == 0]["lq"].mean(), 2) for k in range(1, 9)}
        print("  lag win effect on logit:", eff)
        print("  by condition:", g.groupby("condition")["lq"].mean().round(2).to_dict())
