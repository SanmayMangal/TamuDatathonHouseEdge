import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from itertools import combinations
from discovery_lib import *
q = pd.read_pickle("val_state.pkl")
_, Y = load_split("validation"); W = win_grid(Y)
c = q["cutoff"].to_numpy(); n = len(q)
q["lq"] = np.log(q["qwin"] / (1 - q["qwin"]))
base = {"balanced": -0.64, "jackpot": -2.40, "steady": 0.74}
q["resid0"] = q["lq"] - q["machine_type"].map(base) - 0.47 * q["signal"]
q["true_fav"] = q["resid0"] > 0
q["agree"] = q["fav"] == q["true_fav"]
print("agreement of recent-48 rule with truth:")
print(q.groupby(["condition", "short"])["agree"].mean().round(3).to_string())
# Among disagreements: which pair-state (any of 28 pairs, either sign) matches truth? Count pairs consistent with truth
PAIRS = list(combinations(range(1, 9), 2))
cons = {p: [] for p in PAIRS}
for m in range(n):
    for i, j in PAIRS:
        a, b = W[m, c[m]-i], W[m, c[m]-j]
        cons[(i, j)].append(np.nan if np.isnan(a) or np.isnan(b) else float(a == b))
for p in PAIRS: q[f"s{p[0]}{p[1]}"] = cons[p]
# For each machine, under the full-history best pair, compare flip hypothesis: if rule flipped direction
dis = q[~q["agree"]]
print("\nn disagreements:", len(dis), "of", n)
print("disagreements by condition:", dis["condition"].value_counts().to_dict())
print("median |rz| agree vs disagree:", q.groupby("agree")["rz"].apply(lambda z: z.abs().median()).round(2).to_dict())
# how far is truth from boundary in disagreements
print("median |resid0| agree/disagree:", q.groupby("agree")["resid0"].apply(lambda r: r.abs().median()).round(2).to_dict())
q.to_pickle("val_state2.pkl")
