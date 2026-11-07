import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from discovery_lib import *
q, Y = load_split("validation")
W = win_grid(Y)
qq = q[Q].to_numpy()
q["qwin"] = 1 - qq[:, 0]
# size mix given win
mix = qq[:, 1:] / qq[:, 1:].sum(1, keepdims=True)
for k, c in enumerate(["m1", "m2", "m5", "m20"]): q[c] = mix[:, k]
print("Size mix given win: by type, mean and sd")
print(q.groupby("machine_type")[["m1", "m2", "m5", "m20"]].agg(["mean", "std"]).round(3).to_string())
print("\nby type x signal mean:")
print(q.groupby(["machine_type", "signal"])[["m1", "m2", "m5", "m20"]].mean().round(3).to_string())
print("\nby type x condition sd of m1/m2/m5:")
print(q.groupby(["machine_type", "condition"])[["m1", "m2", "m5", "m20"]].std().round(3).to_string())
print("\nqwin distribution:")
print(np.round(np.quantile(q["qwin"], [0, .05, .1, .25, .5, .75, .9, .95, 1]), 3))
print(q.groupby("machine_type")["qwin"].describe().round(3).to_string())
