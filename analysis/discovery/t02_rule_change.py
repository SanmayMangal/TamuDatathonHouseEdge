import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
import numpy as np, pandas as pd
from phase1.common import load, history_grid
ctx = load()
Y, meta, _ = history_grid(ctx)
W = np.where(Y >= 0, (Y > 0).astype(float), np.nan)
PAIRS = [(1, 3), (2, 5), (4, 7)]
long = ~meta["short_history"].to_numpy()
rows = []
for m in np.flatnonzero(long):
    w = W[m]; c = int(meta.loc[m, "cutoff"])
    t = np.arange(8, c)
    best = None
    for i, j in PAIRS:
        same = (w[t - i] == w[t - j]).astype(float); y = w[t]
        d = y[same == 1].mean() - y[same == 0].mean()
        if best is None or abs(d) > abs(best[1]): best = ((i, j), d, same, y)
    (i, j), d, same, y = best
    # rolling 48-tick window effect sign over time
    win = 48; eff = []
    for s in range(0, len(t) - win + 1, 8):
        sm, yy = same[s:s+win], y[s:s+win]
        eff.append(yy[sm == 1].mean() - yy[sm == 0].mean() if 0 < sm.sum() < win else np.nan)
    eff = np.array(eff)
    half = len(t) // 2
    d1 = y[:half][same[:half] == 1].mean() - y[:half][same[:half] == 0].mean()
    d2 = y[half:][same[half:] == 1].mean() - y[half:][same[half:] == 0].mean()
    late = y[-48:][same[-48:] == 1].mean() - y[-48:][same[-48:] == 0].mean()
    flips = np.flatnonzero(np.diff(np.sign(eff[~np.isnan(eff)])) != 0)
    rows.append({"m": meta.loc[m, "machine_id"], "ep": meta.loc[m, "episode_id"], "cond": meta.loc[m, "condition"],
                 "pair": f"{i}-{j}", "d_full": d, "d_first_half": d1, "d_second_half": d2, "d_last48": late,
                 "sign_flip_halves": np.sign(d1) != np.sign(d2) and abs(d1) > .2 and abs(d2) > .2,
                 "first_flip_tick": 8 + 8*flips[0] + win//2 if len(flips) else np.nan,
                 "abs_d_first": abs(d1), "abs_d_last48": abs(late)})
r = pd.DataFrame(rows)
print(r.groupby("cond")[["abs_d_first", "abs_d_last48"]].median().round(3).to_string())
print(r.groupby("cond")["sign_flip_halves"].mean().round(3).to_string())
# same pair across machines in an episode? and flip timing alignment in shared_change
print("distinct dominant pairs per episode (median):", r.groupby(["cond","ep"])["pair"].nunique().groupby("cond").median().to_dict())
f = r[r["sign_flip_halves"]].groupby(["cond","ep"])["first_flip_tick"].agg(["size","std"])
print(f[f["size"]>=2].groupby("cond")[["size","std"]].median().round(1).to_string())
# share of machines in an episode that flip, shared_change vs drift
print(r.groupby(["cond","ep"])["sign_flip_halves"].mean().groupby("cond").describe()[["mean","50%","75%"]].round(2).to_string())
# drift: gradual weakening? correlation of |eff| decline
