import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from itertools import combinations
from discovery_lib import *
q, Y = load_split("validation")
W = win_grid(Y)
qq = q[Q].to_numpy(); q["qwin"] = 1 - qq[:, 0]
PAIRS = list(combinations(range(1, 9), 2))
def pair_fit(w, c, recent=None):
    t = np.arange(8, c)
    if recent: t = t[t >= c - recent]
    best = None
    for i, j in PAIRS:
        a, b, y = w[t - i], w[t - j], w[t]
        ok = ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(y)
        if ok.sum() < 10: continue
        same = (a == b)[ok]; yy = y[ok]
        if same.all() or (~same).all(): continue
        d = yy[same].mean() - yy[~same].mean()
        se = np.sqrt(0.25 * (1 / same.sum() + 1 / (~same).sum()))
        z = d / se
        if best is None or abs(z) > abs(best[2]): best = ((i, j), d, z)
    return best
rows = []
for m in range(len(q)):
    c = int(q.loc[m, "cutoff"]); w = W[m]
    full = pair_fit(w, c); rec = pair_fit(w, c, recent=48)
    if full is None:
        rows.append({"pair": None, "d": np.nan, "z": np.nan, "match": np.nan, "rpair": None, "rd": np.nan, "rz": np.nan, "rmatch": np.nan}); continue
    (i, j), d, z = full
    match_c = w[c - i] == w[c - j] if not (np.isnan(w[c - i]) or np.isnan(w[c - j])) else np.nan
    if rec is None: rec = full
    (ri, rj), rd, rz = rec
    rmatch = w[c - ri] == w[c - rj] if not (np.isnan(w[c - ri]) or np.isnan(w[c - rj])) else np.nan
    rows.append({"pair": f"{i}-{j}", "d": d, "z": z, "match": match_c,
                 "rpair": f"{ri}-{rj}", "rd": rd, "rz": rz, "rmatch": rmatch})
s = pd.DataFrame(rows); q = pd.concat([q, s], axis=1)
# predicted favorable state: match & d>0 or mismatch & d<0 (using recent rule)
q["fav"] = np.where(q["rd"] > 0, q["rmatch"] == 1, q["rmatch"] == 0)
print("qwin by favorable state (recent-48 rule), per condition x type (median, count):")
print(q.groupby(["condition", "machine_type", "fav"])["qwin"].agg(["median", "mean", "size"]).round(3).to_string())
print("\npair agreement full vs recent:", (q["pair"] == q["rpair"]).mean().round(3))
print("recent pair counts:", q["rpair"].value_counts().head(6).to_dict())
q.to_pickle("val_state.pkl")
