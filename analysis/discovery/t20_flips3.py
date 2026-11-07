import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from itertools import combinations
from discovery_lib import *
r = pd.read_pickle("flips.pkl"); r["B_vs_A"] = r["llB"] - r["llA"]
ALL = list(combinations(range(1, 9), 2))
grids = {}
for split in ("train", "validation"):
    q, Y = load_split(split); grids[split] = (q.set_index("machine_id"), win_grid(Y), q)
prof = {"drift": [], "shared_change": []}; same_pair = []; levels = []
for _, row in r[(r["cond"] != "stable") & (r["B_vs_A"] > 8)].iterrows():
    qi, W, q = grids[row["split"]]; m = q.index[q["machine_id"] == row["mid"]][0]; w = W[m]; c = row["cutoff"]; tau = int(row["tauB"])
    def best_pair(lo, hi):
        t = np.arange(max(lo, 8), hi); b = None
        for i, j in ALL:
            x = (w[t - i] == w[t - j]); y = w[t]
            if x.all() or (~x).all(): continue
            d = y[x].mean() - y[~x].mean()
            if b is None or abs(d) > abs(b[1]): b = ((i, j), d, y[x].mean(), y[~x].mean())
        return b
    pre, post = best_pair(0, tau), best_pair(tau, c)
    if pre is None or post is None: continue
    same_pair.append({"cond": row["cond"], "same": pre[0] == post[0], "sign_flip": np.sign(pre[1]) != np.sign(post[1]),
                      "pre_d": pre[1], "post_d": post[1]})
    i, j = map(int, row["pair"].split("-"))
    t = np.arange(8, c); x = 2.0 * (w[t - i] == w[t - j]) - 1; y = w[t]
    pre_sign = np.sign(pre[1]) if pre[0] == (i, j) else 1
    for k, tt in enumerate(t):
        rel = tt - tau
        if -48 <= rel < 48: prof[row["cond"]].append((rel // 4, (y[k] - y.mean()) * x[k] * pre_sign))
    # levels: P(win|favorable) and P(win|unfav) pre and post
    xs = (w[t - i] == w[t - j]); pre_m = t < tau
    fav_pre = xs == (pre[1] > 0); fav_post = xs == (post[1] > 0)
    levels.append({"cond": row["cond"], "pf_pre": y[pre_m & fav_pre].mean(), "pu_pre": y[pre_m & ~fav_pre].mean(),
                   "pf_post": y[~pre_m & fav_post].mean(), "pu_post": y[~pre_m & ~fav_post].mean()})
sp = pd.DataFrame(same_pair)
print("same best pair before/after:", sp.groupby("cond")["same"].mean().round(3).to_dict(), " sign flips:", sp.groupby("cond")["sign_flip"].mean().round(3).to_dict())
print("median |effect| pre/post:", sp.groupby("cond")[["pre_d", "post_d"]].apply(lambda d: d.abs().median()).round(3).to_string())
print(pd.DataFrame(levels).groupby("cond").median().round(3).to_string())
for cond, v in prof.items():
    d = pd.DataFrame(v, columns=["bin", "e"]).groupby("bin")["e"].mean()
    print(f"\n{cond}: aligned rule effect in 4-tick bins around the change (pre-change sign = +)")
    print("  ", " ".join(f"{int(b*4):+d}:{e:+.2f}" for b, e in d.items()))
