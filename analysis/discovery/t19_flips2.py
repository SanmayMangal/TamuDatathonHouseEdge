import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from discovery_lib import *
r = pd.read_pickle("flips.pkl")
r["B_vs_A"] = r["llB"] - r["llA"]
ch = r[(r["cond"] != "stable") & (r["B_vs_A"] > 8)].copy()
ch["ticks_before_cutoff"] = ch["cutoff"] - ch["tauB"]
print("change position (ticks before cutoff) quantiles by cond:")
print(ch.groupby("cond")["ticks_before_cutoff"].quantile([.05, .25, .5, .75, .95]).unstack().round(0).to_string())
print("\nchange tick quantiles:", ch.groupby("cond")["tauB"].quantile([.05, .5, .95]).unstack().to_string())
g = ch.groupby(["cond", "ep"])["tauB"].agg(["size", "std", "min", "max", lambda x: np.median(np.abs(x - np.median(x)))])
g.columns = ["n", "sd", "min", "max", "mad"]
print("\nwithin-episode alignment of change ticks (episodes with >=3 changed machines):")
print(g[g["n"] >= 3].groupby("cond")[["n", "sd", "mad"]].median().round(1).to_string())
print("share of episode's long machines that change:", (r[r["cond"] != "stable"].assign(c=r["B_vs_A"] > 8).groupby(["cond", "ep"])["c"].mean().groupby("cond").median().round(2)).to_dict())
# do the shared change episodes have ALL machines flip at one tick? show a few
for cond in ("shared_change", "drift"):
    print(f"\nexample {cond} episodes: change ticks sorted")
    for ep, gg in list(ch[ch["cond"] == cond].groupby("ep"))[:5]:
        print("  ", ep, sorted(gg["tauB"].astype(int).tolist()), "cutoff", gg["cutoff"].iloc[0])
