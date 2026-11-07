import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
exec(open(os.path.join(HERE, "t18_flips.py")).read().split("rows = []")[0])  # reuse fit_machine
from itertools import combinations
ALL = list(combinations(range(1, 9), 2))
out = []
q, Y = load_split("test"); W = win_grid(Y)
for m in range(len(q)):
    if q.loc[m, "short"]: continue
    r = fit_machine(W[m], q.loc[m, "cutoff"]); r.update(ep=q.loc[m, "episode_id"], cutoff=q.loc[m, "cutoff"]); out.append(r)
r = pd.DataFrame(out); r["B_vs_A"] = r["llB"] - r["llA"]
print("test long machines:", len(r), " short machines:", int(q["short"].sum()))
print("pair distribution (canonical 3):", r["pair"].value_counts().to_dict())
r["changed"] = r["B_vs_A"] > 8
e = r.groupby("ep").agg(frac_changed=("changed", "mean"), sd_tau=("tauB", lambda x: np.nan), n=("changed", "size"))
e["sd_tau"] = r[r["changed"]].groupby("ep")["tauB"].std()
def label(row):
    if row["frac_changed"] < 0.5: return "stable-like"
    return "shared-like" if row["sd_tau"] < 4 else "drift-like"
e["guess"] = e.apply(label, axis=1)
print("episode types inferred from histories:", e["guess"].value_counts().to_dict(), "(expected 48 each of stable/drift/shared + 48 short)")
print("median frac changed by guess:", e.groupby("guess")["frac_changed"].median().to_dict())
# Same procedure on train/val with known conditions -> accuracy
