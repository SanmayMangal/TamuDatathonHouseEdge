import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from discovery_lib import *
q = pd.read_pickle("val_state.pkl")
_, Y = load_split("validation"); W = win_grid(Y)
c = q["cutoff"].to_numpy(); n = len(q)
X = {}
for k in range(1, 13):
    X[k] = 2 * W[np.arange(n), c - k] - 1
q["lq"] = np.log(q["qwin"] / (1 - q["qwin"]))
L = q[(~q["short"]) & q["rpair"].notna()].copy()
idx = L.index.to_numpy()
rule = np.where(L["rd"] > 0, 1, -1) * np.array([X[int(p.split('-')[0])][m] * X[int(p.split('-')[1])][m] for p, m in zip(L["rpair"], idx)])
def fit(cols, names):
    A = np.column_stack(cols)
    ok = ~np.isnan(A).any(1)
    b, *_ = np.linalg.lstsq(A[ok], L["lq"].to_numpy()[ok], rcond=None)
    r = L["lq"].to_numpy()[ok] - A[ok] @ b
    return dict(zip(names, b.round(3))), r.std().round(3), (1 - r.var() / L["lq"].to_numpy()[ok].var()).round(3)
T = pd.get_dummies(L["machine_type"]).to_numpy().astype(float)
base = [T[:, 0], T[:, 1], T[:, 2]]; bn = ["bal", "jack", "steady"]
print("type only:", fit(base, bn)[1:])
print("type+signal:", fit(base + [L["signal"].to_numpy()], bn + ["signal"])[1:])
print("type+signal+rule:", fit(base + [L["signal"].to_numpy(), rule], bn + ["signal", "rule"]))
rt = [rule * T[:, i] for i in range(3)]
cols = base + [L["signal"].to_numpy()] + rt + [X[k][idx] for k in range(1, 9)]
print("type+signal+rule*type+lags1-8:", fit(cols, bn + ["signal", "rule_bal", "rule_jack", "rule_st"] + [f"x{k}" for k in range(1, 9)]))
cols2 = cols + [X[k][idx] for k in range(9, 13)]
print("+lags9-12:", fit(cols2, [str(i) for i in range(len(cols2))])[1:])
# count of wins in a longer window (hidden momentum?)
for win in (8, 16, 32, 64):
    rate = np.array([np.nanmean(W[m, c[m] - win:c[m]]) for m in idx])
    print(f"+ win rate last {win}:", fit(cols + [rate], [str(i) for i in range(len(cols) + 1)])[1:], "coef", fit(cols + [rate], [str(i) for i in range(len(cols) + 1)])[0][str(len(cols))])
