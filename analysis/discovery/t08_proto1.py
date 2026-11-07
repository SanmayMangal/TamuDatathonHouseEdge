import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from discovery_lib import *
from evaluate import condition_errors
q = pd.read_pickle("val_state2.pkl")
_, Y = load_split("validation"); W = win_grid(Y)
c = q["cutoff"].to_numpy(); n = len(q)
truth = q[Q].to_numpy(); ref = q[["ref_" + x for x in Q]].to_numpy()
def score(pred, label=""):
    e = condition_errors(q, ((pred - truth) ** 2).sum(1)); b = condition_errors(q, ((ref - truth) ** 2).sum(1))
    r = e / b; R = r.mean()
    print(f"{label:45s} pts {60/(1+R):6.2f} R {R:.3f}  " + " ".join(f"{k[:6]} {v:.3f}" for k, v in r.items()))
mix_type = q.groupby("machine_type")[Q[1:]].apply(lambda d: (d.to_numpy() / d.to_numpy().sum(1, keepdims=True)).mean(0))
def mixfor(t): return np.vstack([mix_type[x] for x in t])
base = {"balanced": -0.64, "jackpot": -2.40, "steady": 0.74}
sig = lambda z: 1 / (1 + np.exp(-z))
fav = np.where(q["rpair"].isna(), 0, np.where(q["fav"], 1, -1))
# 1. global formula, type mix
pw = sig(q["machine_type"].map(base).to_numpy() + 0.47 * q["signal"].to_numpy() + 2.17 * fav)
score(np.column_stack([1 - pw, pw[:, None] * mixfor(q["machine_type"])]), "formula (fitted on val!) + type mix")
# 2. oracle win chance, type mix -> how much is size mix
qw = q["qwin"].to_numpy()
score(np.column_stack([1 - qw, qw[:, None] * mixfor(q["machine_type"])]), "ORACLE win chance + type mix")
# 3. oracle mix, formula win
mix_true = truth[:, 1:] / truth[:, 1:].sum(1, keepdims=True)
score(np.column_stack([1 - pw, pw[:, None] * mix_true]), "formula win + ORACLE mix")
# 4. per-machine history estimates: P(win | fav state), P(win | unfav), recent-weighted
rows = []
for m in range(n):
    p = q.loc[m, "rpair"]
    if not isinstance(p, str):
        rows.append((np.nanmean(W[m, :c[m]]),) * 2); continue
    i, j = map(int, p.split("-")); d = q.loc[m, "rd"]
    t = np.arange(8, c[m]); a, b, y = W[m, t-i], W[m, t-j], W[m, t]
    ok = ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(y)
    f = ((a == b) == (d > 0))[ok]; yy = y[ok]; tt = t[ok]
    wgt = 2.0 ** (-(c[m] - 1 - tt) / 32)
    pf = (wgt[f] @ yy[f] + 2 * 0.85) / (wgt[f].sum() + 2); pu = (wgt[~f] @ yy[~f] + 2 * 0.1) / (wgt[~f].sum() + 2)
    rows.append((pf, pu))
pf, pu = np.array(rows).T
pw4 = np.where(fav == 1, pf, np.where(fav == -1, pu, pf))
score(np.column_stack([1 - pw4, pw4[:, None] * mixfor(q["machine_type"])]), "per-machine state rates (hl32) + type mix")
