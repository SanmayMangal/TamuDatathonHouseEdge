import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from discovery_lib import *
q = pd.read_pickle("val_state3.pkl")
_, Y = load_split("validation"); W = win_grid(Y)
c = q["cutoff"].to_numpy(); n = len(q)
lg = lambda p: np.log(p / (1 - p))
base = {"balanced": -0.64, "jackpot": -2.40, "steady": 0.74}
fav = np.where(q["fav"], 1, -1)
form = q["machine_type"].map(base).to_numpy() + 0.47 * q["signal"].to_numpy() + 2.17 * fav
q["res"] = lg(q["qwin"]) - form
# per-machine history: logit of win rate in fav state and unfav state (all history, smoothed), minus formula's
pf, pu = [], []
for m in range(n):
    p = q.loc[m, "rpair"]
    if not isinstance(p, str): pf.append(np.nan); pu.append(np.nan); continue
    i, j = map(int, p.split("-")); d = q.loc[m, "rd"]
    t = np.arange(8, c[m]); a, b, y = W[m, t-i], W[m, t-j], W[m, t]
    ok = ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(y)
    f = ((a == b) == (d > 0))[ok]; yy = y[ok]
    pf.append(lg((yy[f].sum() + .5) / (f.sum() + 1))); pu.append(lg((yy[~f].sum() + .5) / ((~f).sum() + 1)))
q["h_fav"], q["h_unf"] = pf, pu
mt = q["machine_type"].map(base).to_numpy() + 0.47 * q["signal"].to_numpy()
q["h_dev"] = np.where(fav == 1, q["h_fav"] - (mt + 2.17), q["h_unf"] - (mt - 2.17))
q["h_mid"] = (q["h_fav"] + q["h_unf"]) / 2 - mt          # machine base offset
q["h_amp"] = (q["h_fav"] - q["h_unf"]) / 2 - 2.17        # machine amplitude offset
L = q[(~q["short"]) & (q["condition"] == "stable") & q["agree"]]
print("stable, long, correctly-classified machines n =", len(L))
print("residual sd:", L["res"].std().round(3))
for col in ["h_dev", "h_mid", "h_amp"]:
    print(f"corr(res, {col}) = {np.corrcoef(L['res'], L[col])[0,1]:.3f}")
b = np.polyfit(L["h_mid"], L["res"], 1); print("slope of res on h_mid:", b.round(3))
# split by state: does base offset or amplitude matter more
for fv in (1, -1):
    g = L[np.where(L["fav"], 1, -1) == fv]
    print(f"state {fv}: corr(res, h_mid) {np.corrcoef(g['res'], g['h_mid'])[0,1]:.3f}  corr(res, h_amp) {np.corrcoef(g['res'], g['h_amp'])[0,1]:.3f}  corr(res,h_dev) {np.corrcoef(g['res'], g['h_dev'])[0,1]:.3f}")
q.to_pickle("val_state4.pkl")
