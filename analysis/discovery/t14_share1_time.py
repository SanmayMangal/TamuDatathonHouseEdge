import os, sys; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); os.makedirs(os.path.join(HERE, '../artifacts/discovery'), exist_ok=True); os.chdir(os.path.join(HERE, '../artifacts/discovery'))
from discovery_lib import *
for split in ("train", "validation"):
    q, Y = load_split(split)
    acs = {k: [] for k in (1, 2, 3, 5, 8, 16, 32, 64)}
    for m in range(len(q)):
        if q.loc[m, "short"]: continue
        y = Y[m, :q.loc[m, "cutoff"]]
        z = np.where(y > 0, (y == 1).astype(float), np.nan)
        z = z - np.nanmean(z)
        for k in acs:
            a, b = z[k:], z[:-k]; ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() > 10: acs[k].append((a[ok] * b[ok]).mean() / np.nanvar(z))
    print(split, "autocorr of 'payout was 1x' among wins, by tick distance:", {k: round(np.mean(v), 3) for k, v in acs.items()})
    # windowed share drift: first half vs second half share1 per machine, correlation
    f, s = [], []
    for m in range(len(q)):
        if q.loc[m, "short"]: continue
        y = Y[m, :q.loc[m, "cutoff"]]; h = len(y) // 2
        a, b = y[:h][y[:h] > 0], y[h:][y[h:] > 0]
        if len(a) > 20 and len(b) > 20: f.append((a == 1).mean()); s.append((b == 1).mean())
    f, s = np.array(f), np.array(s)
    print(f"   corr(share1 first half, second half) = {np.corrcoef(f, s)[0,1]:.3f}, sd first {f.std():.3f}, sd of diff {np.std(f-s):.3f}")
