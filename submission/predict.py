"""House Edge forecaster: Bayesian "matching-pair" rule model. Python standard library only.

    python predict.py --data /data --out /output/submission.csv            # test split (submission)
    python predict.py --data path/to/public --split validation --out val.csv

Loads hyperparameters.json (fitted offline on the round's train split by training/fit_hyperparameters.py)
and forecasts each query's next payout from that machine's history and its episode's histories,
using only ticks before the cutoff. Deterministic: no randomness, float64 throughout.

Model, per machine (see README.md):
  1. Posterior over hypotheses (lag pair i<j <= 8, direction, flip tick or none). The rule state at tick t
     is whether win/loss at t-i equals win/loss at t-j; one side is favorable. A flip reverses the side.
     Each hypothesis is scored with Beta-Bernoulli marginal likelihoods of the favorable/unfavorable cells.
  2. Predictive win chance at the cutoff, plus episode-wide and machine log-odds offsets estimated from
     the residuals under each machine's most likely hypothesis.
  3. P(1x | win) = machine's 1x share among its wins, shrunk to the type mean; 2x:5x:20x fixed per type.
"""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path

PAYOUTS = (0, 1, 2, 5, 20)
COLUMNS = ("query_id", "p_0", "p_1", "p_2", "p_5", "p_20")
HERE = Path(__file__).resolve().parent


def betaln(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def load(public, split):
    d = Path(public) / split
    with open(d / "queries.csv", newline="", encoding="utf-8") as f:
        queries = [r for r in csv.DictReader(f)]
    queries.sort(key=lambda r: (r["episode_id"], r["machine_id"]))
    hist = {}
    with open(d / "history.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            hist.setdefault(r["machine_id"], {})[int(r["tick"])] = PAYOUTS.index(int(r["payout"]))
    machines = []
    for q in queries:
        c = int(q["cutoff"])
        h = hist.get(q["machine_id"], {})
        if any(t >= c for t in h):
            raise ValueError(f"history at or after cutoff for {q['machine_id']}")
        win = {t: (1 if v > 0 else 0) for t, v in h.items()}
        machines.append({"query_id": q["query_id"], "episode_id": q["episode_id"], "type": q["machine_type"],
                         "signal": int(q["signal"]), "cutoff": c, "payout": h, "win": win,
                         "short": any(t not in h for t in range(c))})
    return machines


class Model:
    def __init__(self, hp):
        self.hp = hp
        self.pairs = [((int(i), int(j)), float(lp)) for i, j, lp in hp["pair_logprior"]]
        self.rates = {(t, int(s)): (pf, pu) for t, s, pf, pu in hp["rates"]}
        self.flip = [(int(k), float(p)) for k, p in hp["flip_prior"]]
        st = hp["settings"]
        self.ep_half_life, self.ep_var = st["ep_half_life"], st["ep_var"]
        self.machine_var, self.share_strength = st["machine_var"], st["share_strength"]

    def machine_posterior(self, m):
        """Returns (posterior predictive win prob, MAP hypothesis) for one machine."""
        c, w = m["cutoff"], m["win"]
        pf0, pu0 = self.rates[(m["type"], m["signal"])]
        sf, su = self.hp["strength"][m["type"]]
        af, bf, au, bu = pf0 * sf, (1 - pf0) * sf, pu0 * su, (1 - pu0) * su
        base_f, base_u = betaln(af, bf), betaln(au, bu)
        p_flip = self.hp["p_flip_short"] if m["short"] else self.hp["p_flip_long"]
        taus = [-1] + [c - k for k, _ in self.flip]
        ltau = [math.log(max(1 - p_flip, 1e-12))] + [math.log(max(p_flip, 1e-12)) + math.log(p) for _, p in self.flip]
        log_half = math.log(0.5)
        best_L, best_h = -math.inf, None
        Ls, Ps = [], []
        for (i, j), lp in self.pairs:
            # Valid ticks t in [8, c): win/loss known at t, t-i and t-j. Prefix sums over them, in tick order.
            ticks, A1, N1, Yc = [], [0], [0], [0]
            for t in range(8, c):
                y, a, b = w.get(t), w.get(t - i), w.get(t - j)
                if y is None or a is None or b is None:
                    continue
                mt = 1 if a == b else 0
                ticks.append(t); A1.append(A1[-1] + mt * y); N1.append(N1[-1] + mt); Yc.append(Yc[-1] + y)
            n = len(ticks)
            A1T, N1T, YT = A1[-1], N1[-1], Yc[-1]
            A0T, N0T = YT - A1T, n - N1T
            xa, xb = w.get(c - i), w.get(c - j)
            now = None if (xa is None or xb is None) else (xa == xb)
            for d in (True, False):
                for tau, lt in zip(taus, ltau):
                    k = bisect.bisect_left(ticks, tau) if tau >= 0 else 0  # ticks before the flip
                    a1p, n1p = A1[k], N1[k]
                    a0p, n0p = Yc[k] - a1p, k - n1p
                    if d:   # after the flip a match is favorable; before it a mismatch was
                        kf, nf = a0p + (A1T - a1p), n0p + (N1T - n1p)
                        ku, nu = a1p + (A0T - a0p), n1p + (N0T - n0p)
                    else:   # after the flip a mismatch is favorable; before it a match was
                        kf, nf = a1p + (A0T - a0p), n1p + (N0T - n0p)
                        ku, nu = a0p + (A1T - a1p), n0p + (N1T - n1p)
                    ll = (betaln(af + kf, bf + nf - kf) - base_f) + (betaln(au + ku, bu + nu - ku) - base_u)
                    pf = (af + kf) / (af + bf + nf)
                    pu = (au + ku) / (au + bu + nu)
                    pw = 0.5 * (pf + pu) if now is None else (pf if now == d else pu)
                    L = ll + lt + lp + log_half
                    Ls.append(L); Ps.append(pw)
                    if L > best_L:
                        best_L, best_h = L, ((i, j), d, tau, pf, pu)
        mx = max(Ls)
        num = den = 0.0
        for L, p in zip(Ls, Ps):
            e = math.exp(L - mx); num += e * p; den += e
        return num / den, best_h

    @staticmethod
    def states(m, hyp):
        """Ticks, outcomes and predicted win probs under one hypothesis (for residual offsets)."""
        (i, j), d, tau, pf, pu = hyp
        w = m["win"]; out = []
        for t in range(8, m["cutoff"]):
            y, a, b = w.get(t), w.get(t - i), w.get(t - j)
            if y is None or a is None or b is None:
                continue
            match = a == b
            fav = (match != d) if t < tau else (match == d)
            out.append((t, y, pf if fav else pu))
        return out

    def predict(self, machines):
        logit, series = [], []
        for m in machines:
            pw, hyp = self.machine_posterior(m)
            logit.append(math.log(pw / (1 - pw)))
            series.append(self.states(m, hyp))
        episodes = {}
        for k, m in enumerate(machines):
            episodes.setdefault(m["episode_id"], []).append(k)
        for idx in episodes.values():
            num = den = 0.0
            for k in idx:
                c = machines[k]["cutoff"]
                for t, y, p in series[k]:
                    wt = 2.0 ** (-(c - 1 - t) / self.ep_half_life)
                    num += wt * (y - p); den += wt * p * (1 - p)
            shift = num / (den + 1 / self.ep_var) if self.ep_var > 0 else 0.0
            for k in idx:
                logit[k] += shift
            if self.machine_var > 0:
                for k in idx:
                    r = sum(y - p for _, y, p in series[k])
                    info = sum(p * (1 - p) for _, _, p in series[k])
                    logit[k] += r / (info + 1 / self.machine_var)
        rows = []
        for m, z in zip(machines, logit):
            pw = 1 / (1 + math.exp(-z))
            wins = [v for v in m["payout"].values() if v > 0]
            ones = sum(1 for v in wins if v == 1)
            s = (ones + self.share_strength * self.hp["share1"][m["type"]]) / (len(wins) + self.share_strength)
            hi = self.hp["hi_mix"][m["type"]]
            probs = [1 - pw, pw * s] + [pw * (1 - s) * x for x in hi]
            total = sum(probs)
            rows.append([m["query_id"]] + [p / total for p in probs])
        return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="the round's public folder (contains test/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--hyperparameters", default=str(HERE / "hyperparameters.json"))
    args = ap.parse_args()
    hp = json.loads(Path(args.hyperparameters).read_text(encoding="utf-8"))
    rows = Model(hp).predict(load(args.data, args.split))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(COLUMNS)
        for r in rows:
            wr.writerow([r[0]] + [repr(float(p)) for p in r[1:]])
    print(f"wrote {len(rows)} forecasts to {out}")


if __name__ == "__main__":
    main()
