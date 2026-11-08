"""House Edge round-two forecaster (generalized Bayesian rule model). Python standard library only.

    python predict.py --data /data --out /output/submission.csv
    python predict.py --data path/to/public --split validation --out val.csv

Loads hyperparameters.json (fitted offline by fit.py on the round's labeled splits). For each query:
  1. Posterior over hypotheses (rule = set of lags with learned prior, direction, flip split). The rule
     state at tick t is the even parity of the encoded outcomes at t-k for k in the rule; one side of the
     state is favorable; one optional flip reverses the side. Beta-Bernoulli marginal likelihoods.
  2. Predictive win chance at the cutoff, plus episode-wide and per-machine log-odds offsets.
  3. Size mix: P(1x | win) and the 2x:5x:20x split, each shrunk from the machine's history to its type.
Uses only ticks before the cutoff, and other machines only from the same episode. Deterministic.
"""
import argparse
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
        queries = list(csv.DictReader(f))
    queries.sort(key=lambda r: (r["episode_id"], r["machine_id"]))
    hist = {}
    with open(d / "history.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            hist.setdefault(r["machine_id"], {})[int(r["tick"])] = PAYOUTS.index(int(r["payout"]))
    out = []
    for q in queries:
        c = int(q["cutoff"]); h = hist.get(q["machine_id"], {})
        if any(t >= c for t in h):
            raise ValueError(f"history at or after cutoff for {q['machine_id']}")
        out.append({"query_id": q["query_id"], "episode_id": q["episode_id"], "type": q["machine_type"],
                    "signal": int(q["signal"]), "cutoff": c, "payout": h,
                    "short": any(t not in h for t in range(c))})
    return out


class Model:
    def __init__(self, hp, min_rule_prior=1e-3):
        self.hp = hp
        self.max_lag = max(max(r) for r, _ in hp["rules"])
        self.rules = [(tuple(r), lp) for r, lp in hp["rules"] if math.exp(lp) >= min_rule_prior]
        self.rates = {(t, int(s)): (pf, pu) for t, s, pf, pu in hp["rates"]}
        self.thr = hp["encoding_threshold"]
        h = hp["flip_hist"]; self.dmax = len(h) - 1
        self.hcum = [0.0] * len(h)
        acc = 0.0
        for i, v in enumerate(h):
            acc += v; self.hcum[i] = acc
        st = hp["settings"]
        self.ep_half_life, self.ep_var = st["ep_half_life"], st["ep_var"]
        self.machine_var, self.share_strength = st["machine_var"], st["share_strength"]

    def H(self, a, b):
        a = min(max(a, 1), self.dmax); b = min(max(b, 0), self.dmax)
        return self.hcum[b] - self.hcum[a - 1] if b >= a else 0.0

    def split_logmass(self, ticks, c, p_flip):
        n = len(ticks)
        if n == 0:
            return [0.0]
        mass = []
        for k in range(n):
            hi = self.dmax if k == 0 else c - ticks[k - 1] - 1
            mass.append(p_flip * self.H(c - ticks[k], hi))
        mass.append(p_flip * self.H(1, c - ticks[-1] - 1))
        mass[0] += 1 - p_flip
        return [math.log(max(m, 1e-300)) for m in mass]

    def rule_series(self, m, rule):
        """Valid ticks, state (even parity), win outcome; and the state at the cutoff (None if unknown)."""
        p, c, thr = m["payout"], m["cutoff"], self.thr
        ticks, S, W = [], [], []
        for t in range(self.max_lag, c):
            y = p.get(t)
            if y is None:
                continue
            par = 0
            for k in rule:
                v = p.get(t - k)
                if v is None:
                    break
                par += v >= thr
            else:
                ticks.append(t); S.append(par % 2 == 0); W.append(1 if y >= 1 else 0)
        par = 0
        for k in rule:
            v = p.get(c - k)
            if v is None:
                par = None
                break
            par += v >= thr
        return ticks, S, W, (None if par is None else par % 2 == 0)

    def machine_posterior(self, m):
        pf0, pu0 = self.rates.get((m["type"], m["signal"]), self.hp["type_rates"][m["type"]])
        sf, su = self.hp["strength"][m["type"]]
        af, bf, au, bu = pf0 * sf, (1 - pf0) * sf, pu0 * su, (1 - pu0) * su
        base_f, base_u = betaln(af, bf), betaln(au, bu)
        p_flip = self.hp["p_flip_short"] if m["short"] else self.hp["p_flip_long"]
        c = m["cutoff"]; log_half = math.log(0.5)
        Ls, Ps = [], []
        best_L, best = -math.inf, None
        for rule, lp in self.rules:
            ticks, S, W, now = self.rule_series(m, rule)
            n = len(ticks)
            A1 = [0]; N1 = [0]; Yc = [0]
            for s, w in zip(S, W):
                A1.append(A1[-1] + (w if s else 0)); N1.append(N1[-1] + (1 if s else 0)); Yc.append(Yc[-1] + w)
            A1T, N1T, YT = A1[-1], N1[-1], Yc[-1]
            A0T, N0T = YT - A1T, n - N1T
            lmass = self.split_logmass(ticks, c, p_flip)
            for k in range(n + 1):
                a1p, n1p = A1[k], N1[k]
                a0p, n0p = Yc[k] - a1p, k - n1p
                kP, nP = a0p + (A1T - a1p), n0p + (N1T - n1p)   # cell P: after split state on, before split state off
                kQ, nQ = a1p + (A0T - a0p), n1p + (N0T - n0p)
                for d in (True, False):                          # d=True: cell P favorable
                    kf, nf, ku, nu = (kP, nP, kQ, nQ) if d else (kQ, nQ, kP, nP)
                    ll = betaln(af + kf, bf + nf - kf) - base_f + betaln(au + ku, bu + nu - ku) - base_u
                    pf = (af + kf) / (af + bf + nf); pu = (au + ku) / (au + bu + nu)
                    pw = 0.5 * (pf + pu) if now is None else (pf if now == d else pu)
                    L = ll + lp + log_half + lmass[k]
                    Ls.append(L); Ps.append(pw)
                    if L > best_L:
                        best_L, best = L, (rule, k, d, pf, pu)
        mx = max(Ls); num = den = 0.0
        for L, p in zip(Ls, Ps):
            e = math.exp(L - mx); num += e * p; den += e
        return num / den, best

    def states(self, m, hyp):
        rule, k, d, pf, pu = hyp
        ticks, S, W, _ = self.rule_series(m, rule)
        out = []
        for idx, (t, s, w) in enumerate(zip(ticks, S, W)):
            inP = s if idx >= k else (not s)
            out.append((t, w, pf if inP == d else pu))
        return out

    def predict(self, machines):
        logit, series = [], []
        for m in machines:
            pw, hyp = self.machine_posterior(m)
            pw = min(max(pw, 1e-9), 1 - 1e-9)
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
                    r = sum(y - p for _, y, p in series[k]); info = sum(p * (1 - p) for _, _, p in series[k])
                    logit[k] += r / (info + 1 / self.machine_var)
        rows = []
        for m, z in zip(machines, logit):
            pw = 1 / (1 + math.exp(-z))
            vals = list(m["payout"].values())
            wins = [v for v in vals if v > 0]
            ones = sum(1 for v in wins if v == 1)
            typ = m["type"]
            s = (ones + self.share_strength * self.hp["share1"][typ]) / (len(wins) + self.share_strength)
            hs, tm = self.hp["hi_strength"][typ], self.hp["hi_mix"][typ]
            hc = [sum(1 for v in vals if v == j) for j in (2, 3, 4)]
            nh = sum(hc)
            hi = [(hc[j] + hs * tm[j]) / (nh + hs) for j in range(3)]
            probs = [1 - pw, pw * s] + [pw * (1 - s) * x for x in hi]
            total = sum(probs)
            rows.append([m["query_id"]] + [p / total for p in probs])
        return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--hyperparameters", default=str(HERE / "hyperparameters.json"))
    args = ap.parse_args()
    hp = json.loads(Path(args.hyperparameters).read_text(encoding="utf-8"))
    rows = Model(hp).predict(load(args.data, args.split))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(COLUMNS)
        for r in rows:
            wr.writerow([r[0]] + [repr(float(p)) for p in r[1:]])
    print(f"wrote {len(rows)} forecasts to {out}")


if __name__ == "__main__":
    main()
