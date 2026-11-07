"""E7. How do the odds change over time?

  memory     : causal forecasts (each tick predicted only from earlier ticks) with recency
               half-lives from 4 ticks to none, scored per condition on history ticks >= 32
               against no decay, and exactly against validation targets at the cutoff.
  changes    : per machine, the strongest single change point in its payout mix (log-likelihood
               ratio of two segments vs one), with a shuffled-within-machine null.
  timing     : in each episode, how tightly the change points of machines with a real change
               line up (shared change should line up; drift shouldn't have sharp points).
  trend      : per machine, a linear trend in the chance of a nonzero payout over time.
  synchrony  : correlation between machines' win-rate swings (16-tick windows) within an
               episode, against pairs drawn from different episodes as a null.
"""
import numpy as np
import pandas as pd

from .common import K, cutoff_predictions, exact_score, grid_gain_rows, history_grid, online_decay

QUESTION = "Do the odds drift or jump, and do machines in an episode change together?"
SHOW = ["memory", "memory_exact", "changes", "timing", "trend", "synchrony"]
HALF_LIVES = (4, 8, 16, 32, 64, 128, 256, None)
STRENGTHS = (5.0, 20.0)
MIN_SEG = 16
WINDOW = 16


def _best_change(seq):
    """Max two-segment vs one-segment multinomial log-likelihood ratio over change points."""
    n = len(seq)
    if n < 2 * MIN_SEG:
        return np.nan, -1
    oh = np.zeros((n, K))
    oh[np.arange(n), seq] = 1
    cum = np.vstack([np.zeros(K), np.cumsum(oh, 0)])
    total = cum[-1]

    def ll(c):
        c = c + 0.5
        return (c * np.log(c / c.sum(-1, keepdims=True))).sum(-1)

    taus = np.arange(MIN_SEG, n - MIN_SEG + 1)
    left = cum[taus]
    llr = ll(left) + ll(total - left) - ll(total[None])
    i = int(np.argmax(llr))
    return float(llr[i]), int(taus[i])


def run(ctx):
    Y, meta, prior = history_grid(ctx)
    out = {}

    # Memory length.
    mem, exact = [], []
    for s in STRENGTHS:
        P_flat = online_decay(Y, prior, s, None)
        exact += exact_score(ctx, meta, cutoff_predictions(meta, P_flat), strength=s, half_life="none")
        for hl in HALF_LIVES[:-1]:
            P = online_decay(Y, prior, s, hl)
            mem += grid_gain_rows(ctx, meta, Y, P_flat, P, min_tick=32, strength=s, half_life=hl)
            exact += exact_score(ctx, meta, cutoff_predictions(meta, P), strength=s, half_life=hl)
    mem = pd.DataFrame(mem).pivot_table(index=["strength", "half_life"], columns="condition", values="dR")
    out["memory"] = mem.reset_index().rename_axis(columns=None)
    if exact:
        ex = pd.DataFrame(exact).pivot_table(index=["strength", "half_life"], columns="condition",
                                             values="ratio", aggfunc="first")
        out["memory_exact"] = ex.reset_index().rename_axis(columns=None)
    else:
        out["memory_exact"] = pd.DataFrame()

    # Change points, with a shuffled null per machine.
    rows = []
    for m in range(len(meta)):
        obs = np.flatnonzero(Y[m] >= 0)
        seq = Y[m, obs]
        llr, tau = _best_change(seq)
        llr0, _ = _best_change(ctx.rng.permutation(seq))
        trend_t = np.nan
        if len(seq) > 10:
            w = (seq > 0).astype(float)
            t = obs - obs.mean()
            slope = (t * (w - w.mean())).sum() / (t ** 2).sum()
            resid = w - w.mean() - slope * t
            se = np.sqrt((resid ** 2).sum() / (len(w) - 2) / (t ** 2).sum())
            trend_t = slope / se if se > 0 else np.nan
        rows.append({"machine_id": meta.loc[m, "machine_id"], "episode_id": meta.loc[m, "episode_id"],
                     "condition": meta.loc[m, "condition"], "llr": llr, "llr_null": llr0,
                     "change_tick": obs[tau] if tau >= 0 else np.nan,
                     "change_before_cutoff": meta.loc[m, "cutoff"] - obs[tau] if tau >= 0 else np.nan,
                     "trend_t": trend_t})
    cp = pd.DataFrame(rows)
    thr = cp["llr_null"].quantile(0.95)
    cp["real_change"] = cp["llr"] > thr
    out["changes"] = cp.groupby("condition").agg(
        machines=("llr", "size"), median_llr=("llr", "median"), median_llr_null=("llr_null", "median"),
        frac_real_change=("real_change", "mean"),
        median_ticks_before_cutoff=("change_before_cutoff", "median")).reset_index()
    out["changes"]["null95_threshold"] = thr

    t = cp[cp["real_change"]].groupby(["condition", "episode_id"]).agg(
        n=("change_tick", "size"), sd=("change_tick", "std"),
        span=("change_tick", lambda x: x.max() - x.min()))
    t = t[t["n"] >= 2]
    out["timing"] = t.groupby("condition").agg(
        episodes=("n", "size"), mean_machines_changed=("n", "mean"),
        median_sd_change_tick=("sd", "median"), median_span=("span", "median")).reset_index()

    out["trend"] = cp.groupby("condition").agg(
        frac_abs_t_gt2=("trend_t", lambda x: (x.abs() > 2).mean()),
        frac_t_gt2=("trend_t", lambda x: (x > 2).mean()),
        frac_t_lt_minus2=("trend_t", lambda x: (x < -2).mean())).reset_index()

    # Synchrony of win-rate swings, long histories only.
    W = np.where(Y >= 0, (Y > 0).astype(float), np.nan)
    nwin = Y.shape[1] // WINDOW
    R = np.nanmean(W[:, : nwin * WINDOW].reshape(len(W), nwin, WINDOW), axis=2)
    R = R - np.nanmean(R, 1, keepdims=True)
    long = ~meta["short_history"].to_numpy()

    def corr(a, b):
        ok = ~np.isnan(a) & ~np.isnan(b)
        return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() >= 5 else np.nan

    syn = []
    eps = meta[long].groupby("episode_id").indices
    ep_list = list(eps)
    for e_i, e in enumerate(ep_list):
        idx = meta.index[long][eps[e]]
        cond = meta.loc[idx[0], "condition"]
        same = [corr(R[a], R[b]) for i, a in enumerate(idx) for b in idx[i + 1:]]
        other_e = ep_list[(e_i + 1) % len(ep_list)]
        oidx = meta.index[long][eps[other_e]]
        cross = [corr(R[a], R[b]) for a in idx for b in oidx]
        syn.append({"condition": cond, "within": np.nanmean(same), "across_episodes": np.nanmean(cross)})
    out["synchrony"] = (pd.DataFrame(syn).groupby("condition").mean().reset_index()
                        if syn else pd.DataFrame())
    out["per_machine_changes"] = cp
    return out
