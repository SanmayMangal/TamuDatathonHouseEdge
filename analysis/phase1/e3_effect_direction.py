"""E3. Do machines of the same type react in opposite directions?

For every machine and lag k = 1..12, the effect is
    d = P(next payout > 0 | payout at lag k > 0) - P(next payout > 0 | payout at lag k = 0)
with a z-score from its standard error. Also the same difference in mean return.
Summaries: per (type, lag) the spread and sign split of significant effects; per
(type, signal, lag) the mean effect (does signal set the direction?); and each machine's
strongest lag. Lags 9-12 give the rate of "significant" effects expected by chance.
"""
import numpy as np
import pandas as pd

from .common import lag

QUESTION = "Are lag effects machine-specific or opposite within a type, and does signal set their direction?"
SHOW = ["by_type_lag", "by_type_signal_lag", "strongest_lag", "within_episode_agreement"]
LAGS = range(1, 13)


def run(ctx):
    h = ctx.hist
    nxt_win = (ctx.y > 0).astype(float)
    ret = h["outcome"].to_numpy().astype(float)
    rows = []
    for k in LAGS:
        x = lag(ctx, k, "win")
        ok = ~np.isnan(x)
        d = pd.DataFrame({"machine_id": h["machine_id"].astype(str).to_numpy()[ok], "x": x[ok],
                          "w": nxt_win[ok], "r": ret[ok]})
        g = d.groupby(["machine_id", "x"]).agg(n=("w", "size"), p=("w", "mean"), r=("r", "mean"),
                                               rv=("r", "var")).unstack("x")
        n1, n0 = g[("n", 1.0)], g[("n", 0.0)]
        p1, p0 = g[("p", 1.0)], g[("p", 0.0)]
        pp = (p1 * n1 + p0 * n0) / (n1 + n0)
        se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
        rse = np.sqrt(g[("rv", 1.0)] / n1 + g[("rv", 0.0)] / n0)
        rows.append(pd.DataFrame({"machine_id": g.index, "lag": k, "n_after_win": n1.values,
                                  "n_after_loss": n0.values, "d_win": (p1 - p0).values,
                                  "z_win": ((p1 - p0) / se).values,
                                  "d_return": (g[("r", 1.0)] - g[("r", 0.0)]).values,
                                  "z_return": ((g[("r", 1.0)] - g[("r", 0.0)]) / rse).values}))
    eff = pd.concat(rows, ignore_index=True)
    meta = (h.groupby("machine_id", observed=True)
            .agg(episode_id=("episode_id", "first"), machine_type=("machine_type", "first"),
                 signal=("signal", "first"), condition=("condition", "first")).reset_index())
    meta["machine_id"] = meta["machine_id"].astype(str)
    meta["machine_type"] = meta["machine_type"].astype(str)
    eff = eff.merge(meta, on="machine_id")

    def split(g):
        z = g["z_win"]
        return pd.Series({"n_machines": len(g), "mean_d": g["d_win"].mean(), "sd_d": g["d_win"].std(),
                          "frac_z>2": (z > 2).mean(), "frac_z<-2": (z < -2).mean(),
                          "frac_|z|>3": (z.abs() > 3).mean()})

    by_type = eff.groupby(["machine_type", "lag"]).apply(split, include_groups=False).reset_index()
    by_sig = (eff.groupby(["machine_type", "signal", "lag"])["d_win"].mean()
              .unstack("lag").reset_index())
    by_sig.columns = [c if isinstance(c, str) else f"lag{c}" for c in by_sig.columns]

    # Machines with no observations after a win (or loss) at a lag have no z there.
    real = eff[(eff["lag"] <= 8) & eff["z_win"].notna()]
    top = real.loc[real["z_win"].abs().groupby(real["machine_id"]).idxmax()]
    ctrl_rows = eff[(eff["lag"] > 8) & eff["z_win"].notna()]
    ctrl = ctrl_rows["z_win"].abs().groupby(ctrl_rows["machine_id"]).max()
    strongest = (top.assign(abs_z=top["z_win"].abs())
                 .groupby(["condition", "lag"]).size().unstack(fill_value=0).reset_index())
    strongest.columns = [c if isinstance(c, str) else f"lag{c}" for c in strongest.columns]
    # Share of machines whose strongest lag-1..8 effect has |z| > 3, vs the same share for
    # their strongest control lag (9..12). Lags 1..8 are 8 tries vs 4, so the control is a floor.
    strongest["frac_top_|z|>3"] = strongest["condition"].map(
        (top["z_win"].abs() > 3).groupby(top["condition"]).mean())
    cond_of = meta.set_index("machine_id")["condition"].reindex(ctrl.index)
    strongest["control_frac_|z|>3"] = strongest["condition"].map((ctrl > 3).groupby(cond_of).mean())

    # Within an episode, do significant effects at the same lag share a sign?
    sig = real[real["z_win"].abs() > 3]
    agree = (sig.groupby(["condition", "episode_id", "lag"])["z_win"]
             .agg(n="size", pos=lambda z: (z > 0).sum()).reset_index())
    agree = agree[agree["n"] >= 2]
    agree["same_sign_share"] = np.maximum(agree["pos"], agree["n"] - agree["pos"]) / agree["n"]
    agreement = agree.groupby("condition").agg(groups=("n", "size"),
                                               mean_same_sign_share=("same_sign_share", "mean")).reset_index()
    return {"by_type_lag": by_type, "by_type_signal_lag": by_sig, "strongest_lag": strongest,
            "within_episode_agreement": agreement, "per_machine_effects": eff}
