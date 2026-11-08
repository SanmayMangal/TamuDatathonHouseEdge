""""What changed?" report for a new kit: checks every structural assumption of the rule model.

    python round2/diagnose.py --data NEW/public --hp hp_train.json --val-pred val_v2.csv \
        [--main-val-pred val_main.csv] --out report.md

Sections: kit facts, learned structure (encoding, rule families, top rules, flips), flip timing and
alignment by condition, state win rates, shape of the true odds on validation (2-parameter check,
fixed high-prize split, bimodality), validation scores (this model, the main-round model, reference),
and residual structure (episode correlation). Lines starting with ALERT flag broken assumptions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

Q = ["p_0", "p_1", "p_2", "p_5", "p_20"]
MAIN_RULES = {(1, 3), (2, 5), (4, 7)}


def cond_err(df, err):
    d = pd.DataFrame({"c": df["condition"].to_numpy(), "e": df["episode_id"].to_numpy(), "x": err})
    return d.groupby(["c", "e"])["x"].mean().groupby(level=0).mean()


def score(val, pred_csv):
    p = pd.read_csv(pred_csv).set_index("query_id").loc[val["query_id"], Q].to_numpy()
    t = val[Q].to_numpy(); r = val[["ref_" + c for c in Q]].to_numpy()
    ratio = cond_err(val, ((p - t) ** 2).sum(1)) / cond_err(val, ((r - t) ** 2).sum(1))
    R = float(ratio.mean())
    return R, 60 / (1 + R), ratio, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--hp", required=True)
    ap.add_argument("--map"); ap.add_argument("--val-pred", required=True)
    ap.add_argument("--main-val-pred"); ap.add_argument("--max-points", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pub = Path(a.data); hp = json.loads(Path(a.hp).read_text())
    mp = pd.read_csv(a.map or Path(a.hp).with_suffix(".map.csv"))
    meta = json.loads((pub / "metadata.json").read_text())
    L, alerts = [], []
    say = L.append

    def alert(msg):
        alerts.append(msg); say(f"**ALERT:** {msg}")

    say(f"# What changed? Diagnostics for round `{meta.get('round')}` (season {meta.get('season_id')})\n")
    # ---- kit facts
    say("## Kit")
    for split in ("train", "validation", "test"):
        q = pd.read_csv(pub / split / "queries.csv")
        line = f"- {split}: {q['episode_id'].nunique()} episodes, {len(q)} queries, machines per episode {q.groupby('episode_id').size().unique().tolist()}, cutoffs {q['cutoff'].min()}–{q['cutoff'].max()}, types {q['machine_type'].value_counts().to_dict()}, signals {sorted(q['signal'].unique().tolist())}"
        if (pub / split / "conditions.csv").exists():
            c = pd.read_csv(pub / split / "conditions.csv").merge(q[["query_id", "episode_id"]], on="query_id")
            line += f", conditions {c.groupby('condition')['episode_id'].nunique().to_dict()}"
        say(line)
    if meta.get("round") not in ("adaptation",):
        say(f"- metadata round = `{meta.get('round')}`")
    # ---- learned structure
    say("\n## Learned structure (fit on train)")
    say(f"- Encoding: **{hp['encoding']}** (log evidence {hp['encoding_evidence']})")
    if hp["encoding"] != "win":
        alert("lag encoding is no longer win/loss")
    fam = hp["family_prior"]; say(f"- Rule families: {{{', '.join(f'{k}: {v:.3f}' for k, v in fam.items())}}}")
    if fam["pair"] < 0.8:
        alert(f"rules are no longer mostly lag pairs (pair weight {fam['pair']:.2f})")
    top = sorted(hp["rules"], key=lambda x: -x[1])[:8]
    say("- Top rules (prior): " + ", ".join(f"{tuple(r)} {np.exp(lp):.3f}" for r, lp in top))
    new_top = {tuple(r) for r, lp in top if np.exp(lp) > 0.05}
    if new_top != MAIN_RULES:
        say(f"- Rules differ from the main round's {sorted(MAIN_RULES)}: now {sorted(new_top)}")
    say(f"- MAP rule-family counts on train machines: {mp['rule_family'].map({1: 'single', 2: 'pair', 3: 'triple'}).value_counts().to_dict()}")
    say(f"- P(flip): long {hp['p_flip_long']:.3f}, short {hp['p_flip_short']:.3f}")
    # ---- flips by condition
    say("\n## Flips by condition (MAP on train)")
    mp["flipped"] = mp["flip_d"].notna() & (mp["flip_d"] > 0)
    if mp["condition"].notna().any():
        g = mp.groupby("condition").agg(machines=("flipped", "size"), frac_flipped=("flipped", "mean"),
                                        flip_d_median=("flip_d", "median"), flip_d_min=("flip_d", "min"), flip_d_max=("flip_d", "max"))
        say(g.round(3).to_markdown())
        f = mp[mp["flipped"]].assign(tick=lambda d: d["cutoff"] - d["flip_d"])
        al = f.groupby(["condition", "episode_id"])["tick"].agg(["size", "std"])
        al = al[al["size"] >= 3].groupby("condition")[["size", "std"]].median()
        say("\nWithin-episode spread of flip ticks (median over episodes with ≥3 flips; ~0 means simultaneous):\n")
        say(al.round(2).to_markdown())
        for cond, row in g.iterrows():
            if cond == "stable" and row["frac_flipped"] > 0.1:
                alert("stable episodes now show flips")
    # ---- rates
    say("\n## State win rates (favorable / unfavorable) by type and signal")
    say(pd.DataFrame(hp["rates"], columns=["type", "signal", "p_fav", "p_unfav"]).round(3).to_markdown(index=False))
    say(f"\nBeta strengths: {hp['strength']}  |  1x share: {hp['share1']}  |  high mix: { {k: [round(x, 3) for x in v] for k, v in hp['hi_mix'].items()} }  |  high-mix strength: {hp['hi_strength']}")
    # ---- truth shape on validation
    val = pd.read_csv(pub / "validation/queries.csv").merge(pd.read_csv(pub / "validation/conditions.csv"), on="query_id") \
        .merge(pd.read_csv(pub / "validation/targets.csv").drop(columns=["episode_id", "outcome"], errors="ignore"), on="query_id") \
        .merge(pd.read_csv(pub / "validation/reference.csv").rename(columns={c: "ref_" + c for c in Q}), on="query_id")
    t = val[Q].to_numpy()
    say("\n## Shape of the true odds (validation)")
    for typ in sorted(val["machine_type"].unique()):
        m = (val["machine_type"] == typ).to_numpy()
        Lg = np.log(np.clip(t[m], 1e-12, None)); Lg -= Lg.mean(1, keepdims=True); Lg -= Lg.mean(0)
        sv = np.linalg.svd(Lg, compute_uv=False); share = sv ** 2 / (sv ** 2).sum()
        hi = t[m, 2:] / t[m, 2:].sum(1, keepdims=True)
        qwin = 1 - t[m, 0]
        say(f"- {typ}: variance in first two components {share[:2].sum():.4f} (main round: 1.000); "
            f"sd of 2x:5x:20x split {np.round(hi.std(0), 4).tolist()} (main: 0); "
            f"win chance quantiles {np.round(np.quantile(qwin, [.1, .25, .5, .75, .9]), 3).tolist()}; "
            f"share of win chances in 0.3–0.7: {((qwin > .3) & (qwin < .7)).mean():.2f}")
        if share[:2].sum() < 0.99:
            alert(f"{typ}: true odds need more than 2 parameters (size mix may depend on state)")
        if hi.std(0).max() > 0.01:
            alert(f"{typ}: the 2x:5x:20x split varies by machine/tick")
    # ---- scores
    say("\n## Validation scores (exact)")
    R, pts, ratio, p = score(val, a.val_pred)
    scale = a.max_points or meta.get("max_points") or 60
    say(f"- Generalized model (fit on train): R **{R:.4f}** ({scale / (1 + R):.2f} / {scale}) | " + ", ".join(f"{k} {v:.3f}" for k, v in ratio.items()))
    if a.main_val_pred:
        R2, _, ratio2, _ = score(val, a.main_val_pred)
        say(f"- Main-round submission, unchanged: R {R2:.4f} ({scale / (1 + R2):.2f} / {scale}) | " + ", ".join(f"{k} {v:.3f}" for k, v in ratio2.items()))
    say(f"- Reference: R 1.0000 ({scale / 2:.2f} / {scale})")
    if R > 0.3:
        alert(f"generalized model R = {R:.2f}: structure is being missed; inspect residuals")
    # ---- residuals
    lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
    res = lg(1 - t[:, 0]) - lg(1 - p[:, 0])
    ep = val["episode_id"].to_numpy()
    nb = np.array([res[(ep == ep[i]) & (np.arange(len(ep)) != i)].mean() for i in range(len(ep))])
    say("\n## Residuals (true vs predicted win log-odds)")
    say(f"- sd {res.std():.3f} (main round ~0.6); by condition {pd.Series(res).groupby(val['condition'].to_numpy()).std().round(3).to_dict()}")
    say(f"- correlation with rest-of-episode mean residual: {np.corrcoef(res, nb)[0, 1]:.3f} (main round: 0.47; stronger offsets did not help there, so a similar value is expected)")
    big = np.abs(res) > 2
    say(f"- |residual| > 2: {big.mean():.3f} of queries; by condition {pd.Series(big).groupby(val['condition'].to_numpy()).mean().round(3).to_dict()}")
    say("\n## Alerts\n" + ("\n".join(f"- {x}" for x in alerts) if alerts else "- none: all main-round assumptions hold"))
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L[-(len(alerts) + 2):]))


if __name__ == "__main__":
    main()
