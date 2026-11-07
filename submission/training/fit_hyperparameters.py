"""Fit the model's hyperparameters from a round's TRAIN split and write hyperparameters.json.

Offline training step (not run by the organizers' reproduction). Needs numpy, pandas and scipy
(tested with numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, Python 3.13).

    python training/fit_hyperparameters.py --data path/to/public --out hyperparameters.json

Everything comes from <data>/train: no validation or test information is used.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prototype_model as pm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="round public folder (contains train/)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    meta = json.loads((args.data / "metadata.json").read_text())
    hp = pm.fit_hyper(args.data.resolve())
    out = {
        "model": "house-edge-bayes-rule-v1",
        "fitted_on": {"split": "train", "round": meta.get("round"), "season_id": meta.get("season_id"),
                      "suite_version": meta.get("suite_version")},
        "settings": {"ep_half_life": pm.EP_HALF_LIFE, "ep_var": pm.EP_VAR, "machine_var": pm.MACHINE_VAR,
                     "share_strength": pm.SHARE_STRENGTH},
        "pair_logprior": [[i, j, float(v)] for (i, j), v in hp["pair_logprior"].items()],
        "rates": [[t, int(s), float(pf), float(pu)] for (t, s), (pf, pu) in hp["rates"].items()],
        "strength": {t: [float(a), float(b)] for t, (a, b) in hp["strength"].items()},
        "flip_prior": [[int(k), float(v)] for k, v in hp["flip_prior"].items()],
        "p_flip_long": hp["p_flip_long"], "p_flip_short": hp["p_flip_short"],
        "share1": {t: float(v) for t, v in hp["share1"].items()},
        "hi_mix": {t: [float(x) for x in v] for t, v in hp["hi_mix"].items()},
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
