# House Edge (TAMU Datathon 2026) — Team Card Sharks

Forecasting the odds of a slot machine's next payout (0×, 1×, 2×, 5×, 20×) from gameplay logs.
We reverse-engineered the simulator's rules and built a Bayesian model that infers each machine's hidden rule.

**Validation: 55.61 / 60 points (R = 0.079)**, against 30.00 for the frozen reference. The score comes from
the kit's `check_submission.py`, and every hyperparameter is fitted on the train split only.

## Repository map

| Path | What it is |
|---|---|
| `submission/` | **The entry.** `predict.py` (standard library only), `hyperparameters.json` (checkpoint), `reproduce.json`, README, and `training/` (offline fitter). |
| `deliverables/main/` | The files uploaded for the main round: test CSV and code ZIP. |
| `analysis/build_table.py`, `evaluate.py`, `verify_table.py` | Phase 0: one example per observed tick, a competition-exact evaluator, and leakage checks. |
| `analysis/phase1/`, `run_phase1.py` | Phase 1: nine experiments (lags, pairs, encodings, neighbors, non-stationarity, short histories, error breakdown) with shuffled-null controls. |
| `analysis/discovery/` | How the simulator works: 26 tests plus `prototype_model.py`. **Start with its README.** |
| `analysis/model_choice/` | Gradient boosting and an MLP compared with the Bayesian model, plus blending and calibration. |
| `analysis/tuning/` | Cross-validated fine-tuning study. Result: keep the submitted settings. |
| `analysis/results/` | Saved outputs of the above (Phase 1 report, benchmark, tuning). |
| `main-participants-60-40/` | The organizers' participant kit and data, unchanged. |

## Reproduce

Python 3.11+, from the repo root:

```sh
# Validation score of the submitted model (standard library only, about 4 s)
python submission/predict.py --data main-participants-60-40/public --split validation --out validation_predictions.csv
cd main-participants-60-40 && python check_submission.py ../validation_predictions.csv --split validation && cd ..

# The uploaded test forecast (about 8 s); matches deliverables/main/house-edge-main-submission.csv
python submission/predict.py --data main-participants-60-40/public --out test_check.csv

# Refit the checkpoint from train only (needs numpy, pandas, scipy; about 20 s); identical to submission/hyperparameters.json
python submission/training/fit_hyperparameters.py --data main-participants-60-40/public --out hp_check.json
```

The analysis scripts need numpy, pandas and scipy, plus scikit-learn for `model_choice/`. Run
`python analysis/build_table.py` once before Phase 0 and Phase 1.

## Approach in brief

1. **Measure exactly.** Every observed tick becomes a training example (about 280k). The evaluator copies
   the competition score: error averaged within episodes, then conditions, as a ratio to the reference.
2. **Find structure, with controls.** Single lags carry almost nothing, while specific *pairs* of lags
   carry a lot. Validation's true probabilities, lined up against each machine's state at the cutoff,
   revealed the rules:
   - The win chance depends on whether win/loss at lags (1,3), (2,5) or (4,7) **match**. Direction varies
     by machine.
   - **Drift** means each machine's rule flips once at its own tick; **shared change** means all machines
     flip on the same tick.
   - The 2×:5×:20× split is fixed per type.
   - An episode-wide hidden offset slowly wanders over time.
3. **Model the generator.** A Beta-Bernoulli posterior over every (pair, direction, flip tick)
   hypothesis, plus episode and machine log-odds offsets, plus the size mix.
4. **Check the choice.** It beats gradient boosting and an MLP (best 53.6). Blending, calibration and
   cross-validated tuning add nothing measurable, so the submitted settings are kept.

## Compliance with the challenge rules

- **Inputs:** predictions use only each machine's own history and its episode's histories, at ticks
  before the cutoff. They use no condition labels, test answers or outside data.
- **Rerun:** Python standard library only, no network, deterministic float64. About 8 s on 2 CPUs, far
  under the 30-minute limit. A fresh read-only rerun of the ZIP reproduces the CSV byte for byte (the
  limit is 1e-8).
- **Files:** the ZIP passes the kit's archive validator: `README.md` and `reproduce.json` at the root,
  `image: python`, 6 files, 13 kB. The CSV passes `check_submission.py`: 1,536 queries, exact header,
  rows sum to 1 within 2e-16.
- **Use of validation answers:** used only to compare ideas, as the guide allows. The checkpoint is fitted
  on train only.
