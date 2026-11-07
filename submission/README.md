# House Edge submission: Bayesian matching-pair rule model

## Entry

- Team: Card Sharks
- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911 (`public/metadata.json`)
- Forecast file: `submission.csv` (uploaded separately; not in this ZIP)
- Forecast SHA-256: 49bd7214e47df343526f1f4fe7dd03a5398cf720a8d436baf557f9a2e6d57d1a
- Code version: branch `submission-main` of the team repository
- Environment: `python` (`reproduce.json`). **Python standard library only**; no `requirements.txt`.

## Setup

- Python: 3.13 (tested 3.13.18). Uses only `argparse`, `bisect`, `csv`, `json`, `math`, `pathlib`.
- OS tested: Linux x86-64.
- Hardware: 1 CPU core is enough. About 8 s and under 300 MB RAM for the test split.
- Data: the round's public folder, mounted at `/data` (reads `/data/test/queries.csv` and `/data/test/history.csv`).

## Recreate the forecast

Run from the ZIP root (`/submission`):

```sh
python predict.py --data /data --out /output/submission.csv
```

- Output: `/output/submission.csv`, with columns `query_id,p_0,p_1,p_2,p_5,p_20`.
- Required model file: `hyperparameters.json` (the saved checkpoint, loaded by `predict.py`).
- Seeds and settings: no randomness. Settings are stored in `hyperparameters.json` under `settings`.
  Arithmetic is float64 throughout and deterministic.
- What the command does: it loads the checkpoint and runs inference only. For each query it uses that
  machine's history and its episode's histories, at ticks before the cutoff only.
- Measured runtime: 8.1 s on 2 vCPUs (Linux, Python 3.13.18).
- Training command, which is separate and not needed for reproduction:
  `python training/fit_hyperparameters.py --data path/to/public --out hyperparameters.json`.
  It reads only `public/train` and needs numpy 2.5.3, pandas 3.0.5 and scipy 1.18.1; it takes about 20 s.

## Model

The simulator gives each machine one lag pair (i, j) with i < j ≤ 8. The rule's state at tick t is whether
the machine's win/loss results at t−i and t−j were equal. One side of the rule has a high win chance, the
other a low one. In drift and shared-change episodes the rule's direction flips once.

`predict.py`:

1. **Posterior over hypotheses.** It enumerates (pair, direction, flip tick or none) and scores each
   hypothesis with Beta-Bernoulli marginal likelihoods. The priors come from the train split: pair
   frequencies, flip probability and flip-timing histogram, and state win rates per type × signal with
   Beta strengths.
2. **Predictive win chance** at the cutoff, then two log-odds offsets: an episode-wide, recency-weighted
   offset (half-life 6) and a per-machine offset. Both are shrunk residual means under each machine's most
   likely hypothesis.
3. **Size mix.** P(1× | win) is the machine's 1× share shrunk to its type mean (60 pseudo-wins). The
   2×:5×:20× split is fixed per type from train.

Validation (scored with the kit's checker, not used to fit): **55.61 / 60** points.

## Local check

```sh
python check_submission.py submission.csv --data path/to/public
```

- Checker result: `VALID FORMAT: 1536 queries | main | test`
- Fresh-folder reproduction checked: yes. The ZIP was extracted to a read-only folder and run with this
  command against a copy of the public data. Output matched `submission.csv` exactly (max difference 0).
