# analysis/

Phase 0 tooling: a shared example table and a competition-style evaluator.

```sh
python analysis/build_table.py     # ~10 s -> analysis/artifacts/examples.pkl (gitignored, ~280 MB)
python analysis/verify_table.py    # sanity checks; add --comparison <run>/comparison to cross-check the kit's checker
```

Both scripts accept `--data path/to/public`, so they run unchanged on the round-two kit.
Requires numpy and pandas (exploration only; nothing here goes into the submission runner yet).

## The table

One row per observed `(split, episode, machine, tick)` in `history.csv`, plus one **query row** per machine at its cutoff.
Every feature on a row at tick `t` comes from ticks strictly below `t`, so history rows are valid
"predict tick t from the past" training examples, and query rows are the real predictions.

| Rows | train | validation | test |
|---|---|---|---|
| history | 139,944 | 141,392 | 282,664 |
| query | 768 | 768 | 1,536 |

| Column(s) | Meaning |
|---|---|
| `split`, `episode_id`, `machine_id` | Identifiers. |
| `machine_slot` | 0–7, the machine's position in its episode, sorted by `machine_id`. |
| `tick`, `cutoff`, `ticks_to_cutoff` | Row tick, episode cutoff, and `cutoff - tick` (0 on query rows). |
| `is_query` | True for the cutoff row. |
| `outcome` | Payout at `tick`. On query rows: the train/validation label; NaN for test. |
| `machine_type`, `signal` | From `queries.csv`. |
| `condition` | Train/validation only (NaN for test); **never use it as a feature**. |
| `short_history` | True if the episode's histories have gaps. Available at test time; matches the short-history condition. |
| `fold` | 0–4: episode-level folds stratified by condition, for train and validation. −1 for test. |
| `n_obs_before`, `gap_since_prev` | Own observations before `tick`; ticks since the previous observation. |
| `own_lag1` … `own_lag12` | Own payout at `tick − k`. NaN if unobserved (never 0). Lags 9–12 are a no-signal control. |
| `nb{s}_type`, `nb{s}_signal` | Type and signal of neighbor `s` (1–7: the other machines in slot order). |
| `nb{s}_lag1` … `nb{s}_lag8` | Neighbor `s`'s payout at `tick − k`. |
| `ref_p_0` … `ref_p_20` | The frozen pooled reference (strength 20, fitted on train) as of `tick`. Matches `reference.csv` exactly on validation queries. |
| `q_0` … `q_20` | True probabilities. Validation query rows only. |

## The evaluator (`evaluate.py`)

`score(rows, pred, against=...)` copies the kit's contest score: mean error per episode, then per condition,
then the ratio against the `ref_p_*` columns. R is the mean of the four ratios; points = 60 / (1 + R).

- `against="target"`: validation query rows vs `q_*`. This is the exact number; it matches `check_submission.py` to 1e-6 for all six starter methods.
- `against="outcome"`: any labeled rows vs the one-hot outcome. Use it to rank models on the ~280k history rows.

**Caveat for outcome scoring.** Expected outcome error = true error + noise, and the noise term (1 − Σq² ≈ 0.38)
is about **twice** the reference's true error (≈ 0.18–0.21). Rankings within a condition are preserved, but
raw ratios are squeezed toward 1. For example, a uniform forecast scores R = 1.49 on raw outcomes vs 2.20 exactly.
`noise=noise_by_condition(df)` subtracts a per-condition estimate taken from validation targets. It over-corrects
somewhat (2.68 for uniform), because noise at history ticks isn't identical to noise at cutoffs. Use outcome scores
to compare candidates, and confirm with `against="target"` before believing a ratio.

`folds(df, 5)` yields train/test masks over train+validation rows, keeping whole episodes together.

## Phase 1 experiments (`phase1/`, `run_phase1.py`)

```sh
python analysis/build_table.py              # once
python analysis/run_phase1.py               # all of E1-E9 on the full train + validation data
python analysis/run_phase1.py --only E2,E7  # a subset
python analysis/run_phase1.py --smoke       # 2 episodes per condition, only to check the code runs
```

Output: `analysis/results/full/report.md` (key tables per experiment), one CSV per table under
`analysis/results/full/E*/`, and `timings.json`.

**Compute.** CPU only, pandas + numpy, single process; no GPU needed. The smoke run (24k rows, 1/12 of
the data) takes about 9 s in total, so a full run should take roughly 2–4 minutes and a few GB of RAM.
E4 (lag pairs) and E6 (neighbors) are the slowest.

**Method.** Most experiments compare a count model that conditions on a feature against the machine's
own overall rates, on the same rows, both scored **leave-one-out** (each row's outcome is removed from the
counts before it is predicted). Gains are Brier improvements on observed outcomes; `dR` divides by the
reference's true error per condition, so it approximates how much R would fall. Each test has a control:
lags 9–12, and a **null** that shuffles the feature within each machine (same mix of values, timing destroyed).
A gain only counts if it clears its null.

| | Question | Main tables |
|---|---|---|
| E1 | How do type and signal shift the odds? Do machines differ beyond type × signal? | `by_type_signal`, `overdispersion` |
| E2 | Which single lags (1–12) carry signal, per machine vs shared per type? | `summary`, `machine_dR_by_condition` |
| E3 | Do same-type machines react in opposite directions? Does signal set the direction? | `by_type_lag`, `by_type_signal_lag`, `strongest_lag` |
| E4 | Do lag pairs carry signal beyond each lag alone (interactions)? | `top_pairs`, `interaction_by_condition` |
| E5 | Which encoding of a lagged payout matters, and does it move win chance or payout size? | `summary` |
| E6 | Do neighbors' earlier payouts help, via specific pairs or the whole episode? | `pairwise_by_lag`, `aggregate`, `coupling` |
| E7 | Best memory length per condition; change points, their timing across machines, trends, synchrony. | `memory`, `memory_exact`, `changes`, `timing` |
| E8 | Short histories: layout, and how much to trust 64 ticks vs the prior. | `layout`, `estimators` |
| E9 | Where the reference's error is concentrated, how peaked the true odds are, label-only ceilings. | `by_group`, `truth_shape`, `ceilings` |

Experiments use train + validation history rows only; test is never touched. E7, E8 and E9 also score
exactly against validation targets, so those numbers reuse the validation answers: treat them as
diagnostics, not as a held-out check.
