# Round two (adaptation): runbook

This folder is separate from the main-round entry. `submission/` and `deliverables/main/` are unchanged.

| File | Purpose |
|---|---|
| `fit.py` | Learns the structure from a kit's labeled histories: rule family (single lag, pair or three-lag parity, lags ≤ 8), lag encoding (win or profit), flip probability and timing anywhere in the history (EM), state rates, 1× share and high-prize mix. numpy/pandas/scipy; about 4 min for train and 7 min for train + validation. |
| `predict.py` | **Standard-library** forecaster for whatever `fit.py` learned. It becomes the round-two entry. |
| `diagnose.py` | "What changed?" report. Every main-round assumption is checked; `ALERT` lines flag broken ones. |
| `run_round2.py` | One command: fit on train → score validation → diagnose → fit on train + validation → forecast test → kit checker → ZIP → fresh read-only rerun, compared byte for byte. |
| `synth_kit.py` | Builds a synthetic kit with different rules, for rehearsal only. |

## At 9 PM

```sh
# 1. Unpack the new kit into its own folder (outside this repo or under a gitignored path)
# 2. Confirm public/metadata.json says "round": "adaptation" and "max_points": 40
python round2/run_round2.py --data PATH/TO/NEW_KIT/public --out runs/round2
# 3. Read runs/round2/report.md, Alerts section first
# 4. If there are no alerts, or the validation score looks right, upload runs/round2/submission.csv and
#    runs/round2/house-edge-adaptation.zip (the Adaptation Google Form) as a safety entry
```

Total time is about 12 minutes. A missing or invalid round-two entry scores 0, so upload the safety
entry first and improve afterwards.

## Reading the report

| Alert | Meaning | Action |
|---|---|---|
| none | Same kind of structure; the refit priors already adapt to new pairs, rates and flip timing | Done. Optionally check that the validation score matches main-round quality (R ≈ 0.08). |
| rules no longer mostly pairs | Singles or triples now matter. The model already covers them; the report shows which | Usually nothing. The rehearsal on a synthetic kit with 30% single and 30% triple rules scored R 0.082. |
| encoding not win/loss | Lags act on "profit" (2×+) rather than on wins | Already handled by the fitter's choice. Check that the validation score is good. |
| stable episodes show flips | Change behaves differently | Look at the flip table. If flips are multiple, add a two-flip hypothesis. |
| true odds need more than 2 parameters | The size mix depends on state | Add a state-dependent 1× share: two Beta cells, like the win rule. |
| 2×:5×:20× split varies | The high-prize mix varies by machine or tick | The learned `hi_strength` already shrinks per machine. Check whether it fell from 1e6. |
| model R > 0.3 | Structure not covered | Use residuals and the discovery tests (`analysis/discovery/t*.py`, pointed at the new kit) to reverse-engineer it, as in the main round. |

Validation scores in the report come from a **train-only** fit, so they're honest. The packaged
checkpoint is refit on train + validation, which cross-validation showed costs nothing.

## Rehearsals (Saturday evening)

| Kit | Alerts | This model | Main-round entry unchanged | Reference |
|---|---|---|---|---|
| Main kit, as a stand-in | none | 55.56 / 60 (R 0.080) | 55.61 / 60 | 30 / 60 |
| Synthetic changed kit: rules (5,), (3,6), (1,4,7), (2,8), (2,3,6); flips 10–60 before cutoff | "not mostly pairs" | **36.96 / 40** (R 0.082) | 20.95 / 40 (R 0.91) | 20 / 40 |

On the synthetic kit, the fitter recovered every planted rule with roughly the right weights, the flip
windows, and simultaneous vs independent flips. Both rehearsals produced a valid CSV and a ZIP that
passed the kit checks and reproduced byte for byte.
