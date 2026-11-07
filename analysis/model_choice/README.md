# Which model? Bayesian rule model vs learned models

The kit allows any model, neural networks included (README §1, §3, FAQ). This folder tests the strongest
learned alternatives we can run here, and weighs them against the rules for reproduction and runtime.

```sh
python analysis/model_choice/benchmark.py   # ~1.5 min
python analysis/model_choice/hybrid.py      # ~1 min
```

## Results (exact score on validation; learned models trained on train-split history ticks)

| Model | Points / 60 | R |
|---|---|---|
| Reference (pooled) | 30.00 | 1.000 |
| Gradient boosting, generic features (type, signal, lags 1–8, recency rates, 28 pair-match flags) | 38.97 | 0.540 |
| Gradient boosting + hand-built per-machine rule-evidence features (162 features) | 53.55 | 0.121 |
| MLP 64×64 + the same evidence features | 49.81 | 0.205 |
| **Bayesian rule model** (`discovery/prototype_model.py`) | **55.61** | **0.079** |
| Bayesian + 10–15% GBM blend (weights chosen out-of-fold) | 55.64 | 0.078 |
| Bayesian + per-type log-odds calibration (out-of-fold) | 55.56 | 0.080 |

**Why the learned models fall short.** The hard part is inference about a hidden quantity: each machine's
pair, direction and flip tick must be worked out from that machine's own past. Train has only 768
machines, so 768 independent draws of that hidden rule, even though it has 130k ticks.

- **A generic learner** never sees which pair applies, so it can't get past about 39 points.
- **Given evidence features,** it gets close, but those features *are* the Bayesian sufficient statistics
  approximated by hand. The tree still has to learn the posterior arithmetic (choosing among 28 pairs,
  reversing after a flip) from limited data.
- **The Bayesian model** does that arithmetic exactly, and its outputs are already calibrated
  (fitted slope ≈ 1.0, shift ≈ 0).

**A sequence model (GRU or transformer)** wasn't testable here: torch isn't installed and package installs
are blocked. It would face the same 768-sequence limit. The only way around that is training on data
simulated from the generator we discovered, and then the network is approximating the posterior that the
Bayesian model already computes exactly. It can't beat it, except on structure the generator misses.

## Against the rules

| Requirement (README §7, RULES) | Bayesian, pure Python | Learned model (GBM or NN) |
|---|---|---|
| Rerun on 2 CPUs, 30 min, no GPU guaranteed | Inference only, est. minutes | GBM: fine. NN: CPU inference fine, retraining slow |
| Exact versions in `requirements.txt`, installed by organizers | None: standard library | sklearn or torch pinned; another failure point |
| Recreated probabilities within 1e-8 | Deterministic float64 math | GBM: deterministic. NN in float32: results can differ across CPUs or BLAS by about 1e-7 |
| Round two: "patterns have changed", 14 h, overnight | Refit priors from the new train split in about 1 min | Retrain and retune; the learned patterns may not carry over |
| Main round: about 6 h left | Port of a working prototype | New pipeline, packaging and checks |

## Recommendation

Use the **Bayesian rule model as the submission**, in pure Python (`math.lgamma` plus prefix sums), with
hyperparameters fitted from the train split and stored in a JSON file next to `predict.py`. Keep
`benchmark.py` and `hybrid.py` as checks. If round two brings structure that the hypothesis class
doesn't cover, the evidence-feature gradient booster (53.6) is the fallback, because it needs no exact
rule. A small blend weight is optional; it's worth about 0.03 points.
