# Cross-validated fine-tuning (result: keep the submitted model)

`cv_tune.py` (~1.5 min) splits the 96 validation episodes into 4 folds, stratified by condition. Every
score is out-of-fold, and settings are chosen on the other 3 folds (nested selection).

| Variant | Out-of-fold points | R | Folds better than current |
|---|---|---|---|
| A. Current (train priors, current settings) | **55.605** | 0.0790 | – |
| B. Priors from train + other validation folds | 55.604 | 0.0791 | 2/4 (differences ≤ 0.0007) |
| C. Settings tuned over 324 combinations | 55.505 | 0.0810 | 1/4 |
| D. Both | 55.502 | 0.0810 | 1/4 |

- **More data for the priors changes nothing.** The priors are already well determined by 768 train machines.
- **Tuning overfits.** The best settings on 3 folds lose on the held-out fold. The in-sample landscape is
  flat: the best of 324 combinations is R 0.0782 against 0.0789 for the current settings (rank 24), a
  difference of about 0.05 points, which is well below the score's sampling noise (±0.23 points).

Decision: keep the submitted model and settings. For round two, priors can be fitted from train + validation
without loss.
