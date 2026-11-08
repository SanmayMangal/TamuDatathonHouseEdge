# What changed? Diagnostics for round `adaptation` (season 5fffdc4d0c700bbf61c3f911)

## Kit
- train: 96 episodes, 768 queries, machines per episode [8], cutoffs 192–256, types {'jackpot': 260, 'steady': 255, 'balanced': 253}, signals [-1, 0, 1], conditions {'drift': 24, 'shared_change': 24, 'short_history': 24, 'stable': 24}
- validation: 96 episodes, 768 queries, machines per episode [8], cutoffs 192–256, types {'balanced': 295, 'jackpot': 266, 'steady': 207}, signals [-1, 0, 1], conditions {'drift': 24, 'shared_change': 24, 'short_history': 24, 'stable': 24}
- test: 192 episodes, 1536 queries, machines per episode [8], cutoffs 192–256, types {'jackpot': 552, 'steady': 501, 'balanced': 483}, signals [-1, 0, 1]

## Learned structure (fit on train)
- Encoding: **win** (log evidence {'win': -45076.656487192966, 'profit': -73472.72143146639})
- Rule families: {single: 0.005, pair: 0.959, triple: 0.036}
- Top rules (prior): (1, 4) 0.334, (3, 8) 0.321, (2, 6) 0.287, (1, 4, 8) 0.001, (1, 3, 7) 0.001, (4, 6, 8) 0.001, (2, 6, 8) 0.001, (4, 5, 6) 0.001
- Rules differ from the main round's [(1, 3), (2, 5), (4, 7)]: now [(1, 4), (2, 6), (3, 8)]
- MAP rule-family counts on train machines: {'pair': 768}
- P(flip): long 0.674, short 0.033

## Flips by condition (MAP on train)
| condition     |   machines |   frac_flipped |   flip_d_median |   flip_d_min |   flip_d_max |
|:--------------|-----------:|---------------:|----------------:|-------------:|-------------:|
| drift         |        192 |              1 |              67 |           34 |           91 |
| shared_change |        192 |              1 |              78 |           52 |          100 |
| short_history |        192 |              0 |             nan |          nan |          nan |
| stable        |        192 |              0 |             nan |          nan |          nan |

Within-episode spread of flip ticks (median over episodes with ≥3 flips; ~0 means simultaneous):

| condition     |   size |   std |
|:--------------|-------:|------:|
| drift         |      8 | 13.04 |
| shared_change |      8 |  1.11 |

## State win rates (favorable / unfavorable) by type and signal
| type     |   signal |   p_fav |   p_unfav |
|:---------|---------:|--------:|----------:|
| balanced |       -1 |   0.778 |     0.032 |
| balanced |        0 |   0.839 |     0.053 |
| balanced |        1 |   0.874 |     0.071 |
| jackpot  |       -1 |   0.499 |     0.01  |
| jackpot  |        0 |   0.587 |     0.017 |
| jackpot  |        1 |   0.664 |     0.022 |
| steady   |       -1 |   0.924 |     0.112 |
| steady   |        0 |   0.95  |     0.159 |
| steady   |        1 |   0.958 |     0.2   |

Beta strengths: {'balanced': [47.371104378394726, 72.23830983567635], 'jackpot': [19.406079528714955, 152.3400456791214], 'steady': [104.78140568231512, 42.26002920212119]}  |  1x share: {'steady': 0.6128182946552827, 'balanced': 0.34326059050064184, 'jackpot': 0.20404926292218697}  |  high mix: {'steady': [0.806, 0.19, 0.004], 'balanced': [0.692, 0.263, 0.045], 'jackpot': [0.379, 0.466, 0.155]}  |  high-mix strength: {'steady': 1000000.0, 'balanced': 1000000.0, 'jackpot': 1000000.0}

## Shape of the true odds (validation)
- balanced: variance in first two components 1.0000 (main round: 1.000); sd of 2x:5x:20x split [0.0, 0.0, 0.0] (main: 0); win chance quantiles [0.024, 0.042, 0.137, 0.839, 0.909]; share of win chances in 0.3–0.7: 0.06
- jackpot: variance in first two components 1.0000 (main round: 1.000); sd of 2x:5x:20x split [0.0, 0.0, 0.0] (main: 0); win chance quantiles [0.005, 0.009, 0.019, 0.522, 0.71]; share of win chances in 0.3–0.7: 0.23
- steady: variance in first two components 1.0000 (main round: 1.000); sd of 2x:5x:20x split [0.0, 0.0, 0.0] (main: 0); win chance quantiles [0.101, 0.158, 0.917, 0.952, 0.975]; share of win chances in 0.3–0.7: 0.04

## Validation scores (exact)
- Generalized model (fit on train): R **0.0705** (18.68 / 20.0) | drift 0.064, shared_change 0.065, short_history 0.081, stable 0.072
- Main-round submission, unchanged: R 0.0937 (18.29 / 20.0) | drift 0.062, shared_change 0.064, short_history 0.161, stable 0.088
- Reference: R 1.0000 (10.00 / 20.0)

## Residuals (true vs predicted win log-odds)
- sd 0.544 (main round ~0.6); by condition {'drift': 0.57, 'shared_change': 0.5, 'short_history': 0.545, 'stable': 0.526}
- correlation with rest-of-episode mean residual: 0.385 (main round: 0.47; stronger offsets did not help there, so a similar value is expected)
- |residual| > 2: 0.000 of queries; by condition {'drift': 0.0, 'shared_change': 0.0, 'short_history': 0.0, 'stable': 0.0}

## Alerts
- none: all main-round assumptions hold
