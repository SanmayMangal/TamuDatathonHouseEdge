"""Phase 1 experiments: map the patterns in the data before building models."""
from . import (e1_base_rates, e2_single_lags, e3_effect_direction, e4_lag_pairs, e5_encodings,
               e6_neighbors, e7_nonstationarity, e8_short_history, e9_error_breakdown)

EXPERIMENTS = {
    "E1": e1_base_rates, "E2": e2_single_lags, "E3": e3_effect_direction,
    "E4": e4_lag_pairs, "E5": e5_encodings, "E6": e6_neighbors,
    "E7": e7_nonstationarity, "E8": e8_short_history, "E9": e9_error_breakdown,
}
