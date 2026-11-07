# Discovery: how the simulator works

Follow-up to Phase 1. These tests use validation's true probabilities at the cutoff to reverse-engineer
the simulator, then confirm each finding on train histories, and where possible on test histories.
`prototype_model.py` turns the findings into a model: **55.61 / 60 on validation (R = 0.079)**, against
30.0 for the reference. All of its hyperparameters come from the train split.

```sh
python analysis/discovery/prototype_model.py      # ~30 s
for f in analysis/discovery/t*.py; do python $f; done   # all tests, ~3 min; order matters (some cache .pkl files)
```

Intermediate files go to `analysis/artifacts/discovery/` (gitignored).

## The generative structure

**1. Two numbers per machine per tick.** The true 5-way distribution has exactly two free parameters
(t12): the win chance, P(payout > 0), which carries about 94% of the variance; and the 1× share of a win,
P(1× | win), which carries about 6%. Given a win that isn't 1×, the 2×:5×:20× split is **fixed per type**
(sd = 0 across all validation machines):

| Type | 1× share (train mean) | 2× : 5× : 20× among the rest (train) |
|---|---|---|
| steady | 0.617 | 0.790 : 0.205 : 0.004 |
| balanced | 0.352 | 0.691 : 0.264 : 0.045 |
| jackpot | 0.211 | 0.376 : 0.461 : 0.163 |

**2. Win chance follows a "matching pair" rule** (t01, t04–t07). Each machine has one lag pair (i, j). The
rule's state is whether the win/loss outcomes at t−i and t−j were equal. One side is favorable, the other
unfavorable.

- **Pairs:** (1,3), (2,5) and (4,7) cover about 94% of machines, roughly equally, in train, validation and
  test (t21, t22). The other 25 pairs are rare.
- **Direction:** match-favorable vs mismatch-favorable is 51/49, unrelated to type or signal.
- **Effect size:** P(win) is about 0.95 / 0.15 (steady), 0.83 / 0.05 (balanced) and 0.59 / 0.014 (jackpot)
  in the favorable / unfavorable states. Validation truth is bimodal to match.
- **Match definition:** win/loss equality. Matching on the exact payout fits worse for 99% of machines (t26).
- **No third lag** adds information within a state (t26), and no single lag matters on its own once the
  pair is known.

**3. Signal** shifts the log-odds of a win by about +0.35 per unit (−1, 0, +1), in both states and every
type (t06, t22). It doesn't touch the size mix.

**4. Conditions are about the rule's direction over time** (t18–t20):

| Condition | What happens |
|---|---|
| stable | No change (flip found in 1% of machines). |
| drift | **Every** machine's rule flips direction once, abruptly, each at its own tick (spread sd ≈ 14 ticks). |
| shared change | **Every** machine flips on the **same tick** (median spread 0 ticks). |
| short history | No flips. 64 observed ticks: 40 consecutive before the cutoff plus 24 sampled older ones, so about 32 ticks have all lags. |

- **Timing:** flips fall 40–104 ticks before the cutoff (median about 67), never in the last 40.
- **What flips:** the same pair and the same state rates; only the direction reverses. The aligned effect
  jumps from +0.33 to −0.33 within one 4-tick bin.
- **Test histories** show the same three regimes: exactly 48 episodes with no flips, and 96 split about
  evenly between independent and simultaneous flips (t21).

**5. Hidden state** (t17, t23, t24). After the rule, the remaining log-odds residual (sd ≈ 0.6) has three parts:

- an **episode-wide offset** that wanders over about 10–30 ticks: residuals correlate 0.53 with the rest of
  the episode, and windows 32 ticks apart correlate 0.28;
- a small, **constant per-machine offset**;
- tick-level noise.

**6. The 1× share** is a persistent per-machine level: it holds across halves of the history (corr 0.86,
t14), with no lag rule (t15), no dependence on state (t25) and no episode link. At the cutoff, though, it
carries extra tick-level noise: history explains only about corr 0.4–0.56 of the true value (t13, t16).
That noise is unpredictable, so the best forecast is the shrunk per-machine mean.

**What doesn't matter:** neighbors' specific payouts, beyond the episode offset (Phase 1 E6 and t24 show no
same-tick or lagged coupling); payout sizes at lags; signal for the size mix; exact payout equality.

## Model design

For each machine, at its cutoff:

1. **Rule posterior.** Enumerate hypotheses (pair, direction, flip tick or none): 28 pairs × 2 directions ×
   (1 + about 80 flip ticks). Score each with a Beta-Bernoulli marginal likelihood over the machine's own
   history. The favorable/unfavorable cells swap sides before the flip.
   - **Priors from train:** pair frequencies; P(flip) = 2/3 for long histories and 0 for short ones;
     flip-tick histogram; Beta rate priors per type × signal, with strengths from between-machine dispersion.
   - **Predictive** P(win) = posterior-weighted rate of the cell that the cutoff state falls in.
2. **Offsets** on the log-odds scale: add the episode offset (recency-weighted, shrunk mean of all 8
   machines' working residuals, half-life about 6) and the per-machine offset (shrunk mean of its own
   residuals).
3. **Size mix.** Set P(1× | win) to the machine's 1×-share among its wins, shrunk to the type mean
   (about 60 pseudo-wins). Split the rest 2×:5×:20× by the fixed type ratios.
4. **Output** p0 = 1 − pw, p1 = pw·s, and p2, p5, p20 = pw·(1 − s)·mix_type.

Ablations on validation:

| Variant | Points | R |
|---|---|---|
| Reference (pooled) | 30.00 | 1.000 |
| Best Phase 1 starter tweak (decay, half-life 16) | ~30.9 | 0.945 |
| Rule posterior + size mix | 55.33 | 0.084 |
| + episode and machine offsets | **55.61** | **0.079** |
| Ceiling: perfect win chance, our size mix | 57.05 | 0.052 |
| Ceiling: our win chance, perfect size mix | 58.04 | 0.034 |

Sharing the flip tick across an episode, or deciding "flip or not" at the episode level, adds nothing,
because each machine's own evidence is already decisive. Prior strengths and half-lives move the score
by less than ±0.1 points.

**For round two:** the hypothesis space already covers all 28 pairs, and every prior is re-estimated from
that round's train split, so a change in *which* pairs recur is handled by refitting. A new kind of rule
would not be. A quick check worth adding there: rerun t01 and t18 on the new train data, and widen the
hypothesis class (single lags, triples) if pairs no longer dominate.

**For the submission:** the runner is Python 3.13 with the standard library only. The math here is
Beta functions (`math.lgamma`) and prefix sums, so a pure-Python port is straightforward. Precompute the
train hyperparameters into a JSON file so `predict.py` only does inference.
