# House rules

House Edge / Slot Machine Forecasting / TAMU Datathon 2026

Predict the next spin's probabilities of returning **0×, 1×, 2×, 5×, or 20×** a one-chip wager. The machines and gameplay logs are simulated for this challenge.

## Two rounds

**Main floor — 80 points.** Learn from the main training and validation data, then submit forecasts for its test queries.

**After the change — 20 points.** Main submissions close before the second kit opens. The new kit contains training data, validation answers, and test histories with changed patterns. Everyone gets **90 minutes from the common release time** to update their forecasts. You may reuse your code and main-round data, refit the same model, or try another approach.

Both rounds score forecast quality directly. The adaptation score is independent of your main score. Exact cutoffs, the Google Form link for each round, and the outage/correction process will be announced before kickoff.

## How points work

For each of the four conditions, average squared probability error within each episode, then average the episodes. Call this error `E[c]`. Calculate `B[c]` the same way for the frozen pooled starter fitted on that round's training data.

```
R = mean over four conditions of E[c] / max(B[c], 1e-9)
main points = 80 / (1 + R_main)
adaptation points = 20 / (1 + R_adaptation)
final points = main points + adaptation points
```

The conditions are stable, short history, drift, and shared change. Each counts equally within its round. Main and adaptation have separate frozen references. The pooled starter normally scores 40 + 10 = 50; exact knowledge of all probabilities scores 100. These are competition points, not accuracy percentages.

Forecasts are compared with the simulator's true conditional probabilities, so sampled jackpots do not decide the ranking. Some machine state is hidden, so perfect knowledge is generally unavailable. Your probabilities should reflect what the observed history can tell you.

All points come from forecasts. Code reproduction and eligibility are required for awards. Equal totals share a rank; organizers will announce the prize tie procedure before kickoff.

## At the table

- Use organizer-released data. Main data may be reused during adaptation. Test answers and private organizer files are off limits.
- Use observations strictly before the prediction tick, including when making training examples from earlier history. Missing ticks are unobserved, not zero payouts.
- Earlier outcomes can interact at lags up to eight ticks. Machines of the same type can respond differently or oppositely; both win chances and winning amounts can change. Other machines' earlier outcomes may help. Condition labels are supplied for training/validation only.
- Use any model family, model-selection tools, or coding assistants. You are responsible for a working, reproducible submission.
- Submitted inference must run locally without a paid or network API. Organizers re-run prize contenders' code on a CPU with a 30-minute limit and 16 GiB of RAM.
- All teams receive the same files and deadlines. Final-test feedback is withheld while submissions are open. Public validation is available for model selection.

## Submit your forecast

For each round, submit one CSV with exactly these columns:

```csv
query_id,p_0,p_1,p_2,p_5,p_20
```

Include every test query once. Probabilities must be finite, between zero and one, and sum to one within `1e-6`. Include your code/model, dependency versions, seeds/settings, and one command that recreates the CSV. A short setup README is enough.

The released kit includes `check_submission.py`, `build_entry.py`, and `SUBMISSION_README.md`. Submit the CSV (at most 10 MB) and code ZIP (at most 100 MB) through the round's Google Form, choosing your team from its list. Code ZIPs require `README.md` and `reproduce.json` at their root; the participant guide explains the runtime contract. A local check does not submit anything.

Your team's latest Form submission before each cutoff counts. Main files cannot be replaced after adaptation opens. A missing or invalid adaptation entry earns zero adaptation points; a valid main entry is required for overall ranking. Organizers freeze one submission per team per round and verify prize contenders' code before awards.

The latest submission stays selected even if it later turns out to be invalid or its code does not reproduce; there is no automatic fallback to an older entry. Re-running code may happen after the cutoff, using the files already submitted. If your model needs libraries beyond the Python standard library, pin their versions in a `requirements.txt` in your code ZIP.
