# House Edge — participant guide

**TAMU Datathon · November 7–8, 2026**

Use past spins to predict the chance of each payout: **0×, 1×, 2×, 5×, or 20×**. You get logs from simulated slot machines, build a model, and submit its predictions. Better forecasts earn more points.

Start with the quick start. Everything after that explains how to improve your model and turn it in.

[Quick start](#1-get-a-working-submission) · [The data](#2-understand-the-data) · [Your own model](#3-build-your-own-model) · [Scoring](#4-how-scoring-works) · [Experiments](#5-make-it-better) · [Second round](#6-the-adaptation-round) · [Submission](#7-turn-it-in) · [Troubleshooting](#8-common-questions-and-fixes)

## What you're predicting

Each test query asks about **one machine's next spin**, at the query's `cutoff` tick. Return five probabilities, in payout order `0, 1, 2, 5, 20`. They describe how uncertain you are; no separate confidence interval is required.

For example, `[0.60, 0.20, 0.12, 0.06, 0.02]` means a 60% chance of no payout and a 2% chance of a 20× payout. It does not mean you expect the next spin to land on all five outcomes.

Every spin wagers one chip. Payouts are **gross returns**: 0× loses the chip, 1× returns it, and 5× returns five chips, for a net gain of four. Your competition score measures forecast quality, not chips won.

You are building a forecasting model. You do not need to choose bets, operate an agent, or build a game. Any model family is welcome. A CPU and Python are enough to run the starter; neural networks and GPUs are optional.

## 1. Get a working submission

Download the organizer's **main participant kit**, unzip it, and open a terminal in the extracted folder. Use the released kit, not the organizer repository or a practice kit, for your event submission.

You need **Python 3.11 or newer**. The starter and checker need no extra packages. Check your version:

```sh
python --version
```

On some systems the command is `python3`, or `py` on Windows. Substitute that in the commands below if needed. Run commands from the kit folder, where `run_starter.py` lives.

```sh
python run_starter.py --out outputs
python check_submission.py outputs/submission.csv
python check_submission.py outputs/validation_predictions.csv --split validation
python build_entry.py --starter outputs --out entry.zip
```

The first command trains a baseline, compares six simple methods on validation data, and writes predictions. **It exports `pooled` by default; it does not automatically export whichever method scored best.** Choose a different method explicitly when you're ready.

The second command checks your test CSV against this kit's queries and prints its round and file hash. The third checks your validation CSV and shows its round points. A successful local check **does not upload the file, reveal its hidden-test score, or confirm acceptance**.

The fourth command packages the saved starter model and prediction code into `entry.zip`. Your two upload files are `outputs/submission.csv` and `entry.zip`. This helper is for `run_starter.py` outputs; custom models and the experiment runner need the custom-code packaging steps below.

Your output folder contains:

| File | What it's for |
| --- | --- |
| `submission.csv` | Predictions for the test queries; this is the forecast file to submit. |
| `validation_predictions.csv` | Predictions you can score locally. Do not submit this as your test file. |
| `baseline.json` | Saved baseline model. |
| `run.json` | Settings, comparisons, runtime, input/output hashes, and reproduction checks. |
| `validation_score.json` | Detailed validation results. |
| `comparison/` | Validation predictions from each baseline. |

Keep this first valid run as a fallback. Use a new output folder for each experiment so you don't overwrite it. `starter.ipynb` offers the same basic workflow if you already use Jupyter; a notebook interface is not required.

## 2. Understand the data

The kit contains `README.md` (this guide), `RULES.md`, `DATA.md`, `EXPERIMENTS.md`, a submission README template, the starter scripts, and a `public/` data folder.

| Folder | What you get | How to use it |
| --- | --- | --- |
| `public/train/` | Queries, histories, observed next-spin labels, and condition labels. | Fit your model and build training examples. |
| `public/validation/` | Queries, histories, observed labels, true probabilities, conditions, and reference forecasts. | Compare methods, choose settings, and check generalization. |
| `public/test/` | Queries and observed histories; no answers or condition labels. | Generate the CSV you turn in. |

Each split also includes `sample_submission.csv` to show the required shape. `public/metadata.json` identifies the data version, season, round, and maximum points. `SHA256.json` records the original kit files' hashes.

### The two input files

`queries.csv` contains:

| Column | Meaning |
| --- | --- |
| `query_id` | The ID to copy into your prediction CSV. |
| `episode_id` | A group of related machines observed at a common cutoff. |
| `machine_id` | The machine within the episode. Group histories by both episode and machine. |
| `machine_type` | An observed category: `steady`, `balanced`, or `jackpot`. |
| `signal` | An observed feature, either -1, 0, or 1. |
| `cutoff` | The tick you're predicting. Inputs must be from earlier ticks. |

`history.csv` contains `episode_id, machine_id, tick, payout`. `labels.csv` maps each training or validation `query_id` to its observed next payout, called `outcome`.

The payout order matters: the five output numbers are always for **0, 1, 2, 5, 20**. They are probabilities between 0 and 1, not percentages between 0 and 100.

### Use the past correctly

- For a query with cutoff 200, use ticks below 200. Another machine's outcome at tick 200 is also unavailable.
- If you create a training example for tick 150 from a longer history, use inputs from below 150. Tick 150 can be its label, but cannot be an input.
- A missing tick is unknown. Do not replace it with a zero payout. A lag of two ticks is not necessarily the second-most-recent observed row.
- Use related machines only within the correct episode. Their earlier outcomes may help.
- Condition labels are available for training and validation, not test. Your final predictor must run without them.

Train, validation, and test use separate episodes. If you make extra evaluation splits, keep whole episodes together; splitting individual rows can put related histories on both sides and make your results look better than they are.

See `DATA.md` for more about short histories, temporal patterns, and machine relationships.

## 3. Build your own model

The quickest change is another included baseline:

```sh
python run_starter.py --method recent --out recent-run
python check_submission.py recent-run/validation_predictions.csv --split validation
python check_submission.py recent-run/submission.csv
```

Available methods are `global`, `context`, `empirical`, `recent`, `pooled`, and `pooled_recent`. These are starting points, not restrictions.

For more control, save the following as `my_model.py` in the kit folder. It loads the public data, fits a model, and writes both validation and test predictions. Replace the model with your own approach when you're ready.

```python
from pathlib import Path
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, read_labels, write_submission

DATA = Path("public")
OUT = Path("custom-run")
if OUT.exists():
    raise SystemExit("Choose a new output folder before running again.")

train_queries, train_history = load_inputs(DATA / "train")
labels = read_labels(DATA / "train/labels.csv", train_queries)
model = FrequencyForecaster(method="pooled_recent", half_life=24)
model.fit(train_queries, train_history, labels)

for split, filename in (
    ("validation", "validation_predictions.csv"),
    ("test", "submission.csv"),
):
    queries, history = load_inputs(DATA / split)
    predictions = model.predict(queries, history)
    write_submission(OUT / filename, predictions)
```

Then run:

```sh
python my_model.py
python check_submission.py custom-run/validation_predictions.csv --split validation
python check_submission.py custom-run/submission.csv
```

The example's `predict()` returns a dictionary mapping each `query_id` to five probabilities in payout order. The writer checks values and preserves numeric precision. The checker then confirms that the exported file contains every required query exactly once.

You can use different libraries, interfaces, or file-writing code. What matters is the final CSV and a working way to recreate it. If you add dependencies, record their versions. If training uses randomness, save the seeds, settings, and model files needed to reproduce the submitted predictions.

## 4. How scoring works

**Main is worth 80 points. Adaptation is worth 20.** Final points are their sum. All points come from forecasts; there are no writeup or presentation points. Working code and eligibility checks are required for awards.

We compare your five probabilities with the simulator's true probabilities. This means one lucky jackpot does not decide the ranking. Some machine state is hidden, so exact probabilities generally cannot be recovered from the visible data.

The scoring process is:

1. Add the five squared differences between your probabilities and the true probabilities for each query.
2. Average those errors within each episode, then across episodes in each condition.
3. Divide each condition's error by the frozen pooled starter's error for that condition.
4. Average those four ratios. Call the result `R`.

```text
R = average across conditions of error / max(reference_error, 0.000000001)
round points = round maximum / (1 + R)
final points = main points + adaptation points
```

The four equally weighted conditions are **stable, short history, drift, and shared change**. Main and adaptation have separate frozen references.

| Example | Main points | Adaptation points | Total |
| --- | --- | --- | --- |
| Same errors as the pooled reference (`R = 1` in both rounds) | 40 | 10 | 50 |
| Half its normalized error (`R = 0.5` in both rounds) | 53.33 | 13.33 | 66.67 |
| Exact probabilities (`R = 0` in both rounds) | 80 | 20 | 100 |

These are examples, not expected scores. **Points are not accuracy percentages.** A score of 60 does not mean 60% of spins were guessed correctly.

Use the checker's validation points or `validation.contest.technical_points` in `run.json` for the event metric. Other diagnostic fields, including the older generic `score` field, are not the event ranking. Higher round points are better; lower error and loss ratio are better. Validation scores help you choose a model but do not guarantee your final result.

## 5. Make it better

Start with one clear idea and compare it with the baseline on the same data.

| Idea | A useful first experiment |
| --- | --- |
| Recent behavior matters more | Compare short and long history windows, or exponential weighting. |
| Machines behave differently | Mix each machine's estimate with a shared average when its history is short. |
| Order matters | Add earlier payouts and pairs of payouts at different tick lags. |
| Patterns change | Compare short and long memory; test a blend or a change detector. |
| Related machines help | Add summaries of other machines' earlier outcomes within the episode. |
| Models make different mistakes | Blend their five-probability forecasts, choosing weights on validation data. |

For a built-in comparison loop:

```sh
python run_experiments.py --out experiments-01
python check_submission.py experiments-01/submission.csv
```

It tries 12 small model settings, chooses using half the validation episodes in each condition, and checks the winner on the remaining half. It writes `submission.csv`, `validation_predictions.csv`, and `experiments.json`. Edit `starter_candidates()` in `slot_forecasting/experiments.py` to try your own candidates.

Treat that second half as a fresh check. Once you repeatedly tune against its results, it is no longer independent. Compare condition-level results as well as the total; a change can help drift while hurting steady machines. Keep a short experiment log for yourselves: change, score, runtime, and keep/revert. This is useful working material, not a graded writeup.

Small sequence networks, autoregressive models, or other approaches are options if they fit your time and compute budget. Architecture names do not earn points. Get a simple valid submission first and keep it available while experimenting. The [experiment guide](EXPERIMENTS.md) has more suggestions.

## 6. The adaptation round

Main submissions close before adaptation opens. Everyone receives a second kit at the same time and gets **90 minutes** to submit predictions for its new test queries.

The new kit includes training data, validation answers, and histories with changed patterns. You can reuse main-round data, code, and saved models; refit your current approach or try a different one. Adaptation measures the quality of the new forecasts directly. There is no bonus for having a weak main submission and then improving it.

1. Save your final main files and the Form confirmation.
2. Extract adaptation into a separate folder. Check `public/metadata.json` says `adaptation`.
3. Bring over your code and required saved models. Keep the new kit's `public/` folder intact.
4. Run a known-good approach early, then spend the remaining time testing changes.
5. Check the new CSV against the adaptation kit and submit it through the **adaptation** Form.

An unchanged main CSV will have the wrong query IDs. A missing or invalid adaptation submission receives zero adaptation points; a valid main submission is required for an overall ranking. Equal final totals share a rank, with the prize tie procedure announced before kickoff.

## 7. Turn it in

You submit through a **Google Form**, one per round. Organizers announce both links and the exact cutoff times. Your kit creates and checks the files; the Form is the only way to turn them in.

### Submit through the Form

1. Open the round's Form and sign in with your Google account. Uploading files to a Google Form requires sign-in.
2. Choose your team from the list. Pick the same team every time; that is how entries are matched.
3. Attach your test forecast CSV (at most 10 MB) and your code ZIP (at most 100 MB) as the two file uploads.
4. Select **Submit** and wait for the confirmation page. The upload must finish before the cutoff.
5. Keep your own copy of the exact two files you submitted. Google sends a copy of your responses if the Form has that option enabled.

You can submit again before the cutoff. **The latest submission for your team before the cutoff counts**, even if it is worse or turns out to be invalid; an earlier one is not used as a fallback. The Form records the time it received each submission, and it stops accepting responses at the cutoff. Uploaded files are copied into the organizers' Drive, so later changes to your own copies have no effect.

If the Form will not load or an upload fails, contact organizers before the cutoff with your team name and the problem. Do not assume late submissions will count.

### Package your own model

For custom models, prepare a code folder with `README.md`, `reproduce.json`, your prediction script, dependencies needed offline, and saved model/config files. Do not put the submitted forecast CSV in the code archive; organizers compare your code's output against the CSV you submitted.

At the root of that folder, `reproduce.json` must contain:

```json
{
  "image": "python",
  "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]
}
```

The command is an argument list, not a shell string. Make your script accept these paths: organizers run it from your unpacked ZIP with the round's public data at `/data` (for adaptation, main-round public data is also at `/main_data`), and it must write the final CSV to `/output/submission.csv`. It must not need network access or any file outside your ZIP and the public data.

Keep `"image": "python"`. If you use libraries beyond the Python standard library, list them with exact versions in a `requirements.txt` at the ZIP root; organizers install those versions when re-running your code. Check with organizers early if your model needs a GPU.

Package the folder with `python build_entry.py --source my-entry --out custom-entry.zip`. The ZIP must have `README.md` and `reproduce.json` directly at its root, with no enclosing folder. It may contain at most 5,000 entries, 100 MB compressed, and 512 MiB unpacked. Symlinks, encrypted files, unsafe paths, and duplicate paths are rejected.

Organizers re-run the code of teams in contention for prizes, on a CPU, with a 30-minute limit and 16 GiB of RAM. The recreated probabilities must match your submitted CSV to within `1e-8` per value. Save a trained model or checkpoint in the ZIP if retraining would give different numbers. Reproduction confirms your forecasts can be recreated; it does not replace data-use and eligibility review.

For each round, prepare:

| Item | Include |
| --- | --- |
| Team and round | Your team ID and `main` or `adaptation`. |
| Forecast CSV | The final `submission.csv` for that round's test queries. |
| Code/model | A code ZIP with `README.md`, `reproduce.json`, source, and any saved files needed for reproduction. |
| Setup README | Dependency versions, seeds/settings, data paths, and the exact command that recreates the CSV. |

Copy [SUBMISSION_README.md](SUBMISSION_README.md) and fill it in. A short setup README is enough; no long report or video is needed by this scoring process. Any separate event-wide submission requirements will come from the event organizers.

### CSV contract

```csv
query_id,p_0,p_1,p_2,p_5,p_20
example_query,0.60,0.20,0.12,0.06,0.02
```

The example ID is a placeholder. Your file must contain the actual IDs from the current round's `public/test/queries.csv`.

- Include every test query exactly once, and no extras. Row order does not matter.
- Use exactly these six column names; no index, outcome, confidence, or team column.
- All probabilities must be finite numbers between 0 and 1 and sum to 1 within `1e-6` per row.
- Save as UTF-8 CSV. Do not submit an Excel workbook renamed to `.csv`.
- Preserve precision rather than rounding every probability to a few decimal places.

### Before uploading

- [ ] This is the right round and released event kit, not practice data.
- [ ] The checker passes on the **exact CSV** I intend to upload.
- [ ] My code ZIP includes `README.md`, `reproduce.json`, dependencies, settings, and required model files.
- [ ] I recreated the forecasts from a fresh folder using my README, without relying on an old notebook session or my machine's absolute paths.
- [ ] Submitted inference runs locally without a paid or network API.
- [ ] I checked the announced cutoff and the right Form for this round.
- [ ] After submitting, I saw the Form's confirmation page and kept a copy of the exact files.

Your team's latest Form submission before the cutoff counts. Main files cannot be replaced once adaptation opens. If an upload fails or you need a correction, contact organizers before the cutoff instead of assuming late replacements will count.

### What happens after submission

After the cutoff, organizers take each team's latest submission, validate it, and score valid forecasts on private test answers. They combine main and adaptation points, then check the code and eligibility of prize contenders before awards. Final-test feedback is withheld while submissions are open.

The local checker verifies format and IDs; it does not submit anything, certify data-use compliance, or confirm eligibility. Your five probabilities already express uncertainty, so there is no separate uncertainty file to upload.

## 8. Common questions and fixes

| Problem or question | What to do |
| --- | --- |
| `python` is not found | Check whether `python3` or `py` works and verify Python 3.11+. |
| Script or module not found | Fully extract the kit, then run from its folder. Keep the supplied `slot_forecasting/` directory beside the scripts. |
| `Incomplete public dataset` or missing CSV | Check that `public/` was fully extracted. Use `--data "path/to/public"` if it is elsewhere. |
| `query_id mismatch` | Check the round and split. Validation predictions need `--split validation`; event uploads use test IDs. |
| Probabilities do not sum to one | Normalize each row, check the five-column order, and avoid aggressive rounding. Do not hide NaNs by silently replacing them. |
| Extra columns or invalid CSV | Remove dataframe indices and use an actual UTF-8 CSV export. With pandas, use `index=False`. |
| Experiment folder already exists | Choose a fresh `--out` folder so previous results stay available. |
| My model's points seem much lower than its `score` field | Read round `technical_points`; the older raw diagnostic score uses a different scale. |
| Can I use coding assistants or automated tuning? | Yes. You remain responsible for a working, reproducible submission and the data-use rules. |
| Do I need RL, a neural network, or a GPU? | No. Any model that produces valid probabilities can compete. |
| Can I use outside data? | Use organizer-released data, as specified in `RULES.md`. Private test answers and organizer files are off limits. |
| Can I calculate my final score locally? | No. Only validation answers are public; a test format check is not a hidden evaluation. |
| Does the slot-machine demo affect my score? | No. It illustrates forecasts using fixed practice odds. Your competition entry is the CSV. |

If you're stuck, send organizers the command, Python version, round, and exact error text. Keep your team's code/model private unless you intend to share it. Check `RULES.md` for the scoring and eligibility rules and the official announcement for event logistics.
