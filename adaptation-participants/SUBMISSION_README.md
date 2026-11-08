# House Edge — submission README template

Copy this into your submission as `README.md`. Replace the bracketed fields with your actual details. Keep it short; this is a setup guide, not a graded writeup.

## Entry

- Team ID: [team ID]
- Round: [main or adaptation]
- Season ID: [copy `season_id` from this round's `public/metadata.json`]
- Forecast file: [path to your submitted CSV]
- Forecast SHA-256: [printed by `check_submission.py`]
- Code version: [commit ID or archive/version name]
- Libraries: ["Python standard library only", or `requirements.txt` with exact versions]

## Setup

- Python/runtime version: [version]
- OS tested: [OS]
- Dependencies: [dependency file and exact install command; or "Python standard library only"]
- Required hardware: [CPU/RAM; GPU and version if used]
- Put the released data at: [path, for example `public/`]

Organizers run your `reproduce.json` command from the unpacked ZIP with the current round's public data at `/data` (adaptation also has main-round public data at `/main_data`). It must write `/output/submission.csv` without network access. Include this README and `reproduce.json` at the code ZIP root.

## Recreate the forecast

Run from: [folder]

```sh
[exact command that recreates the submitted test CSV]
```

- Output file: [path]
- Required model/config files: [paths; say "none" if the command trains everything]
- Random seeds and settings: [values or config path]
- What the command does: [loads a saved model / trains and predicts]
- Measured runtime and hardware: [time and machine used]
- Training command, if separate: [exact command and required inputs]
- Main-round data/model reused in adaptation, if any: [what and where]

If reruns are not deterministic, describe the source of variation and how the submitted predictions were produced. Include the saved checkpoint/settings needed to reproduce them.

## Local check

```sh
python check_submission.py [path/to/submission.csv] --data [path/to/current-round/public]
```

- Checker result: [valid / unresolved issue]
- Fresh-folder reproduction checked: [yes/no, plus any limitations]

Local validation does not submit anything. Submit the forecast CSV and code ZIP through the round's Google Form, and wait for its confirmation page. Prize contenders' code is re-run on a CPU with a 30-minute limit and 16 GiB of RAM.
