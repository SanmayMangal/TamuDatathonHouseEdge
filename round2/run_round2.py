"""One command for a new kit: fit, diagnose, score, package, and verify.

    python round2/run_round2.py --data NEWKIT/public --out runs/r2 [--team "Card Sharks"]

Steps (all outputs under --out):
  1. fit.py on train only            -> hp_train.json (+ .map.csv)
  2. validation forecasts            -> val_v2.csv (this model), val_main.csv (main-round submission, unchanged)
  3. diagnostics                     -> report.md  (read the ALERT lines first)
  4. fit.py on train + validation    -> hp_final.json
  5. package/                        -> predict.py, hyperparameters.json, reproduce.json, README.md, training/fit.py
  6. test forecast + kit checker     -> submission.csv
  7. ZIP via the kit's build_entry.py, then a fresh read-only rerun of the ZIP compared byte for byte
Needs numpy, pandas, scipy for fitting/diagnostics; the packaged predict.py is standard library only.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PY = sys.executable


def run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"command failed: {cmd}")
    return r.stdout


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path, help="new kit's public folder")
    ap.add_argument("--kit", type=Path, help="kit root with check_submission.py/build_entry.py (default: parent of --data)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--team", default="Card Sharks")
    ap.add_argument("--final-splits", default="train,validation")
    ap.add_argument("--skip-fit", action="store_true", help="reuse hp_train.json / hp_final.json in --out")
    args = ap.parse_args()
    data, out = args.data.resolve(), args.out.resolve()
    kit = (args.kit or data.parent).resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data / "metadata.json").read_text())
    rnd = meta.get("round", "unknown")
    print(f"kit round={rnd} season={meta.get('season_id')} max_points={meta.get('max_points')}")
    t0 = time.time()

    # 1-2. train-only fit, validation forecasts
    if not args.skip_fit:
        print(run([PY, "-W", "ignore", HERE / "fit.py", "--data", data, "--splits", "train", "--out", out / "hp_train.json"])[-600:])
    run([PY, HERE / "predict.py", "--data", data, "--split", "validation", "--hyperparameters", out / "hp_train.json", "--out", out / "val_v2.csv"])
    main_pred = REPO / "submission/predict.py"
    have_main = main_pred.exists()
    if have_main:
        run([PY, main_pred, "--data", data, "--split", "validation", "--out", out / "val_main.csv"])
    check_val = run([PY, "check_submission.py", out / "val_v2.csv", "--split", "validation", "--data", data], cwd=kit)
    print(check_val)

    # 3. diagnostics
    cmd = [PY, "-W", "ignore", HERE / "diagnose.py", "--data", data, "--hp", out / "hp_train.json",
           "--val-pred", out / "val_v2.csv", "--out", out / "report.md"]
    if have_main:
        cmd += ["--main-val-pred", out / "val_main.csv"]
    if meta.get("max_points"):
        cmd += ["--max-points", meta["max_points"]]
    print(run(cmd))

    # 4. final fit
    if not args.skip_fit:
        print(run([PY, "-W", "ignore", HERE / "fit.py", "--data", data, "--splits", args.final_splits, "--out", out / "hp_final.json"])[-600:])

    # 5-6. package and test forecast
    pkg = out / "package"
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "training").mkdir(parents=True)
    shutil.copy(HERE / "predict.py", pkg / "predict.py")
    shutil.copy(out / "hp_final.json", pkg / "hyperparameters.json")
    shutil.copy(HERE / "fit.py", pkg / "training/fit.py")
    (pkg / "reproduce.json").write_text(json.dumps(
        {"image": "python", "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]}, indent=2) + "\n")
    t1 = time.time()
    run([PY, pkg / "predict.py", "--data", data, "--out", out / "submission.csv"])
    runtime = time.time() - t1
    check_test = run([PY, "check_submission.py", out / "submission.csv", "--data", data], cwd=kit)
    print(check_test)
    val_points = next((l for l in check_val.splitlines() if "points" in l.lower()), "")
    hp = json.loads((out / "hp_final.json").read_text())
    import numpy as np  # noqa: E402  (only for the README summary)
    top = sorted(hp["rules"], key=lambda x: -x[1])[:4]
    readme = f"""# House Edge submission ({rnd} round): generalized Bayesian rule model

## Entry

- Team: {args.team}
- Round: {rnd}
- Season ID: {meta.get('season_id')} (`public/metadata.json`)
- Forecast file: `submission.csv` (uploaded separately; not in this ZIP)
- Forecast SHA-256: {sha(out / 'submission.csv')}
- Environment: `python` (`reproduce.json`). **Python standard library only**; no `requirements.txt`.

## Setup

- Python 3.13 (tested {platform.python_version()}); uses only `argparse`, `csv`, `json`, `math`, `pathlib`.
- Hardware: 1 CPU core is enough; measured {runtime:.1f} s for the test split.
- Data: the round's public folder at `/data` (reads `/data/test/queries.csv`, `/data/test/history.csv`).

## Recreate the forecast

```sh
python predict.py --data /data --out /output/submission.csv
```

- Output: `/output/submission.csv` with columns `query_id,p_0,p_1,p_2,p_5,p_20`.
- Model file: `hyperparameters.json` (checkpoint). No randomness; float64; deterministic.
- Training (separate, not needed for reproduction): `python training/fit.py --data path/to/public --splits {args.final_splits} --out hyperparameters.json`
  (numpy, pandas, scipy). Fitted on this round's {args.final_splits.replace(',', ' + ')} splits only.

## Model

Bayesian rule model. Each machine's win chance follows a rule: the even parity of its encoded outcomes at a
learned set of lags (single lag, pair or triple; this round: encoding `{hp['encoding']}`, top rules
{', '.join(f'{tuple(r)} {np.exp(lp):.2f}' for r, lp in top)}), with a machine-specific favorable side
and at most one direction flip (P(flip) {hp['p_flip_long']:.2f} long / {hp['p_flip_short']:.2f} short).
Posterior over (rule, direction, flip) with Beta-Bernoulli likelihoods; episode and machine log-odds
offsets; 1x share and 2x:5x:20x split shrunk per machine toward type means.

Validation with train-only fit: {val_points.strip()}

## Local check

- Checker: `{next((l for l in check_test.splitlines() if l.startswith('VALID')), check_test.splitlines()[0])}`
- Fresh-folder reproduction: see `repro` line in the pipeline log (ZIP extracted read-only, rerun, byte comparison).
"""
    (pkg / "README.md").write_text(readme)

    # 7. ZIP + fresh-folder reproduction
    zip_path = out / f"house-edge-{rnd}.zip"
    if zip_path.exists():
        zip_path.unlink()
    run([PY, "build_entry.py", "--source", pkg, "--out", zip_path], cwd=kit)
    fresh = out / "_repro"
    if fresh.exists():
        for root, dirs, files in os.walk(fresh):
            os.chmod(root, stat.S_IRWXU)
        shutil.rmtree(fresh)
    (fresh / "submission").mkdir(parents=True); (fresh / "output").mkdir()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(fresh / "submission")
    for root, dirs, files in os.walk(fresh / "submission"):
        for f in files:
            os.chmod(Path(root) / f, stat.S_IRUSR)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(fresh)}
    subprocess.run([PY, "-I", "-B", "predict.py", "--data", str(data), "--out", str(fresh / "output/submission.csv")],
                   cwd=fresh / "submission", env=env, check=True, capture_output=True)
    same = (fresh / "output/submission.csv").read_bytes() == (out / "submission.csv").read_bytes()
    print(f"repro: fresh read-only rerun of the ZIP {'IDENTICAL' if same else 'DIFFERS'} to submission.csv")
    print(f"\nDONE in {time.time() - t0:.0f}s\n  report:   {out / 'report.md'}\n  CSV:      {out / 'submission.csv'}\n  ZIP:      {zip_path}\n  {val_points.strip()}")
    if not same:
        raise SystemExit("reproduction mismatch")


if __name__ == "__main__":
    main()
