"""Run the Phase 1 experiments and write their tables and a markdown report.

    python analysis/run_phase1.py                    # all experiments, full data
    python analysis/run_phase1.py --only E2,E4       # a subset
    python analysis/run_phase1.py --smoke            # 2 episodes per condition: checks the code runs

Needs analysis/artifacts/examples.pkl (python analysis/build_table.py).
Writes analysis/results/<run>/E*/<table>.csv, report.md and timings.json.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1 import EXPERIMENTS  # noqa: E402
from phase1.common import DEFAULT_TABLE, ROOT, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--only", default=",".join(EXPERIMENTS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = args.out or ROOT / "analysis/results" / ("smoke" if args.smoke else "full")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    ctx = load(args.table, smoke=args.smoke)
    timings = {"load": time.perf_counter() - t0}
    print(f"loaded {len(ctx.hist):,} history rows, {len(ctx.vq)} validation queries "
          f"({'smoke' if args.smoke else 'full'}) in {timings['load']:.1f}s", flush=True)

    pd.set_option("display.float_format", lambda x: f"{x:.4g}")
    report = [f"# Phase 1 results ({'smoke test: tiny subset, numbers are meaningless' if args.smoke else 'full data'})\n",
              f"History rows: {len(ctx.hist):,} (train + validation). Validation queries: {len(ctx.vq)}.\n",
              "Gains are leave-one-out Brier improvements on observed outcomes; `dR` is the gain divided by the "
              "reference's true error in that condition, roughly the drop in R it would give. Positive = helps.\n"]
    failed = []
    for name in [n.strip().upper() for n in args.only.split(",") if n.strip()]:
        mod = EXPERIMENTS[name]
        t = time.perf_counter()
        try:
            tables = mod.run(ctx)
        except Exception:
            failed.append(name)
            print(f"{name} FAILED\n{traceback.format_exc()}", flush=True)
            report.append(f"## {name}: FAILED\n\n```\n{traceback.format_exc()}\n```\n")
            continue
        timings[name] = time.perf_counter() - t
        d = out / name
        d.mkdir(exist_ok=True)
        for tname, df in tables.items():
            df.to_csv(d / f"{tname}.csv", index=False)
        report.append(f"## {name}. {mod.QUESTION}\n\n_{timings[name]:.1f}s. All tables: `{d.relative_to(ROOT)}/`_\n")
        for tname in mod.SHOW:
            df = tables.get(tname)
            if df is not None and not df.empty:
                report.append(f"### {tname}\n\n{df.to_markdown(index=False, floatfmt='.4g')}\n")
        print(f"{name} done in {timings[name]:.1f}s", flush=True)

    (out / "report.md").write_text("\n".join(report))
    (out / "timings.json").write_text(json.dumps(timings, indent=2))
    print(f"report: {out / 'report.md'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
