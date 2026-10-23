# Author: Narayana Darapaneni
# Code for "Why Verification Fails in Multi-Agent LLM Systems" (2026). MIT License.
"""Runs one phase end to end: preflight, X1, X2, X3, analysis.

  python run_all.py --phase test            # about $1-2
  python run_all.py --phase final           # about $5-10
  python run_all.py --phase test --dry-run  # offline rehearsal, no API calls, no cost
"""
import argparse
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--phase", choices=["test", "final"], required=True)
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--workers", type=int, default=8, help="parallel API requests")
ap.add_argument("--skip", nargs="*", default=[], help="experiments to skip, e.g. --skip x3")
a = ap.parse_args()
extra = ["--dry-run"] if a.dry_run else []
steps = [["preflight.py"], ["x1_judges.py"], ["x2_bestofn.py"], ["x3_repair.py"], ["analyze.py"]]
for st in steps:
    name = st[0].split("_")[0].replace(".py", "")
    if name in a.skip:
        continue
    cmd = [sys.executable, st[0], "--phase", a.phase] + extra
    if st[0].startswith("x"):
        cmd += ["--workers", str(a.workers)]
    print("\n" + "=" * 70 + "\n$ " + " ".join(cmd) + "\n" + "=" * 70, flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"step {st[0]} failed with exit code {r.returncode}; fix the problem and re-run "
                 "(finished calls are cached and will not be paid again).")
print("\nDone. Results are in results/%s (summary.json, tables.tex, figures, cost_report.txt)." % a.phase)
