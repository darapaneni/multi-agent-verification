# Author: Narayana Darapaneni
# Code for "Why Verification Fails in Multi-Agent LLM Systems" (2026). MIT License.
"""Step 1 of every phase: checks the API key, model access and the cost plan.

  python preflight.py --phase test            # cost estimate + one tiny call per model (< $0.01)
  python preflight.py --phase test --no-ping  # cost estimate only, no API calls
  python preflight.py --phase test --dry-run  # everything offline
"""
import argparse

from common import (PHASES, MODELS, LLM, Budget, BudgetExceeded, load_mast, load_gsm8k,
                    load_humaneval, truncate, est_tokens)


def estimate(phase):
    """Rough expected cost per experiment (list prices).  Reasoning models are assumed to use
    about 800 output tokens per judgement and 600 per solution; actual numbers vary."""
    P = PHASES[phase]
    rows = []
    recs = load_mast()
    c1 = P["x1"]
    tin = sum(est_tokens(truncate(r["text"], c1["max_chars"])) + 900 for r in recs[: c1["n_traces"]])
    tin = tin * c1["n_traces"] / min(c1["n_traces"], len(recs))
    for j in c1["judges"]:
        _, pin, pout, reas = MODELS[j]
        reps = c1["repeats"].get(j, 1)
        tout = c1["n_traces"] * (800 if reas else 60)
        rows.append(("x1", j, reps * (tin * pin + tout * pout) / 1e6))
    c2 = P["x2"]
    n, k = c2["n_problems"], c2["n_max"]
    for role, model, calls, ti, to in [("generator", c2["generator"], n * k, 180, 70),
                                      ("solver", c2["solver"], n, 150, 600),
                                      ("judge", c2["judge"], n * k, 260, 3),
                                      ("blind", c2["blind"], n * k, 130, 5)]:
        _, pin, pout, reas = MODELS[model]
        to = to if not reas else max(to, 500)
        rows.append(("x2", f"{model} ({role})", calls * (ti * pin + to * pout) / 1e6))
    c3 = P["x3"]
    for m in c3["repairers"]:
        _, pin, pout, reas = MODELS[m]
        nf = int(0.25 * c3["n_problems"])                 # assume 25% initial failures
        calls_gen = c3["n_problems"]
        calls_rep = nf * len(c3["policies"]) * c3["rounds"] * 0.6
        to = 900 if reas else 250
        rows.append(("x3", m, (calls_gen * (250 * pin + to * pout) +
                               calls_rep * (1300 * pin + to * pout)) / 1e6))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["test", "final"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-ping", action="store_true")
    a = ap.parse_args()
    P = PHASES[a.phase]
    print("Preparing data (first run downloads MAST, GSM8K and HumanEval) ...")
    load_gsm8k(); load_humaneval()
    rows = estimate(a.phase)
    print(f"\nExpected cost for phase '{a.phase}' (cap ${P['cap']:.2f}):")
    for exp in ("x1", "x2", "x3"):
        sub = [r for r in rows if r[0] == exp]
        tot = sum(r[2] for r in sub)
        print(f"  {exp}: about ${tot:.3f}  (sub-budget ${P[exp]['budget']:.2f})")
        for _, m, c in sub:
            print(f"      {m:28s} ${c:.4f}")
    print(f"  total about ${sum(r[2] for r in rows):.3f}.  The budget guard stops any experiment at its "
          "sub-budget, so the cap cannot be exceeded at list prices.")

    models = sorted({m for exp in ("x1", "x2", "x3") for key in ("judges", "repairers")
                     for m in P[exp].get(key, [])} |
                    {P["x2"][r] for r in ("generator", "solver", "judge", "blind")})
    missing = [m for m in models if MODELS[m][1] is None]
    if missing:
        print("\nWARNING: no price set for", missing, "- edit MODELS in common.py.")
    if a.no_ping:
        return
    print("\nChecking access to each model with one tiny request ...")
    llm = LLM(Budget(a.phase, "preflight", 0.05, a.dry_run), a.dry_run)
    ok = True
    for m in models:
        reas = MODELS[m][3]
        try:
            t, c = llm(m, "You are a test.", "Reply with the word OK.", max_out=200 if reas else 5,
                       effort="low", tag="preflight")
            print(f"  {m:16s} OK   (reply {t.strip()[:20]!r}, ${c:.5f})")
        except (RuntimeError, BudgetExceeded) as e:
            ok = False
            print(f"  {m:16s} FAILED: {e}")
    print("\nAll models reachable." if ok else
          "\nSome models failed. Remove them from PHASES in common.py (or fix the model id) before running.")


if __name__ == "__main__":
    main()
