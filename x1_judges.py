# Author: Narayana Darapaneni
# Code for "Why Verification Fails in Multi-Agent LLM Systems" (2026). MIT License.
"""X1: LLM judges as verifiers on the 61 human-annotated MAST traces.

Each judge (one OpenAI model) reads a trace and states, for each of six MAST failure
modes, whether the mode is present.  The judge for model k and mode m is verifier
(k, m); analyze.py estimates its TPR, FPR and informedness J = (TPR - FPR)^+ against the
human labels and feeds the matrix to Proposition 4.1 / Algorithm 3 of the paper.

Usage:  python x1_judges.py --phase test|final [--dry-run]
"""
import argparse
import json
import random

from common import (PHASES, MODES, MODELS, LLM, Budget, load_mast, truncate, save_jsonl,
                    parallel_map, RESULTS)

SYSTEM = ("You are an expert annotator of execution traces of multi-agent LLM systems. "
          "You apply the MAST failure taxonomy strictly and only report a failure mode when the "
          "trace contains concrete evidence of it.")

DEFINITIONS = """Failure modes (from the MAST taxonomy):
FM-1.3 Step repetition: an agent or the system unnecessarily repeats a phase, task or stage that was already completed (for example the same thought, command or message repeated).
FM-2.1 Conversation reset: unexpected or unwarranted restarting of the dialogue or trajectory, losing context and progress.
FM-2.2 Fail to ask for clarification: faced with unclear, incomplete, ambiguous or contradictory information, the agents proceed on assumptions instead of detecting the problem or requesting clarification.
FM-2.3 Task derailment: deviation from the intended objective or focus of the task, leading to irrelevant or unproductive actions.
FM-3.2 No or incomplete verification: the system does not attempt to check its final outcome, or its checks are superficial and miss essential aspects.
FM-3.3 Incorrect verification: a verifier or evaluator agent exists but fails to be critical; it accepts wrong results or does not do what it was asked to check."""

USER_TMPL = """{defs}

TASK GIVEN TO THE SYSTEM:
{task}

EXECUTION TRACE ({fw}; long traces are shortened in the middle):
<<<TRACE
{trace}
TRACE>>>

For each failure mode decide whether it is present in this trace.
Reply with one JSON object only, no other text, with exactly these keys and boolean values:
{{"FM-1.3": true/false, "FM-2.1": true/false, "FM-2.2": true/false, "FM-2.3": true/false, "FM-3.2": true/false, "FM-3.3": true/false}}
MODES_JSON"""


def parse_flags(text):
    try:
        s = text[text.index("{"): text.rindex("}") + 1]
        d = json.loads(s)
        out = {}
        for m in MODES:
            v = d.get(m)
            if isinstance(v, str):
                v = v.strip().lower() in ("true", "yes", "1")
            out[m] = bool(v) if v is not None else None
        return out, all(out[m] is not None for m in MODES)
    except Exception:
        return {m: None for m in MODES}, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["test", "final"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = PHASES[a.phase]["x1"]
    budget = Budget(a.phase, "x1", cfg["budget"], a.dry_run)
    llm = LLM(budget, a.dry_run)
    out_path = RESULTS / a.phase / ("x1_judgements_dry.jsonl" if a.dry_run else "x1_judgements.jsonl")

    recs = load_mast()
    rng = random.Random(2026)
    idx = list(range(len(recs)))
    if cfg["n_traces"] < len(recs):     # stratified subset (only if n_traces < 61)
        ag = [i for i in idx if recs[i]["fw"] == "AG2"]
        hy = [i for i in idx if recs[i]["fw"] == "HyperAgent"]
        k = cfg["n_traces"] // 2
        idx = sorted(rng.sample(ag, k) + rng.sample(hy, cfg["n_traces"] - k))

    rows, stopped = [], None
    for judge in cfg["judges"]:
        if stopped:
            break
        reasoning = MODELS[judge][3]
        for rep in range(cfg["repeats"].get(judge, 1)):

            def one(i, judge=judge, rep=rep):
                r = recs[i]
                user = USER_TMPL.format(defs=DEFINITIONS, task=truncate(r["task"], 3000), fw=r["fw"],
                                        trace=truncate(r["text"], cfg["max_chars"]))
                text, cost = llm(judge, SYSTEM, user, max_out=3000 if reasoning else 300,
                                 temperature=0.0 if rep == 0 else 1.0, effort=cfg["effort"],
                                 tag=f"x1:{judge}:{r['id']}", sample=rep)
                flags, ok = parse_flags(text)
                return dict(trace=r["id"], idx=i, fw=r["fw"], judge=judge, rep=rep,
                            flags=flags, parse_ok=ok, cost=cost)

            res, stop = parallel_map(one, idx, a.workers)
            got = [x for x in res if x is not None]
            rows += got
            print(f"{judge} rep {rep}: {len(got)} of {len(idx)} traces judged, "
                  f"{sum(not x['parse_ok'] for x in got)} unparsable; x1 spend ${budget.spent():.4f}")
            if stop:
                if stop.startswith("budget"):
                    stopped = stop
                break
    save_jsonl(out_path, rows)
    print(f"saved {len(rows)} judgements to {out_path}; x1 spend ${budget.spent():.4f}"
          + (f"; stopped early: {stopped}" if stopped else ""))


if __name__ == "__main__":
    main()
