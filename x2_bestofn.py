# Author: Narayana Darapaneni
# Code for "Why Verification Fails in Multi-Agent LLM Systems" (2026). MIT License.
"""X2: best-of-N selection with sound and blind verifiers on GSM8K (Proposition 3.1 and Theorem 3.2).

For each problem a cheap generator produces n_max short candidate solutions.
Selectors that are compared in analyze.py:
  oracle      : true answer known (upper bound)
  solver      : sound verifier; an independent solver computes a reference answer once per
                problem, and candidates agreeing with it are preferred
  judge       : LLM-as-judge that sees question and candidate and says CORRECT/INCORRECT
  majority    : self-consistency vote over the candidates (no extra calls)
  blind       : judge that sees only the candidate text, NOT the question; it can only use
                surface features, which is the blind verifier of Theorem 3.2
  random      : baseline (equals single-sample accuracy)

Usage:  python x2_bestofn.py --phase test|final [--dry-run]
"""
import argparse
import random
import re

from common import (PHASES, MODELS, LLM, Budget, load_gsm8k, parse_number, save_jsonl,
                    parallel_map, RESULTS)

GEN_SYS = "You solve grade-school math word problems."
GEN_USER = ("{q}\n\nKeep your reasoning to at most two short sentences. "
            "End with a line of the form 'Answer: <number>'.{mock}")
SOLVER_SYS = "You are a careful mathematician. Solve the problem exactly."
SOLVER_USER = "{q}\n\nEnd with a line of the form 'Answer: <number>'.{mock}"
JUDGE_SYS = "You check solutions to grade-school math problems."
JUDGE_USER = ("PROBLEM:\n{q}\n\nPROPOSED SOLUTION:\n{c}\n\nIs the final answer of the proposed "
              "solution correct? Reply with one word: CORRECT or INCORRECT. VERDICT")
BLIND_SYS = "You rate short written solutions."
BLIND_USER = ("Here is a short written solution to some math problem (the problem itself is not "
              "shown).\n\nSOLUTION:\n{c}\n\nHow likely is it that this solution is correct? "
              "Reply only with 'Score: <integer from 1 to 10>'. PLAUSIBILITY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["test", "final"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = PHASES[a.phase]["x2"]
    budget = Budget(a.phase, "x2", cfg["budget"], a.dry_run)
    llm = LLM(budget, a.dry_run)
    out_path = RESULTS / a.phase / ("x2_bestofn_dry.jsonl" if a.dry_run else "x2_bestofn.jsonl")

    probs = load_gsm8k()
    rng = random.Random(7)
    order = list(range(len(probs)))
    rng.shuffle(order)                     # fixed order: the test set is a prefix of the final set
    sel = order[: cfg["n_problems"]]
    eff = cfg["effort"]

    def one(pid):
        p = probs[pid]
        mock = f" MOCK_TRUTH={p['answer']}" if a.dry_run else ""
        row = dict(pid=pid, answer=p["answer"], cands=[], judge=[], blind=[], solver=None)
        g_reason = MODELS[cfg["generator"]][3]
        for k in range(cfg["n_max"]):
            t, _ = llm(cfg["generator"], GEN_SYS, GEN_USER.format(q=p["question"], mock=mock),
                       max_out=1500 if g_reason else 200, temperature=1.0, effort=eff,
                       tag=f"x2:gen:{pid}", sample=k)
            row["cands"].append(dict(text=t, ans=parse_number(t)))
        s_reason = MODELS[cfg["solver"]][3]
        t, _ = llm(cfg["solver"], SOLVER_SYS, SOLVER_USER.format(q=p["question"], mock=mock),
                   max_out=4000 if s_reason else 600, temperature=0.0, effort=eff,
                   tag=f"x2:solver:{pid}")
        row["solver"] = parse_number(t)
        j_reason, b_reason = MODELS[cfg["judge"]][3], MODELS[cfg["blind"]][3]
        for c in row["cands"]:
            t, _ = llm(cfg["judge"], JUDGE_SYS, JUDGE_USER.format(q=p["question"], c=c["text"]),
                       max_out=2000 if j_reason else 5, temperature=0.0, effort=eff,
                       tag=f"x2:judge:{pid}")
            u = t.upper()
            row["judge"].append(1 if re.search(r"\bCORRECT\b", u) and "INCORRECT" not in u else 0)
            t, _ = llm(cfg["blind"], BLIND_SYS, BLIND_USER.format(c=c["text"]),
                       max_out=2000 if b_reason else 8, temperature=0.0, effort=eff,
                       tag=f"x2:blind:{pid}")
            m = re.search(r"(\d+)", t)
            row["blind"].append(min(10, max(1, int(m.group(1)))) if m else 5)
        return row

    res, stopped = parallel_map(one, sel, a.workers)
    rows = [r for r in res if r is not None]
    save_jsonl(out_path, rows)
    print(f"saved {len(rows)} problems to {out_path}; x2 spend ${budget.spent():.4f}"
          + (f"; stopped early: {stopped}" if stopped else ""))


if __name__ == "__main__":
    main()
