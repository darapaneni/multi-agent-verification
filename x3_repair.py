# Author: Narayana Darapaneni
# Code for "Why Verification Fails in Multi-Agent LLM Systems" (2026). MIT License.
"""X3: verify-repair dynamics on HumanEval (experiment E8, Section 5.3 of the paper).

A model writes a solution; the unit tests of the task are split into individual checks
(the verifiers).  Solutions that fail some checks enter a repair loop under three policies:

  P0  single feedback : the repairer sees only the FIRST failing check (the "most violated
                         verifier") and every new version is accepted.  This is the usual
                         verify-and-fix loop of ChatDev/MetaGPT-style systems and the
                         discrete analogue of exact metric repair (Proposition 3.5).
  P1  full feedback   : the repairer sees ALL failing checks; every new version is accepted.
  P2  potential-guided: the repairer sees all failing checks and the passing checks that
                         must stay passing; a new version is accepted only if the number of
                         failing checks strictly decreases (a potential function, as in the
                         Fejer property of Theorem 3.8); two rejected proposals in a row end
                         the loop (termination rule).

Recorded per episode: final pass, rounds, revisits of earlier code states (step repetition),
revisits of earlier failing sets, and oscillating checks (pass -> fail after having passed).

SAFETY: this script executes code written by a language model.  It runs each program in a
separate Python process with a timeout, in a temporary folder, but it is not a sandbox.
Run it inside a virtual machine or container if you can (see README).

Usage:  python x3_repair.py --phase test|final [--dry-run]
"""
import argparse
import ast
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile

from common import (PHASES, MODELS, LLM, Budget, load_humaneval, save_jsonl, parallel_map,
                    RESULTS)

SYS = "You are an expert Python programmer. You write correct, self-contained Python functions."
GEN_USER = ("Write the complete function described below, including the signature and any imports "
            "it needs. Reply with a single ```python code block and nothing else.\n\n{prompt}{mock}")
REPAIR_HEAD = ("The function below is supposed to solve this task:\n\n{prompt}\n\n"
               "CURRENT CODE:\n```python\n{code}\n```\n\n")
P0_TAIL = ("It fails this check:\n{fails}\n\nFix the code. Reply with the complete corrected "
           "function in a single ```python code block and nothing else.{mock}")
P1_TAIL = ("It fails these checks:\n{fails}\n\nFix the code. Reply with the complete corrected "
           "function in a single ```python code block and nothing else.{mock}")
P2_TAIL = ("It fails these checks:\n{fails}\n\nIt already passes these checks, which must keep "
           "passing:\n{passes}\n\nMake the smallest change that fixes the failing checks without "
           "breaking the passing ones. Reply with the complete corrected function in a single "
           "```python code block and nothing else.{mock}")


# --------------------------------------------------------------------------- tests
def split_tests(test_src, entry_point):
    """Returns (module_setup_src, [unit_src ...]).  Each unit is one top-level statement of
    check() that contains an assert, preceded by the non-assert statements before it."""
    tree = ast.parse(test_src)
    setup_mod, check = [], None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            check = node
        else:
            setup_mod.append(ast.unparse(node))
    units, prefix = [], []
    for st in check.body:
        has_assert = any(isinstance(n, ast.Assert) for n in ast.walk(st))
        src = ast.unparse(st)
        if has_assert:
            units.append("\n".join(prefix + [src]))
        else:
            prefix.append(src)
    return "\n".join(setup_mod), units


HARNESS = r'''
import json, sys, traceback
{imports}

{code}

{setup}

def __units():
    return [{unit_funcs}]

{unit_defs}

if __name__ == "__main__":
    try:
        candidate = {entry}
    except Exception as e:
        for i in range({n}):
            print(json.dumps({{"i": i, "ok": False, "err": "entry point missing: " + repr(e)[:150]}}), flush=True)
        sys.exit(0)
    for i, f in enumerate(__units()):
        try:
            f(candidate)
            print(json.dumps({{"i": i, "ok": True}}), flush=True)
        except AssertionError as e:
            print(json.dumps({{"i": i, "ok": False, "err": "AssertionError " + repr(e)[:150]}}), flush=True)
        except Exception as e:
            print(json.dumps({{"i": i, "ok": False, "err": type(e).__name__ + ": " + str(e)[:150]}}), flush=True)
'''


def run_tests(task, code, units, setup, timeout=20):
    imports = "\n".join(l for l in task["prompt"].splitlines()
                        if l.startswith("import ") or l.startswith("from "))
    defs, names = [], []
    for i, u in enumerate(units):
        body = "\n".join("    " + l for l in u.splitlines())
        defs.append(f"def __u{i}(candidate):\n{body}\n")
        names.append(f"__u{i}")
    prog = HARNESS.format(imports=imports, code=code, setup=setup, unit_funcs=", ".join(names),
                          unit_defs="\n".join(defs), entry=task["entry_point"], n=len(units))
    res = {i: (False, "not run (timeout or crash)") for i in range(len(units))}
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "prog.py")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(prog)
        try:
            p = subprocess.run([sys.executable, "-I", f], cwd=d, capture_output=True, text=True,
                               timeout=timeout, env={k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "TEMP", "TMP")
                                                     if k in os.environ})
            out, err = p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = "timeout"
        for line in out.splitlines():
            try:
                r = json.loads(line)
                res[r["i"]] = (r["ok"], r.get("err", ""))
            except Exception:
                pass
        if not out.strip() and err:
            msg = err.strip().splitlines()[-1][:150] if err.strip() else "error"
            res = {i: (False, msg) for i in range(len(units))}
    return res


def extract_code(text, entry):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    for b in blocks:
        if f"def {entry}" in b:
            return b.strip()
    return blocks[0].strip() if blocks else text.strip()


def norm_hash(code):
    return hashlib.sha256(re.sub(r"\s+", "", code).encode()).hexdigest()[:16]


def fmt_checks(units, ids, res=None):
    out = []
    for i in ids:
        s = f"[check {i}] {units[i].strip()}"
        if res is not None:
            s += f"\n    -> {res[i][1]}"
        out.append(s)
    return "\n".join(out) if out else "(none)"


# --------------------------------------------------------------------------- repair loop
def episode(llm, model, policy, task, code0, units, setup, rounds, eff, mock):
    reasoning = MODELS[model][3]
    res = run_tests(task, code0, units, setup)
    fail = sorted(i for i in res if not res[i][0])
    code = code0
    hist = [dict(round=0, hash=norm_hash(code), fail=fail, accepted=True)]
    seen_hash, seen_fail = {norm_hash(code)}, {tuple(fail)}
    ever_pass = {i for i in res if res[i][0]}
    revisits_code = revisits_fail = 0
    oscillating = set()
    stalls = 0
    calls = 0
    for t in range(1, rounds + 1):
        if not fail:
            break
        head = REPAIR_HEAD.format(prompt=task["prompt"], code=code)
        if policy == "P0":
            user = head + P0_TAIL.format(fails=fmt_checks(units, fail[:1], res), mock=mock)
        elif policy == "P1":
            user = head + P1_TAIL.format(fails=fmt_checks(units, fail, res), mock=mock)
        else:
            passing = [i for i in range(len(units)) if i not in fail]
            user = head + P2_TAIL.format(fails=fmt_checks(units, fail, res),
                                         passes=fmt_checks(units, passing), mock=mock)
        text, _ = llm(model, SYS, user, max_out=4000 if reasoning else 1200, temperature=0.7,
                      effort=eff, tag=f"x3:{policy}:{task['task_id']}", sample=t)
        calls += 1
        new = extract_code(text, task["entry_point"])
        nres = run_tests(task, new, units, setup)
        nfail = sorted(i for i in nres if not nres[i][0])
        accept = True
        if policy == "P2" and len(nfail) >= len(fail):
            accept = False
        h = norm_hash(new)
        if h in seen_hash:
            revisits_code += 1
        if tuple(nfail) in seen_fail and nfail != fail:
            revisits_fail += 1
        seen_hash.add(h)
        seen_fail.add(tuple(nfail))
        if accept:
            for i in nfail:
                if i in ever_pass:
                    oscillating.add(i)
            ever_pass |= {i for i in nres if nres[i][0]}
            code, res, fail = new, nres, nfail
            stalls = 0
        else:
            stalls += 1
        hist.append(dict(round=t, hash=h, fail=nfail, accepted=accept))
        if policy == "P2" and stalls >= 2:
            break
    return dict(final_pass=not fail, final_fail=len(fail), init_fail=len(hist[0]["fail"]),
                n_checks=len(units), calls=calls, revisits_code=revisits_code,
                revisits_fail=revisits_fail, oscillating=len(oscillating), hist=hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["test", "final"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = PHASES[a.phase]["x3"]
    budget = Budget(a.phase, "x3", cfg["budget"], a.dry_run)
    llm = LLM(budget, a.dry_run)
    out_path = RESULTS / a.phase / ("x3_repair_dry.jsonl" if a.dry_run else "x3_repair.jsonl")

    tasks = load_humaneval()
    rng = random.Random(11)
    order = list(range(len(tasks)))
    rng.shuffle(order)                     # fixed order: the test set is a prefix of the final set
    sel = order[: cfg["n_problems"]]
    rows, stopped = [], None
    for model in cfg["repairers"]:
        if stopped:
            break
        reasoning = MODELS[model][3]

        def one(ti, model=model, reasoning=reasoning):
            task = tasks[ti]
            setup, units = split_tests(task["test"], task["entry_point"])
            mock = (f"\nMOCK_SOLUTION<<<{task['prompt']}{task['canonical_solution']}>>>"
                    if a.dry_run else "")
            text, _ = llm(model, SYS, GEN_USER.format(prompt=task["prompt"], mock=mock),
                          max_out=4000 if reasoning else 1200, temperature=0.8,
                          effort=cfg["effort"], tag=f"x3:gen:{task['task_id']}")
            code0 = extract_code(text, task["entry_point"])
            res0 = run_tests(task, code0, units, setup)
            init_fail = [i for i in res0 if not res0[i][0]]
            base = dict(task=task["task_id"], model=model, n_checks=len(units),
                        init_fail=len(init_fail))
            if not init_fail:
                return [dict(base, policy="none", final_pass=True)]
            out = []
            for pol in cfg["policies"]:
                ep = episode(llm, model, pol, task, code0, units, setup, cfg["rounds"],
                             cfg["effort"], mock)
                out.append(dict(base, policy=pol, **ep))
            return out

        res, stop = parallel_map(one, sel, a.workers)
        got = [r for lst in res if lst for r in lst]
        rows += got
        n_fail = len({r["task"] for r in got if r["policy"] != "none"})
        n_done = len({r["task"] for r in got})
        print(f"{model}: {n_done} tasks done, {n_fail} initial solutions failed some check; "
              f"x3 spend ${budget.spent():.4f}")
        if stop and stop.startswith("budget"):
            stopped = stop
    save_jsonl(out_path, rows)
    print(f"saved {len(rows)} rows to {out_path}; x3 spend ${budget.spent():.4f}"
          + (f"; stopped early: {stopped}" if stopped else ""))


if __name__ == "__main__":
    main()
