# Author: Narayana Darapaneni
# Code for "Why Verification Fails in Multi-Agent LLM Systems" (2026). MIT License.
"""Shared utilities for the LLM experiments of
"Why Verification Fails in Multi-Agent LLM Systems".

Contents
  * MODELS        : model registry (API id, prices per 1M tokens, reasoning flag)
  * PHASES        : test ($2) and final ($10) configurations
  * Budget        : hard spending guard (estimates before every call, books actual usage after)
  * LLM           : cached, retrying, budget-guarded wrapper around the OpenAI Responses API
  * MockClient    : offline stand-in used with --dry-run (no network, no cost)
  * data loaders  : MAST human-annotated traces, GSM8K test split, HumanEval
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = HERE / "cache"
RESULTS = HERE / "results"

# ---------------------------------------------------------------------------
# Model registry.  Prices are USD per 1M tokens (Standard tier, short context).
#   * GPT-5.6 Sol/Terra/Luna, GPT-5.5, GPT-5.5 Pro, GPT-5.4, GPT-5.4 Mini:
#     published rates after the 30 July 2026 price change.
#   * Older models: list prices as last known; CHECK them on
#     https://platform.openai.com/docs/pricing before the final phase.
#   * price None  -> the budget guard refuses to call the model until you fill it in.
# "reasoning": True means the model takes reasoning={"effort": ...} and no temperature.
# ---------------------------------------------------------------------------
MODELS = {
    # name in this package : (api id, input $/1M, output $/1M, reasoning)
    "gpt-5.6-sol":   ("gpt-5.6-sol",   5.00, 30.00, True),
    "gpt-5.6-terra": ("gpt-5.6-terra", 2.00, 12.00, True),
    "gpt-5.6-luna":  ("gpt-5.6-luna",  0.20,  1.20, True),
    "gpt-5.5":       ("gpt-5.5",       5.00, 30.00, True),
    "gpt-5.5-pro":   ("gpt-5.5-pro",  30.00, 180.0, True),
    "gpt-5.4":       ("gpt-5.4",       2.50, 15.00, True),
    "gpt-5.4-pro":   ("gpt-5.4-pro",   None,  None, True),
    "gpt-5.4-mini":  ("gpt-5.4-mini",  0.75,  4.50, True),
    "gpt-5.2":       ("gpt-5.2",       None,  None, True),
    "gpt-5.2-pro":   ("gpt-5.2-pro",   None,  None, True),
    "gpt-5":         ("gpt-5",         1.25, 10.00, True),
    "gpt-5-mini":    ("gpt-5-mini",    0.25,  2.00, True),
    "gpt-5-nano":    ("gpt-5-nano",    0.05,  0.40, True),
    "gpt-5-pro":     ("gpt-5-pro",    15.00, 120.0, True),
    "o3":            ("o3",            2.00,  8.00, True),
    "o3-pro":        ("o3-pro",       20.00, 80.00, True),
    "gpt-4.1":       ("gpt-4.1",       2.00,  8.00, False),
    "gpt-4.1-mini":  ("gpt-4.1-mini",  0.40,  1.60, False),
    "gpt-4o":        ("gpt-4o",        2.50, 10.00, False),
    "gpt-4o-mini":   ("gpt-4o-mini",   0.15,  0.60, False),
}

# ---------------------------------------------------------------------------
# Phase configurations.  Experiments get separate sub-budgets so that one
# experiment cannot use up the money of another.  The sum stays below the cap,
# leaving a margin for estimation error.
# Models are listed cheapest first: if a sub-budget runs out, the expensive
# models are the ones that are dropped.
# ---------------------------------------------------------------------------
PHASES = {
    "test": dict(
        cap=2.00,
        # The test phase is a real pilot.  It uses the same prompts, traces and problem
        # order as the final phase, so every call made here is cached and reused (not paid
        # again) in the final phase.
        x1=dict(budget=1.00, judges=["gpt-5-nano", "gpt-4o-mini", "gpt-5.6-luna", "gpt-4.1-mini",
                                     "gpt-5-mini"],
                repeats={}, n_traces=61, max_chars=24000, effort="low"),
        x2=dict(budget=0.45, n_problems=100, n_max=8, generator="gpt-4o-mini",
                solver="gpt-5.6-luna", judge="gpt-4o-mini", blind="gpt-4o-mini", effort="low"),
        x3=dict(budget=0.35, n_problems=80, rounds=8, repairers=["gpt-4o-mini"],
                policies=["P0", "P1", "P2"], effort="low"),
    ),
    "final": dict(
        cap=10.00,
        x1=dict(budget=5.00,
                judges=["gpt-5-nano", "gpt-4o-mini", "gpt-5.6-luna", "gpt-4.1-mini", "gpt-5-mini",
                        "gpt-5.4-mini", "gpt-5.6-terra", "o3"],
                repeats={"gpt-5-nano": 2, "gpt-4o-mini": 2, "gpt-5.6-luna": 2},
                n_traces=61, max_chars=24000, effort="low"),
        x2=dict(budget=2.50, n_problems=250, n_max=16, generator="gpt-4o-mini",
                solver="gpt-5.6-luna", judge="gpt-4o-mini", blind="gpt-4o-mini", effort="low"),
        x3=dict(budget=1.50, n_problems=164, rounds=8, repairers=["gpt-4o-mini", "gpt-5-nano"],
                policies=["P0", "P1", "P2"], effort="low"),
    ),
}


# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------
class BudgetExceeded(RuntimeError):
    pass


def est_tokens(text: str) -> int:
    """Conservative token estimate (about 3 characters per token)."""
    return int(len(text) / 3) + 20


def price(model: str):
    _, pin, pout, _ = MODELS[model]
    if pin is None or pout is None:
        raise BudgetExceeded(f"No price set for {model}; edit MODELS in common.py before using it.")
    return pin, pout


class Budget:
    """Books every call in a JSONL ledger.  Spending is restored from the ledger at start-up,
    so the guard survives restarts.  Thread safe: before a call the worst-case cost is
    reserved, so parallel workers cannot overshoot the limit together."""

    def __init__(self, phase: str, experiment: str, limit: float, dry_run: bool = False):
        self.phase, self.exp, self.limit = phase, experiment, limit
        self.dir = RESULTS / phase
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger = self.dir / ("ledger_dry.jsonl" if dry_run else "ledger.jsonl")
        self.cap = PHASES[phase]["cap"]
        self.lock = threading.Lock()
        self._exp, self._phase, self.pending = self._read(experiment), self._read(None), 0.0

    def _read(self, exp):
        tot = 0.0
        if self.ledger.exists():
            for line in self.ledger.read_text().splitlines():
                r = json.loads(line)
                if exp is None or r["exp"] == exp:
                    tot += r["cost"]
        return tot

    def spent(self):
        return self._exp

    def spent_phase(self):
        return self._phase

    def check(self, model, prompt_tokens, max_out):
        """Reserves the worst-case cost of the next call; raises BudgetExceeded if it does not fit."""
        pin, pout = price(model)
        worst = prompt_tokens * pin / 1e6 + max_out * pout / 1e6
        with self.lock:
            if self._exp + self.pending + worst > self.limit:
                raise BudgetExceeded(f"{self.exp}: sub-budget ${self.limit:.2f} reached "
                                     f"(spent ${self._exp:.4f}).")
            if self._phase + self.pending + worst > self.cap:
                raise BudgetExceeded(f"phase cap ${self.cap:.2f} reached.")
            self.pending += worst
        return worst

    def release(self, reserved):
        with self.lock:
            self.pending -= reserved

    def book(self, model, tin, tout, tag, reserved=0.0):
        pin, pout = price(model)
        cost = tin * pin / 1e6 + tout * pout / 1e6
        with self.lock:
            with self.ledger.open("a") as f:
                f.write(json.dumps(dict(exp=self.exp, model=model, tin=tin, tout=tout,
                                        cost=cost, tag=tag, t=time.time())) + "\n")
            self._exp += cost
            self._phase += cost
            self.pending -= reserved
        return cost


# ---------------------------------------------------------------------------
# Offline mock client (for --dry-run)
# ---------------------------------------------------------------------------
@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _Resp:
    output_text: str
    usage: _Usage
    status: str = "completed"


class MockClient:
    """Imitates client.responses.create.  Produces random but well-formed outputs."""

    class _R:
        def __init__(self, outer):
            self.outer = outer

        def create(self, model, input, max_output_tokens, **kw):
            rnd = random.Random(hashlib.sha256((json.dumps(input) + str(kw.get("_sample", 0))).encode()).hexdigest())
            user = input[-1]["content"]
            if "MODES_JSON" in user:
                out = json.dumps({m: rnd.random() < 0.3 for m in
                                  ["FM-1.3", "FM-2.1", "FM-2.2", "FM-2.3", "FM-3.2", "FM-3.3"]})
            elif "PLAUSIBILITY" in user:
                out = f"Score: {rnd.randint(1, 10)}"
            elif "VERDICT" in user:
                out = rnd.choice(["CORRECT", "INCORRECT"])
            elif "Answer:" in user:
                m = re.search(r"MOCK_TRUTH=(-?\d+)", user)
                truth = int(m.group(1)) if m else 0
                out = f"Short reasoning.\nAnswer: {truth if rnd.random() < 0.7 else truth + rnd.randint(1, 9)}"
            elif "```python" in user or "Write the complete function" in user:
                m = re.search(r"MOCK_SOLUTION<<<(.*?)>>>", user, re.S)
                code = m.group(1) if (m and rnd.random() < 0.6) else "def placeholder():\n    return None\n"
                out = "```python\n" + code + "\n```"
            else:
                out = "ok"
            tin = est_tokens(json.dumps(input))
            tout = min(max_output_tokens, est_tokens(out) + (200 if kw.get("reasoning") else 0))
            return _Resp(out, _Usage(tin, tout))

    def __init__(self):
        self.responses = MockClient._R(self)


# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------
class LLM:
    def __init__(self, budget: Budget, dry_run=False):
        self.budget, self.dry = budget, dry_run
        if dry_run:
            self.client = MockClient()
            self.cache_dir = CACHE / "mock"
        else:
            from openai import OpenAI  # imported here so --dry-run works without the package
            if not os.environ.get("OPENAI_API_KEY"):
                sys.exit("OPENAI_API_KEY is not set (see README, step 3).")
            self.client = OpenAI(max_retries=0, timeout=180)
            self.cache_dir = CACHE / "live"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, model, system, user, max_out=600, temperature=0.0, effort="low",
                 tag="", sample=0):
        """Returns (text, cost).  cost is the list-price cost of the call (also reported
        for cache hits, so per-model cost tables are complete).  Identical requests are
        served from the cache and are NOT paid again, so re-running a script, or running
        the final phase after the test phase, never pays twice for the same call."""
        api_id, _, _, reasoning = MODELS[model]
        key = dict(model=api_id, system=system, user=user, max_out=max_out,
                   temperature=None if reasoning else temperature,
                   effort=effort if reasoning else None, sample=sample)
        h = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        cf = self.cache_dir / f"{h}.json"
        if cf.exists():                       # cache hit: nothing is spent
            c = json.loads(cf.read_text())
            pin, pout = price(model)
            return c["text"], c["tin"] * pin / 1e6 + c["tout"] * pout / 1e6

        reserved = self.budget.check(model, est_tokens(system + user), max_out)
        kwargs = dict(model=api_id,
                      input=[{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                      max_output_tokens=max_out)
        if reasoning:
            kwargs["reasoning"] = {"effort": effort}
        else:
            kwargs["temperature"] = temperature
        if self.dry:
            kwargs["_sample"] = sample          # lets the mock vary between samples
        else:
            kwargs["store"] = False

        resp, delay = None, 2.0
        for attempt in range(6):
            try:
                resp = self.client.responses.create(**kwargs)
                break
            except Exception as e:  # rate limits, timeouts, transient 5xx
                name = type(e).__name__
                msg = str(e)
                fatal = any(s in name for s in ("BadRequest", "Authentication", "PermissionDenied",
                                                "NotFound"))
                if fatal or attempt == 5:
                    self.budget.release(reserved)
                    raise RuntimeError(f"{model}: {name}: {msg[:300]}") from e
                time.sleep(delay)
                delay *= 2
        text = getattr(resp, "output_text", "") or ""
        u = resp.usage
        cost = self.budget.book(model, u.input_tokens, u.output_tokens, tag, reserved)
        cf.write_text(json.dumps(dict(key=key, text=text, tin=u.input_tokens,
                                      tout=u.output_tokens, status=getattr(resp, "status", ""))))
        return text, cost


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
MODEMAP = {
    "FM-1.3": ["Step repetition"],
    "FM-2.1": ["Trajectory restart"],
    "FM-2.2": ["Fail to detect ambiguities/contradictions", "Proceed with incorrect assumptions",
               "Fail to elicit clarification"],
    "FM-2.3": ["Derailing from task objectives"],
    "FM-3.2": ["No attempt to verify outcome"],
    "FM-3.3": ["Evaluator agent fails to be critical"],
}
MODES = list(MODEMAP)


def ensure_mast():
    """Clones the MAST repository (only the traces are needed)."""
    root = DATA / "MAST"
    if not (root / "traces").exists():
        DATA.mkdir(exist_ok=True)
        print("Cloning MAST repository (about 200 MB) ...")
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/multi-agent-systems-failure-taxonomy/MAST.git",
                        str(root)], check=True)
    return root / "traces"


def load_mast():
    """The 61 human-annotated traces (AG2 and HyperAgent), with labels mapped to six MAST modes.
    Same mapping and order as exp_real.py."""
    root = ensure_mast()
    recs = []
    for f in sorted(root.glob("**/*_human.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        opts = d["note"]["options"]
        lab = {}
        for m, keys in MODEMAP.items():
            present = [k for k in keys if k in opts]
            lab[m] = None if not present else any(str(opts[k]).lower().startswith("yes") for k in present)
        fw = "AG2" if "AG2" in f.parts else "HyperAgent"
        # 'lines'/'ptext' reproduce exactly the representation used by the programmatic
        # verifiers W1-W6 of the paper (exp_real.py); 'text' is the version shown to judges.
        if fw == "AG2":
            raw = ["\n".join(t["content"]) if isinstance(t["content"], list) else str(t["content"])
                   for t in d["trajectory"]]
            plines = [l for m_ in raw[1:] for l in m_.split("\n")]
        else:
            plines = [l.split(" - INFO - ", 1)[-1] for l in d["trajectory"] if isinstance(l, str)]
        if fw == "AG2":
            msgs = []
            for t in d["trajectory"]:
                c = "\n".join(t["content"]) if isinstance(t["content"], list) else str(t["content"])
                msgs.append(f"[{t.get('name', t.get('role', 'agent'))}]\n{c}")
            task = "\n".join(d["problem_statement"]) if isinstance(d["problem_statement"], list) \
                else str(d["problem_statement"])
            text = "\n\n".join(msgs)
        else:
            lines = [l.split(" - INFO - ", 1)[-1] for l in d["trajectory"] if isinstance(l, str)]
            text = "\n".join(lines)
            ps = d.get("problem_statement", "")
            task = "\n".join(ps) if isinstance(ps, list) else str(ps)
        recs.append(dict(id=f.stem, fw=fw, lab=lab, text=text, task=task, lines=plines,
                         ptext="\n".join(plines),
                         other=d.get("other_data", {}), path=str(f)))
    return recs


def truncate(text, max_chars):
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    return text[:head] + f"\n\n[... {len(text) - max_chars} characters omitted ...]\n\n" + text[-tail:]


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_gsm8k():
    p = _download("https://raw.githubusercontent.com/openai/grade-school-math/master/"
                  "grade_school_math/data/test.jsonl", DATA / "gsm8k_test.jsonl")
    out = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        d = json.loads(line)
        ans = d["answer"].split("####")[-1].strip().replace(",", "")
        out.append(dict(id=i, question=d["question"], answer=ans))
    return out


def load_humaneval():
    p = _download("https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
                  DATA / "HumanEval.jsonl.gz")
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def parse_number(text):
    """Extracts the final numeric answer (after 'Answer:' if present)."""
    m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        m = re.findall(r"(-?[\d,]*\.?\d+)", text)
    if not m:
        return None
    s = m[-1].replace(",", "")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return None


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def parallel_map(fn, items, workers):
    """Runs fn over items with a thread pool.  Returns (results_in_order, stop_message).
    Items whose call raised BudgetExceeded give None; after the first BudgetExceeded no new
    items are started.  Other exceptions are reported and give None."""
    from concurrent.futures import ThreadPoolExecutor
    results, stop = [None] * len(items), None
    stop_flag = threading.Event()
    errors = [0]
    elock = threading.Lock()

    def run(i):
        if stop_flag.is_set():
            return i, None, None
        try:
            return i, fn(items[i]), None
        except BudgetExceeded as e:
            stop_flag.set()
            return i, None, f"budget: {e}"
        except Exception as e:  # noqa: BLE001
            with elock:
                errors[0] += 1
                if errors[0] >= 5:            # e.g. wrong model id: do not hammer the API
                    stop_flag.set()
            return i, None, f"error: {type(e).__name__}: {str(e)[:200]}"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, res, err in ex.map(run, range(len(items))):
            results[i] = res
            if err:
                print("  !", err)
                if err.startswith("budget") and stop is None:
                    stop = err
    if stop is None and errors[0] >= 5:
        stop = "error: stopped after 5 failed calls"
    return results, stop
