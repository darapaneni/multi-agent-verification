# Author: Narayana Darapaneni
# Code for "Why Verification Fails in Multi-Agent LLM Systems" (2026). MIT License.
"""X0 (free, no API calls): the programmatic verifiers of the paper on the full
MAST dataset (MAD_full_dataset.json, 1642 traces, 14 failure modes labelled by an LLM judge).

Download (about 200 MB) on your own machine, for example:
    pip install huggingface_hub
    python -c "from huggingface_hub import hf_hub_download as d; print(d('mcemri/MAST-Data','MAD_full_dataset.json',repo_type='dataset',local_dir='data'))"
Then run:
    python x0_mad_full.py data/MAD_full_dataset.json
Writes results/mad_full/summary.json and results/mad_full/tables.tex.

Record format (from the dataset card): mas_name, llm_name, benchmark_name, trace_id,
trace = {key, index, trajectory}, mast_annotation = {"1.1": 0/1/null, ..., "3.3": 0/1/null}.
The reference-answer verifier W6 of the paper needs reference answers, which MAD does not
contain, so only W1-W5 are used here.  Labels come from an LLM judge, not from humans.
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

CODES = ["1.1", "1.2", "1.3", "1.4", "1.5", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "3.1", "3.2", "3.3"]
NAMES = ["W1 repetition monitor", "W2 budget/termination monitor", "W3 verification-evidence check",
         "W4 restart detector", "W5 assumption detector"]
VERIF = re.compile(r"verif|double-check|check (the|our|if|that)|substitut|tests? pass|pytest|"
                   r"run(ning)? (the )?test|reproduc", re.I)
ASSUME = re.compile(r"\bassum|insufficient|not enough information|missing information|unclear|ambigu", re.I)
MIN_POS = 20          # modes with fewer positive (or negative) traces are left out
DELTA = 0.1
rng = np.random.default_rng(7)


def norm(l):
    return re.sub(r"\s+", " ", l.strip().lower())


def lines_of(traj):
    if isinstance(traj, list):
        out = []
        for t in traj:
            if isinstance(t, dict):
                c = t.get("content", "")
                c = "\n".join(c) if isinstance(c, list) else str(c)
                out += c.split("\n")
            else:
                out += str(t).split("\n")
        return out
    return str(traj).split("\n")


def load(path):
    data = json.load(open(path, encoding="utf-8"))
    recs = []
    for d in data:
        tr = d.get("trace", {})
        traj = tr.get("trajectory", "") if isinstance(tr, dict) else tr
        ls = lines_of(traj)
        ann = d.get("mast_annotation", {}) or {}
        lab = [np.nan if ann.get(c) is None else float(ann.get(c)) for c in CODES]
        recs.append(dict(mas=d.get("mas_name", "?"), lines=ls, text="\n".join(ls), lab=lab))
    return recs


def flags(recs):
    med = {}
    for m in {r["mas"] for r in recs}:
        med[m] = np.median([len(r["lines"]) for r in recs if r["mas"] == m])
    F = []
    for r in recs:
        c = collections.Counter(norm(l) for l in r["lines"] if len(l.strip()) >= 40)
        ls = [norm(l) for l in r["lines"] if len(l.strip()) >= 40]
        F.append([max(c.values(), default=0) >= 3,
                  len(r["lines"]) > med[r["mas"]] or r["text"].count("Continue. Please keep solving") >= 2,
                  VERIF.search(r["text"]) is None,
                  bool(ls) and ls.count(ls[0]) >= 2,
                  ASSUME.search(r["text"]) is not None])
    return np.array(F, float)


def rates(F, L, idx):
    K, n = F.shape[1], L.shape[1]
    T = np.zeros((K, n)); P = np.zeros((K, n)); npos = np.zeros(n); nneg = np.zeros(n)
    for m in range(n):
        pos = [i for i in idx if L[i, m] == 1]; neg = [i for i in idx if L[i, m] == 0]
        npos[m], nneg[m] = len(pos), len(neg)
        if pos: T[:, m] = F[pos].mean(0)
        if neg: P[:, m] = F[neg].mean(0)
    return T, P, npos, nneg


def game(A):
    K, n = A.shape
    r = linprog(np.r_[np.zeros(K), -1.0], A_ub=np.hstack([-A.T, np.ones((n, 1))]), b_ub=np.zeros(n),
                A_eq=np.r_[np.ones(K), 0.0][None], b_eq=[1], bounds=[(0, None)] * K + [(None, None)])
    return max(0.0, float(-r.fun)), r.x[:K]


def cma(T, P, npos, nneg):
    K, n = T.shape
    lg = np.log(4 * K * n / DELTA)
    r = np.sqrt(lg / (2 * npos)) + np.sqrt(lg / (2 * nneg))
    vL, pL = game(np.clip(T - P - r[None], 0, None))
    vU, _ = game(np.clip(T - P + r[None], 0, 1))
    return vL, vU, (pL if vL > 1e-12 else np.ones(K) / K), float(r.max())


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python x0_mad_full.py path/to/MAD_full_dataset.json")
    recs = load(sys.argv[1])
    L_all = np.array([r["lab"] for r in recs])
    keep = [j for j in range(len(CODES)) if np.nansum(L_all[:, j] == 1) >= MIN_POS and np.nansum(L_all[:, j] == 0) >= MIN_POS]
    modes = [CODES[j] for j in keep]
    L = L_all[:, keep]
    F = flags(recs)
    N = len(recs)
    T, P, npos, nneg = rates(F, L, range(N))
    J = np.clip(T - P, 0, None)
    lam, pi = game(J)
    vL, vU, pL, rad = cma(T, P, npos, nneg)
    out = dict(n_traces=N, by_framework=dict(collections.Counter(r["mas"] for r in recs)), modes=modes,
               positives=npos.astype(int).tolist(), negatives=nneg.astype(int).tolist(),
               verifiers=NAMES, TPR=np.round(T, 3).tolist(), FPR=np.round(P, 3).tolist(), J=np.round(J, 3).tolist(),
               single_worst={NAMES[k]: round(float(J[k].min()), 3) for k in range(len(NAMES))},
               lambda_hat=round(lam, 3), pi_hat=np.round(pi, 3).tolist(),
               uniform_worst=round(float((np.ones(len(NAMES)) / len(NAMES) @ J).min()), 3),
               lambda_LCB=round(vL, 3), lambda_UCB=round(vU, 3), max_radius=round(rad, 3), delta=DELTA)
    # held-out evaluation over half splits
    res = collections.defaultdict(list)
    for _ in range(300):
        perm = rng.permutation(N); cal, test = perm[: N // 2], perm[N // 2:]
        t1, f1, p1, n1 = rates(F, L, cal); t2, f2, p2, n2 = rates(F, L, test)
        if (p1 == 0).any() or (p2 == 0).any() or (n1 == 0).any() or (n2 == 0).any():
            continue
        Jc, Jt = np.clip(t1 - f1, 0, None), np.clip(t2 - f2, 0, None)
        fail = [i for i in cal if np.nansum(L[i]) > 0]
        k_avg = int(np.argmax(F[fail].mean(0)))
        K = len(NAMES)
        for nm, w in {"uniform": np.ones(K) / K, "best single (avg. detection)": np.eye(K)[k_avg],
                      "minimax (plug-in)": game(Jc)[1], "minimax (LCB, Alg. 3)": cma(t1, f1, p1, n1)[2]}.items():
            res[nm + "|worst"].append(float((w @ Jt).min())); res[nm + "|avg"].append(float((w @ Jt).mean()))
    out["splits"] = {k: dict(mean=round(float(np.mean(v)), 3), sd=round(float(np.std(v)), 3)) for k, v in res.items()}
    d = Path("results/mad_full"); d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(out, indent=1))
    rows = ["\\begin{tabular}{@{}l" + "c" * len(modes) + "@{}}", "\\toprule",
            " & " + " & ".join(f"FM-{m}" for m in modes) + "\\\\", "\\midrule"]
    rows.append("pos./neg. & " + " & ".join(f"${int(a)}/{int(b)}$" for a, b in zip(npos, nneg)) + "\\\\\n\\midrule")
    for nm, row in zip(NAMES, J):
        rows.append(nm + " & " + " & ".join(f"${v:.3f}$" if v > 0 else "$0$" for v in row) + "\\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    (d / "tables.tex").write_text("\n".join(rows))
    print(json.dumps({k: out[k] for k in ("n_traces", "modes", "single_worst", "lambda_hat", "lambda_LCB",
                                          "lambda_UCB", "uniform_worst")}, indent=1))
    print("splits:", out["splits"])
    print("Wrote results/mad_full/summary.json and tables.tex")


if __name__ == "__main__":
    main()
