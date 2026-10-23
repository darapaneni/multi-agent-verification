# Author: Narayana Darapaneni
# Code for "Why Verification Fails in Multi-Agent LLM Systems" (2026). MIT License.
"""E5: verifier soundness and portfolio allocation on real, human-annotated MAST traces.

Data: the human-annotated traces released in the MAST repository
(https://github.com/multi-agent-systems-failure-taxonomy/MAST, folder traces/, files *_human.json;
AG2 math traces and HyperAgent SWE-bench traces).  Usage:  python3 exp_real.py /path/to/MAST/traces
"""
import sys, re, json, glob, collections, itertools
import numpy as np
from scipy.optimize import linprog

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/MAST/traces"
rng = np.random.default_rng(7)

# ---- mapping of the annotation labels used in the released human traces to MAST modes ----
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

def load():
    recs = []
    for f in sorted(glob.glob(f"{ROOT}/**/*_human.json", recursive=True)):
        d = json.load(open(f))
        opts = d["note"]["options"]
        lab = {}
        for m, keys in MODEMAP.items():
            present = [k for k in keys if k in opts]
            lab[m] = None if not present else any(str(opts[k]).lower().startswith("yes") for k in present)
        fw = "AG2" if "/AG2/" in f else "HyperAgent"
        if fw == "AG2":
            msgs = ["\n".join(t["content"]) if isinstance(t["content"], list) else str(t["content"])
                    for t in d["trajectory"]]
            lines = [l for m_ in msgs[1:] for l in m_.split("\n")]
        else:
            lines = [l.split(" - INFO - ", 1)[-1] for l in d["trajectory"] if isinstance(l, str)]
        recs.append(dict(fw=fw, lab=lab, lines=lines, text="\n".join(lines), other=d.get("other_data", {})))
    return recs

def norm(l): return re.sub(r"\s+", " ", l.strip().lower())

# ---- six programmatic verifiers (deterministic, no LLM calls) ----
def v_repetition(r):           # V1: loop / repetition monitor
    c = collections.Counter(norm(l) for l in r["lines"] if len(l.strip()) >= 40)
    return max(c.values(), default=0) >= 3
def v_budget(r, med):          # V2: termination / budget monitor
    return len(r["lines"]) > med[r["fw"]] or r["text"].count("Continue. Please keep solving") >= 2
VERIF = re.compile(r"verif|double-check|check (the|our|if|that)|substitut|tests? pass|pytest|run(ning)? (the )?test|reproduc", re.I)
def v_no_evidence(r):          # V3: verification-evidence check (flags when no verification activity)
    return VERIF.search(r["text"]) is None
def v_restart(r):              # V4: restart detector (first substantive line re-appears)
    ls = [norm(l) for l in r["lines"] if len(l.strip()) >= 40]
    return bool(ls) and ls.count(ls[0]) >= 2
ASSUME = re.compile(r"\bassum|insufficient|not enough information|missing information|unclear|ambigu", re.I)
def v_assumption(r):           # V5: assumption / ambiguity detector
    return ASSUME.search(r["text"]) is not None
def v_reference(r):            # V6: objective-level reference check (only where a reference answer exists)
    return r["fw"] == "AG2" and r["other"].get("correct") is False
VNAMES = ["V1 repetition monitor", "V2 budget/termination monitor", "V3 verification-evidence check",
          "V4 restart detector", "V5 assumption detector", "V6 reference-answer check"]

def flags(recs):
    med = {fw: np.median([len(r["lines"]) for r in recs if r["fw"] == fw]) for fw in ("AG2", "HyperAgent")}
    F = np.array([[v_repetition(r), v_budget(r, med), v_no_evidence(r), v_restart(r), v_assumption(r),
                   v_reference(r)] for r in recs], float)
    return F

def rates(F, recs, idx):
    """TPR, FPR, counts per (verifier, mode) on the traces idx."""
    K, n = F.shape[1], len(MODES)
    TPR = np.zeros((K, n)); FPR = np.zeros((K, n)); npos = np.zeros(n); nneg = np.zeros(n)
    for j, m in enumerate(MODES):
        pos = [i for i in idx if recs[i]["lab"][m] is True]
        neg = [i for i in idx if recs[i]["lab"][m] is False]
        npos[j], nneg[j] = len(pos), len(neg)
        TPR[:, j] = F[pos].mean(0) if pos else 0
        FPR[:, j] = F[neg].mean(0) if neg else 0
    return TPR, FPR, npos, nneg

def game(A):
    K, n = A.shape
    c = np.r_[np.zeros(K), -1.0]
    r = linprog(c, A_ub=np.hstack([-A.T, np.ones((n, 1))]), b_ub=np.zeros(n),
                A_eq=np.r_[np.ones(K), 0.0][None], b_eq=[1], bounds=[(0, None)] * K + [(None, None)])
    return -r.fun, r.x[:K]

def worst(pi, A): return float((pi @ A).min())

recs = load(); F = flags(recs); N = len(recs); K = F.shape[1]
all_idx = list(range(N))
TPR, FPR, npos, nneg = rates(F, recs, all_idx)
J = np.clip(TPR - FPR, 0, None)
lam_hat, pi_hat = game(J)

# certified bounds (Theorem 4.4): Hoeffding radii with delta split over 4Kn events
delta = 0.1
R = np.sqrt(np.log(4 * K * len(MODES) / delta) / (2 * np.maximum(npos, 1)))[None] + \
    np.sqrt(np.log(4 * K * len(MODES) / delta) / (2 * np.maximum(nneg, 1)))[None]
R = np.repeat(R, K, axis=0)
L = np.clip(TPR - FPR - R, 0, None); U = np.clip(TPR - FPR + R, 0, None)
lam_L, pi_L = game(L); lam_U, _ = game(U)
need = 8 * np.log(4 * K * len(MODES) / delta) / 0.1 ** 2   # per-cell sample size for eps = 0.1

# repeated half splits: calibrate on one half, evaluate worst-mode informedness on the other
res = collections.defaultdict(list)
for rep in range(500):
    perm = rng.permutation(N); cal, test = perm[: N // 2], perm[N // 2:]
    t1, f1, p1, n1 = rates(F, recs, cal); t2, f2, p2, n2 = rates(F, recs, test)
    if (p1 == 0).any() or (p2 == 0).any(): continue
    Jc, Jt = np.clip(t1 - f1, 0, None), np.clip(t2 - f2, 0, None)
    # average-detection learner: verifier with largest mean flag rate on failing traces (what Exp3 tracks)
    fail_cal = [i for i in cal if any(recs[i]["lab"][m] for m in MODES if recs[i]["lab"][m] is not None)]
    k_avg = int(np.argmax(F[fail_cal].mean(0)))
    Rc = np.sqrt(np.log(4 * K * len(MODES) / delta) / (2 * p1))[None] + \
         np.sqrt(np.log(4 * K * len(MODES) / delta) / (2 * np.maximum(n1, 1)))[None]
    vL, pL = game(np.clip(t1 - f1 - Rc, 0, None))
    pi_lcb = pL if vL > 1e-12 else np.ones(K) / K          # certified rule; uniform when nothing is certified
    pis = {"uniform": np.ones(K) / K, "best single (avg. detection)": np.eye(K)[k_avg],
           "minimax (plug-in)": game(Jc)[1], "minimax (LCB, Alg. 3)": pi_lcb}
    for name, pi in pis.items():
        res[name + "|worst"].append(worst(pi, Jt))
        res[name + "|avg"].append(float((pi @ Jt).mean()))

OUT = dict(
    n_traces=N, by_framework=dict(collections.Counter(r["fw"] for r in recs)),
    modes=MODES, verifiers=VNAMES,
    positives={m: int(p) for m, p in zip(MODES, npos)}, negatives={m: int(q) for m, q in zip(MODES, nneg)},
    TPR=np.round(TPR, 3).tolist(), FPR=np.round(FPR, 3).tolist(), J=np.round(J, 3).tolist(),
    lambda_hat=round(float(lam_hat), 3), pi_hat=np.round(pi_hat, 3).tolist(),
    lambda_LCB=round(float(lam_L), 3), lambda_UCB=round(float(lam_U), 3), delta=delta,
    max_radius=round(float(R.max()), 3), cells_needed_eps01=int(np.ceil(need)),
    single_worst={VNAMES[k]: round(float(J[k].min()), 3) for k in range(K)},
    uniform_worst=round(worst(np.ones(K) / K, J), 3),
    splits={k: dict(mean=round(float(np.mean(v)), 3), sd=round(float(np.std(v)), 3)) for k, v in res.items()},
    n_splits=len(res["uniform|worst"]),
)
json.dump(OUT, open("results_real.json", "w"), indent=1)
print(json.dumps(OUT, indent=1))
