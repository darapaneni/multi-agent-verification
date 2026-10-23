# Author: Narayana Darapaneni
# Code for "Why Verification Fails in Multi-Agent LLM Systems" (2026). MIT License.
"""Uncertainty and robustness analysis for E5 (61 human-annotated MAST traces) and multi-seed
repeats for E1 and E4.  Usage: python3 exp_robust.py /path/to/MAST/traces
Writes results_robust.json."""
import sys, re, json, glob, collections, itertools
import numpy as np
from scipy.optimize import linprog, minimize

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/MAST/traces"
OUT = {}

# ------------------------------------------------------------------ E5 data (as in exp_real.py)
MODEMAP = {"FM-1.3": ["Step repetition"], "FM-2.1": ["Trajectory restart"],
           "FM-2.2": ["Fail to detect ambiguities/contradictions", "Proceed with incorrect assumptions",
                      "Fail to elicit clarification"],
           "FM-2.3": ["Derailing from task objectives"], "FM-3.2": ["No attempt to verify outcome"],
           "FM-3.3": ["Evaluator agent fails to be critical"]}
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


VERIF = re.compile(r"verif|double-check|check (the|our|if|that)|substitut|tests? pass|pytest|run(ning)? (the )?test|reproduc", re.I)
ASSUME = re.compile(r"\bassum|insufficient|not enough information|missing information|unclear|ambigu", re.I)


def flags(recs, minlen=40, reps=3, budget_mult=1.0, restart_reps=2):
    med = {fw: np.median([len(r["lines"]) for r in recs if r["fw"] == fw]) for fw in ("AG2", "HyperAgent")}
    out = []
    for r in recs:
        c = collections.Counter(norm(l) for l in r["lines"] if len(l.strip()) >= minlen)
        ls = [norm(l) for l in r["lines"] if len(l.strip()) >= minlen]
        out.append([max(c.values(), default=0) >= reps,
                    len(r["lines"]) > budget_mult * med[r["fw"]] or r["text"].count("Continue. Please keep solving") >= 2,
                    VERIF.search(r["text"]) is None,
                    bool(ls) and ls.count(ls[0]) >= restart_reps,
                    ASSUME.search(r["text"]) is not None,
                    r["fw"] == "AG2" and r["other"].get("correct") is False])
    return np.array(out, float)


def labmat(recs):
    return np.array([[np.nan if r["lab"][m] is None else float(r["lab"][m]) for m in MODES] for r in recs])


def Jmat(F, L, idx):
    K = F.shape[1]; n = L.shape[1]
    T = np.zeros((K, n)); P = np.zeros((K, n)); npos = np.zeros(n); nneg = np.zeros(n)
    for m in range(n):
        pos = [i for i in idx if L[i, m] == 1]; neg = [i for i in idx if L[i, m] == 0]
        npos[m], nneg[m] = len(pos), len(neg)
        if pos: T[:, m] = F[pos].mean(0)
        if neg: P[:, m] = F[neg].mean(0)
    return np.clip(T - P, 0, None), T, P, npos, nneg


def game(A):
    K, n = A.shape
    r = linprog(np.r_[np.zeros(K), -1.0], A_ub=np.hstack([-A.T, np.ones((n, 1))]), b_ub=np.zeros(n),
                A_eq=np.r_[np.ones(K), 0.0][None], b_eq=[1], bounds=[(0, None)] * K + [(None, None)])
    return max(0.0, -r.fun), r.x[:K]


recs = load(); N = len(recs); L = labmat(recs)
F0 = flags(recs)
J0, T0, P0, npos, nneg = Jmat(F0, L, range(N))
lam0, pi0 = game(J0)

# --- bootstrap (resample traces) ---
rng = np.random.default_rng(11)
B = 2000
boot = collections.defaultdict(list)
for b in range(B):
    idx = rng.integers(0, N, N)
    # Jmat with repeated indices: build index list with repetition
    J, T, P, pp, nn = Jmat(F0, L, list(idx))
    if (pp == 0).any() or (nn == 0).any():
        continue
    lam, _ = game(J)
    boot["lam"].append(lam)
    boot["uniform"].append(float((np.ones(6) / 6 @ J).min()))
    boot["all_single_blind"].append(float(all(J[k].min() == 0 for k in range(6))))
    boot["W3_FM32"].append(float(T[2, 4] - P[2, 4]))
    boot["single_worst_max"].append(float(J.min(axis=1).max()))
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
OUT["E5_bootstrap"] = dict(B=len(boot["lam"]), lambda_hat=round(lam0, 3), lambda_ci=ci(boot["lam"]),
                           uniform_ci=ci(boot["uniform"]),
                           frac_all_single_have_zero=float(np.mean(boot["all_single_blind"])),
                           W3_FM32_informedness_point=float(T0[2, 4] - P0[2, 4]),
                           W3_FM32_ci=ci(boot["W3_FM32"]),
                           best_single_worst_ci=ci(boot["single_worst_max"]))

# --- threshold sensitivity (robustness of the qualitative conclusion) ---
sens = []
for minlen, reps, bm, rr in itertools.product([20, 40, 60], [2, 3, 4], [0.75, 1.0, 1.25], [2, 3]):
    F = flags(recs, minlen, reps, bm, rr)
    J, *_ = Jmat(F, L, range(N))
    lam, _ = game(J)
    sens.append(dict(minlen=minlen, reps=reps, budget_mult=bm, restart_reps=rr, lam=lam,
                     all_single_zero=bool(all(J[k].min() == 0 for k in range(6)))))
lams = [s["lam"] for s in sens]
OUT["E5_sensitivity"] = dict(n_settings=len(sens), lam_min=float(min(lams)), lam_max=float(max(lams)),
                             lam_median=float(np.median(lams)),
                             frac_all_single_zero=float(np.mean([s["all_single_zero"] for s in sens])))

# ------------------------------------------------------------------ E1 multi-seed (Hilbert and l_p)
def Jp(x, p):
    n_ = np.linalg.norm(x, p)
    return np.zeros_like(x) if n_ == 0 else n_ ** (2 - p) * np.sign(x) * np.abs(x) ** (p - 1)


def breg_half(x, phi, c, p):
    q = p / (p - 1)
    if phi @ x <= c: return x.copy()
    xi = Jp(x, p); f = lambda a: phi @ Jp(xi - a * phi, q) - c
    lo, hi = 0.0, 1.0
    while f(hi) > 0: hi *= 2
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return Jp(xi - hi * phi, q)


def bregD(y, x, p):
    return 0.5 * np.linalg.norm(y, p) ** 2 - 0.5 * np.linalg.norm(x, p) ** 2 - Jp(x, p) @ (y - x)


e1 = collections.defaultdict(list)
kappa = {1.25: 0.25, 1.5: 0.5, 2.0: 1.0}
for seed in range(10):
    r = np.random.default_rng(1000 + seed)
    n, m, K = 30, 5, 400; d = n - m
    s = r.standard_normal(n); x0 = s + 4 * r.standard_normal(n)
    for p in (1.25, 1.5, 2.0):
        q = p / (p - 1)
        G = r.standard_normal((K, n)); G[:, d:] = 0
        Phi = np.vstack([np.eye(n)[:d], -np.eye(n)[:d], G])
        Phi /= np.linalg.norm(Phi, q, axis=1, keepdims=True)
        x = x0.copy(); D0 = bregD(s, x0, p); lam = d ** (-1 / p)
        ok, ds = True, []
        for t in range(300):
            viol = Phi @ (x - s); dist = np.linalg.norm((x - s)[:d], p); ds.append(dist)
            bound = np.sqrt(2 * D0 / (kappa[p] * lam ** 2 * (t + 1)))
            if min(ds) > bound + 1e-9: ok = False
            k = int(np.argmax(viol))
            if viol[k] > 0: x = breg_half(x, Phi[k], Phi[k] @ s, p)
        e1[f"p{p}_final"].append(ds[-1]); e1[f"p{p}_ok"].append(ok)
OUT["E1_seeds"] = {k: (dict(all=bool(all(v))) if k.endswith("_ok") else
                       dict(median=float(np.median(v)), max=float(np.max(v)))) for k, v in e1.items()}

json.dump(OUT, open("results_robust.json", "w"), indent=1)
print(json.dumps(OUT, indent=1))
