# Author: Narayana Darapaneni
# Code for "Why Verification Fails in Multi-Agent LLM Systems" (2026). MIT License.
"""Simulations E1 to E4 for 'Why Verification Fails in Multi-Agent LLM Systems'.
Produces the figures (PDF) and a JSON file of the numbers quoted in the paper."""
import json, numpy as np
from scipy.optimize import minimize, linprog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 150, "pdf.fonttype": 42})
OUT = {}
rng = np.random.default_rng(2026)

# ---------------- l_p geometry helpers ----------------
def Jp(x, p):
    n = np.linalg.norm(x, p)
    if n == 0: return np.zeros_like(x)
    return n ** (2 - p) * np.sign(x) * np.abs(x) ** (p - 1)

def bregman_D(y, x, p):
    return 0.5 * np.linalg.norm(y, p) ** 2 - 0.5 * np.linalg.norm(x, p) ** 2 - Jp(x, p) @ (y - x)

def bregman_halfspace(x, phi, c, p):
    """Bregman projection of x onto {y: phi.y <= c} for h = 0.5||.||_p^2."""
    q = p / (p - 1)
    if phi @ x <= c: return x.copy()
    xi = Jp(x, p)
    f = lambda a: phi @ Jp(xi - a * phi, q) - c
    lo, hi = 0.0, 1.0
    while f(hi) > 0: hi *= 2
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0: lo = mid
        else: hi = mid
    return Jp(xi - hi * phi, q)

def dist_affine(x, s, B, p):
    """d(x, s + span B) in l_p."""
    z = x - s
    c0 = np.linalg.lstsq(B, z, rcond=None)[0]
    r = minimize(lambda c: np.linalg.norm(z - B @ c, p), c0, method="BFGS", options={"gtol": 1e-10})
    return min(r.fun, np.linalg.norm(z - B @ c0, p))

# ---------------- E1: verify-repair dynamics ----------------
# Specification S = s + M with M = span{e_{d+1},...,e_n}; then X/M is l_p^d isometrically and
# d(x,S) = ||(x-s)_{1:d}||_p.  Verifiers: the 2d signed coordinate functionals plus K random
# functionals vanishing on M, all normalised in the dual norm l_q.  Because conv{+-e_i} is the unit
# ball of l_1^d, which contains d^{-1/p} B_{l_q^d}, Theorem 2.3 gives the certified global bound
# lambda_S(Phi) >= d^{-1/p}.  An upper bound on lambda_S(Phi) is obtained by local minimisation of the
# ratio d_Phi/d over many starting points (any point gives a valid upper bound on the infimum).
n, m, K = 30, 5, 400
d = n - m
s_star = rng.standard_normal(n)
x0 = s_star + 4 * rng.standard_normal(n)

def dist_S(x, p): return np.linalg.norm((x - s_star)[:d], p)

def make_verifiers(p, K, blind=False):
    q = p / (p - 1)
    G = rng.standard_normal((K, n)); G[:, d:] = 0
    E = np.vstack([np.eye(n)[:d], -np.eye(n)[:d]])
    if blind:                                   # remove +-e_1 and annihilate e_1 everywhere
        G[:, 0] = 0; E = E[[i for i in range(2 * d) if i % d != 0]]
    Phi = np.vstack([E, G])
    Phi /= np.linalg.norm(Phi, q, axis=1, keepdims=True)
    return Phi

def modulus_upper(Phi, p, starts=300):
    Pd = Phi[:, :d]
    f = lambda z: max(np.max(Pd @ z), 0) / np.linalg.norm(z, p)
    best = np.inf
    for _ in range(starts):
        z0 = rng.standard_normal(d)
        r = minimize(f, z0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-12})
        best = min(best, r.fun, f(z0))
    return best

def run_repair(p, Phi, T=300):
    x = x0.copy(); ds, dphis = [], []
    for t in range(T):
        viol = Phi @ (x - s_star)
        ds.append(dist_S(x, p)); dphis.append(max(viol.max(), 0))
        k = int(np.argmax(viol))
        if viol[k] > 0:
            x = bregman_halfspace(x, Phi[k], Phi[k] @ s_star, p)
    return np.array(ds), np.array(dphis)

kappa = {1.25: 0.25, 1.5: 0.5, 2.0: 1.0}
res = {}
for p in [1.25, 1.5, 2.0]:
    Phi = make_verifiers(p, K)
    ds, dph = run_repair(p, Phi)
    mask = ds > 1e-9
    lam_traj = float(np.min(dph[mask] / ds[mask]))
    lam_lb = d ** (-1 / p)
    lam_ub = float(modulus_upper(Phi, p))
    D0 = bregman_D(s_star, x0, p)
    T = np.arange(1, len(ds) + 1)
    bound = np.sqrt(2 * D0 / (kappa[p] * lam_lb ** 2 * T))
    runmin = np.minimum.accumulate(ds)
    res[p] = dict(ds=ds, dph=dph, bound=bound)
    OUT[f"E1_p{p}"] = dict(lambda_certified_lower=lam_lb, lambda_upper=lam_ub, lambda_traj=lam_traj,
                           D0=float(D0), d0=float(ds[0]), d_final=float(ds[-1]),
                           bound_holds=bool(np.all(runmin <= bound + 1e-9)))

# blind spot (Hilbert): no verifier sees e_1, so lambda_S(Phi) = 0 by Corollary 3.2
Phi_b = make_verifiers(2.0, K, blind=True)
ds_b, dph_b = run_repair(2.0, Phi_b)
OUT["E1_blind"] = dict(d_final=float(ds_b[-1]), dphi_final=float(dph_b[-1]), d0=float(ds_b[0]),
                       blind_component=float(abs((x0 - s_star)[0])))

# termination times vs epsilon (p = 1.5)
p = 1.5; Phi = make_verifiers(p, K)
eps_list = np.array([2.0, 1.0, 0.5, 0.25, 0.125, 0.0625])
Teps, bnd = [], []
D0 = bregman_D(s_star, x0, p)
for e in eps_list:
    x = x0.copy(); t = 0
    while True:
        viol = Phi @ (x - s_star); k = int(np.argmax(viol))
        if viol[k] <= e: break
        x = bregman_halfspace(x, Phi[k], Phi[k] @ s_star, p); t += 1
    Teps.append(t); bnd.append(2 * D0 / (kappa[p] * e ** 2))
OUT["E1_termination"] = dict(eps=eps_list.tolist(), T=Teps, bound=[float(b) for b in bnd])

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
cols = {1.25: "#3b6ea8", 1.5: "#c0504d", 2.0: "#4f9a5b"}
for p in [1.25, 1.5, 2.0]:
    ax[0].semilogy(res[p]["ds"], color=cols[p], label=fr"$\ell_{{{p:g}}}$")
    ax[0].semilogy(res[p]["bound"], color=cols[p], ls=":", lw=0.9)
ax[0].set_xlabel("repair round $t$"); ax[0].set_ylabel(r"$d(x_t,S)$")
ax[0].set_title("(a) verify-repair, certified bound", fontsize=9); ax[0].legend(fontsize=7)
ax[1].semilogy(ds_b, color="#333333", label=r"$d(x_t,S)$")
ax[1].semilogy(np.maximum(dph_b, 1e-12), color="#c0504d", ls="--", label=r"$d_\Phi(x_t,S)$")
ax[1].set_xlabel("repair round $t$"); ax[1].set_title("(b) blind spot, $\\lambda_S(\\Phi)=0$", fontsize=9)
ax[1].legend(fontsize=7)
ax[2].loglog(eps_list, Teps, "o-", color="#3b6ea8", label="observed $T_\\varepsilon$")
ax[2].loglog(eps_list, bnd, ":", color="#333333", label="bound (Thm 5.5b)")
ax[2].set_xlabel(r"tolerance $\varepsilon$"); ax[2].set_title("(c) termination time", fontsize=9)
ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig("fig_repair.pdf"); plt.close(fig)

# ---------------- E2: coordination gain ----------------
gain = {}
for pp in [1, 1.5, 2, 4]:
    gain[str(pp)] = {str(k): float(k ** (-1 / pp)) for k in [1, 2, 4, 8, 16]}
# numerical confirmation for p = 1, k = 4 blocks of R^3 Euclidean, Phi_j = lambda * ball
k, lam = 4, 0.8
xs = rng.standard_normal((20000, k, 3))
nb = np.linalg.norm(xs, axis=2)
sep = lam * nb.max(axis=1) / nb.sum(axis=1)                 # separate checks, l_1 sum
OUT["E2"] = dict(theory=gain, sep_min_ratio_p1_k4=float(sep.min()), predicted=lam / k)

# ---------------- E3: best-of-N with and without blind spot ----------------
eta, Ns, R = 0.5, [1, 2, 4, 8, 16, 32, 64, 128], 20000
succ = {"oracle": [], "full": [], "noisy": [], "blind": []}
for N in Ns:
    V = rng.standard_normal((R, N)); W = rng.standard_normal((R, N))
    d = np.sqrt(V ** 2 + W ** 2)
    dphi_full = np.maximum(np.abs(V), np.abs(W))           # Phi = {+-e1, +-e2}, lambda = 1/sqrt2
    noisy = dphi_full + rng.uniform(-0.1, 0.1, size=V.shape)
    blind = np.abs(V)                                       # sees only first coordinate
    idx = lambda s: d[np.arange(R), s.argmin(axis=1)]
    succ["oracle"].append(float(np.mean(d.min(axis=1) <= eta)))
    succ["full"].append(float(np.mean(idx(dphi_full) <= eta)))
    succ["noisy"].append(float(np.mean(idx(noisy) <= eta)))
    succ["blind"].append(float(np.mean(idx(blind) <= eta)))
pW = float(np.mean(np.abs(rng.standard_normal(10 ** 6)) <= eta))
OUT["E3"] = dict(N=Ns, **succ, blind_ceiling=pW, eta=eta)
fig, ax = plt.subplots(figsize=(3.4, 2.3))
for key, c, lab in [("oracle", "#333333", "oracle selection"), ("full", "#4f9a5b", r"$\lambda=1/\sqrt{2}$"),
                    ("noisy", "#3b6ea8", r"$\lambda=1/\sqrt{2}$, $\beta=0.1$"), ("blind", "#c0504d", r"blind, $\lambda=0$")]:
    ax.semilogx(Ns, succ[key], "o-", ms=3, color=c, label=lab)
ax.axhline(pW, color="#c0504d", ls=":", lw=0.9)
ax.set_xlabel("number of candidates $N$"); ax.set_ylabel(r"$\Pr[d(\hat x,S)\leq\eta]$")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("fig_bestofN.pdf"); plt.close(fig)

# ---------------- E4: verifier allocation calibrated on MAST ----------------
modes = ["1.1","1.2","1.3","1.4","1.5","2.1","2.2","2.3","2.4","2.5","2.6","3.1","3.2","3.3"]
# Table 6 of the MAST paper (open-source models, 100 traces each)
T6 = {"Qwen/ChatDev":   [35,4,96,1,94,2,1,9,0,2,20,1,16,12],
      "Qwen/MetaGPT":   [12,1,35,0,3,0,4,0,0,12,16,47,51,32],
      "CodeLlama/ChatDev":[76,45,97,46,97,50,16,76,2,42,93,25,67,69],
      "CodeLlama/MetaGPT":[94,12,99,23,76,9,15,57,0,40,18,26,55,56]}
fig1 = [11.8,1.5,15.7,2.8,12.4,2.2,6.8,7.4,0.8,1.9,13.2,6.2,8.2,9.1]   # Figure 1 prevalences
verifiers = ["V1 static/compile", "V2 unit tests", "V3 role auditor", "V4 loop/termination monitor",
             "V5 cross-agent consistency", "V6 objective-level check"]
A = np.zeros((6, 14)); ix = {mm: i for i, mm in enumerate(modes)}
def setrow(r, d):
    for kk, v in d.items(): A[r, ix[kk]] = v
setrow(0, {"1.1": .2, "3.2": .3, "3.3": .2})
setrow(1, {"1.1": .6, "3.2": .7, "3.3": .6, "2.6": .3, "3.1": .4})
setrow(2, {"1.2": .8, "2.4": .5, "2.5": .6, "2.2": .4, "2.1": .3})
setrow(3, {"1.3": .9, "1.5": .8, "1.4": .4, "2.1": .5, "3.1": .5})
setrow(4, {"2.6": .7, "2.3": .6, "2.5": .4, "2.2": .4, "1.4": .5})
setrow(5, {"1.1": .7, "2.3": .7, "3.1": .6, "3.2": .5, "3.3": .5, "1.5": .3})

def game_value(A):
    Kk, nn = A.shape
    # max t s.t. (pi^T A)_m >= t, pi in simplex. variables (pi, t)
    c = np.zeros(Kk + 1); c[-1] = -1
    A_ub = np.hstack([-A.T, np.ones((nn, 1))]); b_ub = np.zeros(nn)
    A_eq = np.hstack([np.ones((1, Kk)), np.zeros((1, 1))])
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1], bounds=[(0, None)] * Kk + [(None, None)])
    return -r.fun, r.x[:Kk]

lib = {"{V1}": [0], "{V1,V2}": [0, 1], "{V1,V2,V6}": [0, 1, 5], "{V1,...,V5}": [0, 1, 2, 3, 4],
       "{V1,...,V6}": [0, 1, 2, 3, 4, 5]}
OUT["E4_library"] = {}
for name, rows in lib.items():
    v, pi = game_value(A[rows]); OUT["E4_library"][name] = dict(value=round(float(v), 4),
                                                              pi=[round(float(z), 3) for z in pi])
lam_star, pi_star = game_value(A)
OUT["E4_lambda_star"] = float(lam_star); OUT["E4_pi_star"] = pi_star.tolist()

def exp3(Tn, mode_seq, A, gamma, rng):
    Kk = A.shape[0]; w = np.ones(Kk); tot = 0.0; hist = []
    for t in range(Tn):
        p = (1 - gamma) * w / w.sum() + gamma / Kk
        k = rng.choice(Kk, p=p)
        r = float(rng.random() < A[k, mode_seq[t]])
        tot += r; hist.append(tot / (t + 1))
        w[k] *= np.exp(gamma * (r / p[k]) / Kk); w /= w.max()
    return np.array(hist)

def hedge(Tn, mode_seq, A, eta, rng):
    Kk = A.shape[0]; L = np.zeros(Kk); tot = 0.0; hist = []
    for t in range(Tn):
        p = np.exp(eta * (L - L.max())); p /= p.sum()
        det = (rng.random(Kk) < A[:, mode_seq[t]]).astype(float)
        tot += p @ det; L += det; hist.append(tot / (t + 1))
    return np.array(hist)

Tn = 5000
OUT["E4_runs"] = {}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
for j, (name, counts) in enumerate([("MAST Figure 1", fig1), ("CodeLlama/ChatDev", T6["CodeLlama/ChatDev"])]):
    qd = np.array(counts, float); qd /= qd.sum()
    seq = rng.choice(14, size=Tn, p=qd)
    gam = min(1, np.sqrt(6 * np.log(6) / ((np.e - 1) * Tn)))
    e3L = [exp3(Tn, seq, A, gam, np.random.default_rng(s)) for s in range(20)]
    hdL = [hedge(Tn, seq, A, np.sqrt(8 * np.log(6) / Tn), np.random.default_rng(s)) for s in range(20)]
    e3 = np.mean(e3L, axis=0); hd = np.mean(hdL, axis=0)
    OUT.setdefault("E4_sd", {})[name] = dict(exp3_sd=float(np.std([v[-1] for v in e3L])), hedge_sd=float(np.std([v[-1] for v in hdL])))
    best = (A @ qd).max(); unif = (A @ qd).mean()
    OUT["E4_runs"][name] = dict(exp3_final=float(e3[-1]), hedge_final=float(hd[-1]), best_single=float(best),
                                uniform=float(unif), lambda_star=float(lam_star),
                                regret_term=float(2 * np.sqrt(np.e - 1) * np.sqrt(6 * np.log(6) / Tn)))
    ax[j].plot(e3, color="#3b6ea8", label="Exp3 (bandit)")
    ax[j].plot(hd, color="#4f9a5b", label="Hedge (full info)")
    ax[j].axhline(best, color="#333333", ls=":", lw=0.9, label="best single verifier")
    ax[j].axhline(lam_star, color="#c0504d", ls="--", lw=0.9, label=r"game value $\lambda^\star$")
    ax[j].axhline(unif, color="#999999", ls="-.", lw=0.9, label="uniform allocation")
    ax[j].set_ylim(0.1, 0.34); ax[j].set_xlabel("trace index $t$"); ax[j].set_title(name, fontsize=9)
ax[0].set_ylabel("average detection rate")
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=5, fontsize=7)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("fig_allocation.pdf"); plt.close(fig)

# adversarial (oblivious) sequence: best response to uniform play repeated
# ---- E4b: average detection is not worst-case soundness (Exp3 final allocation) ----
def exp3_pi(Tn, mode_seq, A, gamma, rng):
    Kk = A.shape[0]; w = np.ones(Kk); cnt = np.zeros(Kk)
    for t in range(Tn):
        p = (1 - gamma) * w / w.sum() + gamma / Kk
        k = rng.choice(Kk, p=p); r = float(rng.random() < A[k, mode_seq[t]])
        cnt[k] += 1
        w[k] *= np.exp(gamma * (r / p[k]) / Kk); w /= w.max()
    return cnt / Tn                      # empirical allocation (fraction of rounds per verifier)
qd = np.array(fig1, float); qd /= qd.sum()
seq = rng.choice(14, size=Tn, p=qd); gam = min(1, np.sqrt(6 * np.log(6) / ((np.e - 1) * Tn)))
pis = [exp3_pi(Tn, seq, A, gam, np.random.default_rng(100 + s_)) for s_ in range(20)]
OUT["E4b_sd"] = float(np.std([(pp @ A).min() for pp in pis]))
OUT["E4b_exp3_worstcase"] = dict(mean_lambda_of_learned_pi=float(np.mean([(pp @ A).min() for pp in pis])),
                                 mean_pi=np.round(np.mean(pis, axis=0), 3).tolist(),
                                 lambda_star=float(lam_star))
# two-by-two counterexample of the review: identity matrix, all failures of mode 1
A2 = np.eye(2); seq2 = np.zeros(Tn, int)
p2 = exp3_pi(Tn, seq2, A2, min(1, np.sqrt(2 * np.log(2) / ((np.e - 1) * Tn))), np.random.default_rng(5))
OUT["E4b_identity"] = dict(learned_pi=np.round(p2, 3).tolist(), lambda_learned=float((p2 @ A2).min()),
                           lambda_star=0.5)

# ---- E4c: certified minimax allocation (Algorithm 3) with N labelled samples per cell ----
TPRt = np.minimum(1, A + 0.05); FPRt = np.full_like(A, 0.05); Jt = np.clip(TPRt - FPRt, 0, None)
lam_true, _ = game_value(Jt)
delta = 0.1; Kk, nn = A.shape; Ns_c = [25, 100, 400, 1600, 6400]
cert = {k_: [] for k_ in ["N", "lam_pihat", "lam_plugin", "LCB", "UCB", "cover_pi", "cover_star"]}
for Nc in Ns_c:
    rows = []
    for rep in range(200):
        tp = rng.binomial(Nc, TPRt) / Nc; fp = rng.binomial(Nc, FPRt) / Nc
        r = 2 * np.sqrt(np.log(4 * Kk * nn / delta) / (2 * Nc))
        L = np.clip(tp - fp - r, 0, None); U = np.clip(tp - fp + r, 0, None)
        vL, pL = game_value(L); vU, _ = game_value(U); _, pP = game_value(np.clip(tp - fp, 0, None))
        lp = (pL @ Jt).min() if vL > 1e-12 else (np.ones(Kk) / Kk @ Jt).min()
        rows.append([lp, (pP @ Jt).min(), vL, vU, lp >= vL - 1e-12, vL - 1e-12 <= lam_true <= vU + 1e-12])
    rows = np.array(rows, float)
    for k_, v_ in zip(["lam_pihat", "lam_plugin", "LCB", "UCB", "cover_pi", "cover_star"], rows.mean(0)):
        cert[k_].append(float(v_))
    cert["N"].append(Nc)
OUT["E4c_certified"] = dict(lambda_star=float(lam_true), delta=delta, **cert)
fig, ax = plt.subplots(figsize=(3.6, 2.4))
ax.semilogx(Ns_c, cert["lam_pihat"], "o-", color="#3b6ea8", ms=3, label=r"$\lambda(\hat\pi)$ (Alg. 3)")
ax.semilogx(Ns_c, cert["lam_plugin"], "s--", color="#999999", ms=3, label="plug-in minimax")
ax.semilogx(Ns_c, cert["LCB"], "v-", color="#c0504d", ms=3, label="certified LCB")
ax.semilogx(Ns_c, cert["UCB"], "^-", color="#4f9a5b", ms=3, label="UCB")
ax.axhline(lam_true, color="#333333", ls=":", lw=0.9, label=r"$\lambda^\star$")
ax.set_xlabel("labelled traces per cell $N$"); ax.set_ylabel("worst-mode informedness")
ax.set_ylim(0, 1); ax.legend(fontsize=6.5, ncol=2)
fig.tight_layout(); fig.savefig("fig_certify.pdf"); plt.close(fig)

json.dump(OUT, open("results.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in OUT.items() if k != "E2"}, indent=1, default=float)[:4000])
print(OUT["E2"]["sep_min_ratio_p1_k4"], OUT["E2"]["predicted"])
