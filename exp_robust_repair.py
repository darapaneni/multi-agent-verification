# Author: Narayana Darapaneni
# Code for "Why Verification Fails in Multi-Agent LLM Systems" (2026). MIT License.
"""E1b: Bregman verify-repair (Hilbert case) with beta-noisy reports and inexact repairs.
Checks Theorem 3.12: termination bound (a), certification (a) and error floor (b)."""
import json, numpy as np
rng = np.random.default_rng(5)
n, m, K = 30, 5, 400; d = n - m
s = rng.standard_normal(n); x0 = s + 4 * rng.standard_normal(n)
G = rng.standard_normal((K, n)); G[:, d:] = 0
Phi = np.vstack([np.eye(n)[:d], -np.eye(n)[:d], G]); Phi /= np.linalg.norm(Phi, axis=1, keepdims=True)
lam = d ** -0.5                                   # certified lower bound on the modulus (Theorem 2.3)
dist = lambda x: np.linalg.norm((x - s)[:d])
D = lambda a, b: 0.5 * np.linalg.norm(a - b) ** 2
out = []
for beta in (0.0, 0.01, 0.1):
    for r in (0.0, 0.02, 0.1):                    # size of the repair error
        for eps in (0.0, 0.5):
            x = x0.copy(); ds = []; eta = 0.0; rho = 0.0; T = 0; stopped = False
            for t in range(400):
                g = Phi @ (x - s); ghat = g + rng.uniform(-beta, beta, size=g.shape)
                ds.append(dist(x))
                if ghat.max() <= eps: stopped = True; break
                k = int(np.argmax(ghat)); phi = Phi[k]; c = phi @ s
                v = phi @ x - c
                y = x - max(v, 0) * phi                   # exact projection onto {phi <= c}
                e = rng.standard_normal(n); e *= r / np.linalg.norm(e)
                xn = y + e                                # inexact repair
                eta = max(eta, D(s, xn) + D(xn, x) - D(s, x))   # slack in (R1) for reference s
                rho = max(rho, phi @ xn - c)              # residual in (R2)
                x = xn; T += 1
            rho = max(rho, 0.0); eta = max(eta, 0.0); D0 = D(s, x0)
            rec = dict(beta=beta, r=r, eps=eps, repairs=T, stopped=stopped, eta=eta, rho=rho,
                       final=ds[-1], min_d=min(ds))
            gam = eps - beta - rho                         # theta = 1
            if eps > 0 and gam > 0 and gam ** 2 > 2 * eta:
                rec["T_bound"] = 2 * D0 / (gam ** 2 - 2 * eta); rec["T_ok"] = T <= rec["T_bound"]
                rec["cert_bound"] = (eps + beta) / lam; rec["cert_ok"] = (not stopped) or ds[-1] <= rec["cert_bound"]
            if eps == 0:
                floor = (2 * beta + rho + np.sqrt(2 * D0 / max(T, 1) + 2 * eta)) / lam
                rec["floor_bound"] = floor; rec["floor_ok"] = min(ds) <= floor + 1e-12
            out.append(rec)
json.dump(out, open("results_robust_repair.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
for r_ in out: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r_.items()})
