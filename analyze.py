# Author: Narayana Darapaneni
# Code for "Why Verification Fails in Multi-Agent LLM Systems" (2026). MIT License.
"""Analysis of X1, X2 and X3.  Produces, in results/<phase>/:
  summary.json       all numbers reported in the paper
  tables.tex         LaTeX tables in the style of the paper
  fig_x2_bestofn.pdf best-of-N curves
  fig_x3_repair.pdf  repair-policy comparison
  cost_report.txt    spending per experiment and per model

Usage:  python analyze.py --phase test|final [--dry-run]
"""
import argparse
import collections
import json
import re

import numpy as np
from scipy.optimize import linprog

from common import (PHASES, MODES, MODELS, RESULTS, load_mast, load_jsonl)

rng = np.random.default_rng(7)


# =========================================================================== helpers
def game(A):
    K, n = A.shape
    c = np.r_[np.zeros(K), -1.0]
    r = linprog(c, A_ub=np.hstack([-A.T, np.ones((n, 1))]), b_ub=np.zeros(n),
                A_eq=np.r_[np.ones(K), 0.0][None], b_eq=[1], bounds=[(0, None)] * K + [(None, None)])
    return max(0.0, float(-r.fun)), r.x[:K]


def rates(F, lab, idx):
    """F: (traces x verifiers x modes) flags in {0,1}; lab: (traces x modes) in {1,0,nan}."""
    K, n = F.shape[1], F.shape[2]
    TPR = np.zeros((K, n)); FPR = np.zeros((K, n)); npos = np.zeros(n); nneg = np.zeros(n)
    for m in range(n):
        pos = [i for i in idx if lab[i, m] == 1]
        neg = [i for i in idx if lab[i, m] == 0]
        npos[m], nneg[m] = len(pos), len(neg)
        if pos:
            TPR[:, m] = F[pos, :, m].mean(0)
        if neg:
            FPR[:, m] = F[neg, :, m].mean(0)
    return TPR, FPR, npos, nneg


def cma(TPR, FPR, npos, nneg, delta=0.1):
    """Algorithm 3 (certified minimax allocation)."""
    K, n = TPR.shape
    lg = np.log(4 * K * n / delta)
    r = np.sqrt(lg / (2 * np.maximum(npos, 1))) + np.sqrt(lg / (2 * np.maximum(nneg, 1)))
    L = np.clip(TPR - FPR - r[None], 0, None)
    U = np.clip(TPR - FPR + r[None], 0, 1)
    vL, pL = game(L)
    vU, _ = game(U)
    if vL <= 1e-12:
        pL = np.ones(K) / K
    return vL, vU, pL, float(r.max())


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def auroc(scores, y):
    scores, y = np.asarray(scores, float), np.asarray(y, int)
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


# =========================================================================== programmatic W1-W6
def norm(l):
    return re.sub(r"\s+", " ", l.strip().lower())


VERIF = re.compile(r"verif|double-check|check (the|our|if|that)|substitut|tests? pass|pytest|"
                   r"run(ning)? (the )?test|reproduc", re.I)
ASSUME = re.compile(r"\bassum|insufficient|not enough information|missing information|unclear|ambigu", re.I)


def programmatic(recs):
    med = {fw: np.median([len(r["lines"]) for r in recs if r["fw"] == fw]) for fw in ("AG2", "HyperAgent")}
    out = []
    for r in recs:
        c = collections.Counter(norm(l) for l in r["lines"] if len(l.strip()) >= 40)
        ls = [norm(l) for l in r["lines"] if len(l.strip()) >= 40]
        out.append([max(c.values(), default=0) >= 3,
                    len(r["lines"]) > med[r["fw"]] or r["ptext"].count("Continue. Please keep solving") >= 2,
                    VERIF.search(r["ptext"]) is None,
                    bool(ls) and ls.count(ls[0]) >= 2,
                    ASSUME.search(r["ptext"]) is not None,
                    r["fw"] == "AG2" and r["other"].get("correct") is False])
    return np.array(out, float)


W_NAMES = ["W1 repetition monitor", "W2 budget/termination monitor", "W3 verification-evidence check",
           "W4 restart detector", "W5 assumption detector", "W6 reference-answer check"]


# =========================================================================== X1
def analyse_x1(phase, suffix):
    rows = load_jsonl(RESULTS / phase / f"x1_judgements{suffix}.jsonl")
    if not rows:
        return None
    recs = load_mast()
    N = len(recs)
    lab = np.array([[np.nan if r["lab"][m] is None else float(r["lab"][m]) for m in MODES] for r in recs])
    judges = list(dict.fromkeys(r["judge"] for r in rows))
    rep0 = {(r["judge"], r["idx"]): r for r in rows if r["rep"] == 0}
    cover = {j: sorted(i for (jj, i) in rep0 if jj == j) for j in judges}
    common_idx = sorted(set.intersection(*[set(v) for v in cover.values()]))
    judges = [j for j in judges if len(cover[j]) == max(len(v) for v in cover.values())] or judges
    common_idx = sorted(set.intersection(*[set(cover[j]) for j in judges]))
    out = dict(n_traces=len(common_idx), judges=judges, modes=MODES)

    # flags: LLM judges are mode-specific; unparsable answers count as "no flag"
    FL = np.zeros((N, len(judges), len(MODES)))
    parse_fail = {}
    for k, j in enumerate(judges):
        bad = 0
        for i in common_idx:
            f = rep0[(j, i)]["flags"]
            bad += not rep0[(j, i)]["parse_ok"]
            FL[i, k] = [1.0 if f.get(m) else 0.0 for m in MODES]
        parse_fail[j] = bad
    W = programmatic(recs)
    FW = np.repeat(W[:, :, None], len(MODES), axis=2)        # any-flag semantics, as in the paper
    F = np.concatenate([FL, FW], axis=1)
    names = [f"LLM {j}" for j in judges] + W_NAMES

    TPR, FPR, npos, nneg = rates(F, lab, common_idx)
    J = np.clip(TPR - FPR, 0, None)
    out["counts"] = dict(pos=npos.astype(int).tolist(), neg=nneg.astype(int).tolist())
    out["verifiers"] = names
    out["TPR"], out["FPR"], out["J"] = (np.round(TPR, 3).tolist(), np.round(FPR, 3).tolist(),
                                        np.round(J, 3).tolist())
    out["parse_failures"] = parse_fail

    # agreement with humans (accuracy and Cohen's kappa per judge and mode)
    agree = {}
    for k, j in enumerate(judges):
        row = {}
        for mi, m in enumerate(MODES):
            ii = [i for i in common_idx if not np.isnan(lab[i, mi])]
            row[m] = dict(acc=float(np.mean(FL[ii, k, mi] == lab[ii, mi])),
                          kappa=kappa(FL[ii, k, mi], lab[ii, mi]))
        agree[j] = row
    out["agreement"] = agree

    # self-consistency of repeated judgements
    sc = {}
    for j in judges:
        reps = sorted({r["rep"] for r in rows if r["judge"] == j})
        if len(reps) > 1:
            r1 = {r["idx"]: r for r in rows if r["judge"] == j and r["rep"] == reps[1]}
            same = [rep0[(j, i)]["flags"][m] == r1[i]["flags"][m] for i in common_idx if i in r1
                    for m in MODES]
            sc[j] = float(np.mean(same)) if same else None
    out["self_consistency"] = sc

    # cost per judge per trace (list price, cached calls included)
    cost = collections.defaultdict(list)
    for r in rows:
        if r["rep"] == 0:
            cost[r["judge"]].append(r["cost"])
    out["cost_per_trace"] = {j: float(np.mean(v)) for j, v in cost.items()}

    # Proposition 4.1 and Algorithm 3 for three libraries
    libs = {"LLM judges": list(range(len(judges))),
            "programmatic W1-W6": list(range(len(judges), len(names))),
            "all": list(range(len(names)))}
    out["libraries"] = {}
    for ln, rowsk in libs.items():
        v, pi = game(J[rowsk])
        vL, vU, pL, rad = cma(TPR[rowsk], FPR[rowsk], npos, nneg)
        out["libraries"][ln] = dict(
            lambda_hat=round(v, 3), pi_hat={names[k]: round(float(p), 3) for k, p in zip(rowsk, pi) if p > 1e-3},
            lambda_LCB=round(vL, 3), lambda_UCB=round(vU, 3), max_radius=round(rad, 3),
            single_worst={names[k]: round(float(J[k].min()), 3) for k in rowsk},
            uniform_worst=round(float((np.ones(len(rowsk)) / len(rowsk) @ J[rowsk]).min()), 3))

    # held-out evaluation over random half splits (library "all")
    res = collections.defaultdict(list)
    K = len(names)
    for _ in range(500):
        perm = rng.permutation(common_idx)
        cal, test = list(perm[: len(perm) // 2]), list(perm[len(perm) // 2:])
        t1, f1, p1, n1 = rates(F, lab, cal)
        t2, f2, p2, n2 = rates(F, lab, test)
        if (p1 == 0).any() or (p2 == 0).any():
            continue
        Jc, Jt = np.clip(t1 - f1, 0, None), np.clip(t2 - f2, 0, None)
        fail_cal = [i for i in cal if np.nansum(lab[i]) > 0]
        k_avg = int(np.argmax(F[fail_cal].max(axis=2).mean(0)))
        k_avgJ = int(np.argmax(Jc.mean(1)))
        pis = {"uniform": np.ones(K) / K, "best single (avg. detection)": np.eye(K)[k_avg],
               "best single (avg. informedness)": np.eye(K)[k_avgJ],
               "minimax (plug-in)": game(Jc)[1], "minimax (LCB, Alg. 3)": cma(t1, f1, p1, n1)[2]}
        for nm, pi in pis.items():
            res[nm + "|worst"].append(float((pi @ Jt).min()))
            res[nm + "|avg"].append(float((pi @ Jt).mean()))
    out["splits"] = {k: dict(mean=round(float(np.mean(v)), 3), sd=round(float(np.std(v)), 3))
                     for k, v in res.items()}
    out["n_splits"] = len(res.get("uniform|worst", []))
    return out


# =========================================================================== X2
def analyse_x2(phase, suffix, S=50):
    rows = load_jsonl(RESULTS / phase / f"x2_bestofn{suffix}.jsonl")
    if not rows:
        return None
    nmax = min(len(r["cands"]) for r in rows)
    Ns = [n for n in (1, 2, 4, 8, 16, 32) if n <= nmax]
    corr = [[c["ans"] is not None and c["ans"] == r["answer"] for c in r["cands"][:nmax]] for r in rows]
    corr = np.array(corr, float)
    p = float(corr.mean())
    sel = ["oracle", "solver", "judge", "majority", "blind", "random"]
    succ = {s: np.zeros((len(rows), len(Ns))) for s in sel}
    for pi_, r in enumerate(rows):
        c = corr[pi_]
        ans = [x["ans"] for x in r["cands"][:nmax]]
        for ni, N in enumerate(Ns):
            acc = {s: 0.0 for s in sel}
            for _ in range(S):
                sub = rng.choice(nmax, size=N, replace=False)
                acc["oracle"] += c[sub].max()
                acc["random"] += c[rng.choice(sub)]
                agree = [k for k in sub if r["solver"] is not None and ans[k] == r["solver"]]
                acc["solver"] += c[rng.choice(agree)] if agree else c[rng.choice(sub)]
                ok = [k for k in sub if r["judge"][k] == 1]
                acc["judge"] += c[rng.choice(ok)] if ok else c[rng.choice(sub)]
                votes = collections.Counter(ans[k] for k in sub if ans[k] is not None)
                if votes:
                    top = max(votes.values())
                    winner = rng.choice([a for a, v in votes.items() if v == top])
                    acc["majority"] += float(winner == r["answer"])
                b = np.array([r["blind"][k] for k in sub], float)
                best = sub[b == b.max()]
                acc["blind"] += c[rng.choice(best)]
            for s in sel:
                succ[s][pi_, ni] = acc[s] / S
    boot = lambda v: [float(np.percentile([rng.choice(v, len(v)).mean() for _ in range(1000)], q)) for q in (2.5, 97.5)]
    out = dict(n_problems=len(rows), N=Ns, base_accuracy=p,
               success={s: succ[s].mean(0).round(4).tolist() for s in sel},
               ci_at_Nmax={s: boot(succ[s][:, -1]) for s in sel})
    flat_c = corr.ravel()
    judge = np.array([r["judge"][:nmax] for r in rows], float).ravel()
    blind = np.array([r["blind"][:nmax] for r in rows], float).ravel()
    out["solver_accuracy"] = float(np.mean([r["solver"] == r["answer"] for r in rows]))
    out["judge_TPR"] = float(judge[flat_c == 1].mean()) if (flat_c == 1).any() else None
    out["judge_FPR"] = float(judge[flat_c == 0].mean()) if (flat_c == 0).any() else None
    out["blind_AUROC"] = auroc(blind, flat_c)
    out["judge_AUROC"] = auroc(judge, flat_c)
    return out


# =========================================================================== X3
def analyse_x3(phase, suffix):
    rows = load_jsonl(RESULTS / phase / f"x3_repair{suffix}.jsonl")
    if not rows:
        return None
    out = {}
    for model in dict.fromkeys(r["model"] for r in rows):
        rm = [r for r in rows if r["model"] == model]
        tasks = sorted({r["task"] for r in rm})
        init_pass = [r for r in rm if r["policy"] == "none"]
        failing = sorted({r["task"] for r in rm if r["policy"] != "none"})
        d = dict(n_tasks=len(tasks), initial_pass_rate=len(init_pass) / max(len(tasks), 1),
                 n_failing=len(failing), policies={})
        for pol in ("P0", "P1", "P2"):
            e = [r for r in rm if r["policy"] == pol]
            if not e:
                continue
            d["policies"][pol] = dict(
                episodes=len(e),
                repaired=float(np.mean([x["final_pass"] for x in e])),
                overall_pass=(len(init_pass) + sum(x["final_pass"] for x in e)) / max(len(tasks), 1),
                mean_calls=float(np.mean([x["calls"] for x in e])),
                frac_code_revisit=float(np.mean([x["revisits_code"] > 0 for x in e])),
                mean_code_revisits=float(np.mean([x["revisits_code"] for x in e])),
                frac_failset_revisit=float(np.mean([x["revisits_fail"] > 0 for x in e])),
                frac_oscillation=float(np.mean([x["oscillating"] > 0 for x in e])),
                mean_final_fail_frac=float(np.mean([x["final_fail"] / max(x["n_checks"], 1) for x in e])),
                mean_init_fail_frac=float(np.mean([x["init_fail"] / max(x["n_checks"], 1) for x in e])))
        out[model] = d
    return out


# =========================================================================== outputs
def figures(phase, x2, x3, suffix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "pdf.fonttype": 42})
    d = RESULTS / phase
    if x2:
        fig, ax = plt.subplots(figsize=(3.6, 2.5))
        sty = {"oracle": ("#333333", "o", "oracle"), "solver": ("#4f9a5b", "s", "independent solver"),
               "judge": ("#3b6ea8", "^", "LLM judge"), "majority": ("#8a6bb0", "d", "majority vote"),
               "blind": ("#c0504d", "v", "blind judge"), "random": ("#999999", ".", "random")}
        for s, (c, mk, lb) in sty.items():
            ax.semilogx(x2["N"], x2["success"][s], marker=mk, ms=3, color=c, label=lb)
        ax.axhline(x2["base_accuracy"], color="#c0504d", ls=":", lw=0.9)
        ax.set_xlabel("number of candidates $N$"); ax.set_ylabel("success rate")
        ax.set_xticks(x2["N"]); ax.set_xticklabels([str(n) for n in x2["N"]])
        ax.legend(fontsize=6.5, ncol=2); fig.tight_layout()
        fig.savefig(d / f"fig_x2_bestofn{suffix}.pdf"); plt.close(fig)
    if x3:
        models = list(x3)
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
        w = 0.25
        for mi, m in enumerate(models):
            for pi_, pol in enumerate(("P0", "P1", "P2")):
                q = x3[m]["policies"].get(pol)
                if not q:
                    continue
                x = mi + (pi_ - 1) * w
                col = ["#c0504d", "#3b6ea8", "#4f9a5b"][pi_]
                axs[0].bar(x, q["repaired"], w, color=col, label=pol if mi == 0 else None)
                axs[1].bar(x, q["frac_code_revisit"], w, color=col)
        for a_, t in zip(axs, ["episodes repaired", "episodes with a revisited code state"]):
            a_.set_xticks(range(len(models))); a_.set_xticklabels(models, fontsize=7)
            a_.set_ylim(0, 1); a_.set_title(t, fontsize=9)
        axs[0].legend(fontsize=7); fig.tight_layout()
        fig.savefig(d / f"fig_x3_repair{suffix}.pdf"); plt.close(fig)


def tables(x1, x2, x3):
    T = []
    if x1:
        T.append("% ---- X1: informedness of LLM judges and programmatic verifiers ----")
        T.append("\\begin{tabular}{@{}l" + "c" * len(MODES) + "@{}}\n\\toprule")
        T.append(" & " + " & ".join(MODES) + "\\\\\n\\midrule")
        T.append("positive / negative traces & " + " & ".join(
            f"${p}/{n}$" for p, n in zip(x1["counts"]["pos"], x1["counts"]["neg"])) + "\\\\\n\\midrule")
        for nm, row in zip(x1["verifiers"], x1["J"]):
            T.append(nm.replace("_", "\\_") + " & " + " & ".join(f"${v:.3f}$" if v > 0 else "$0$" for v in row) + "\\\\")
        T.append("\\bottomrule\n\\end{tabular}\n")
        T.append("% ---- X1: game values and certificates ----")
        T.append("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nLibrary & $\\hat\\lambda$ & $\\lambda_{\\mathrm{LCB}}$ & "
                 "$\\lambda_{\\mathrm{UCB}}$ & uniform\\\\\n\\midrule")
        for ln, q in x1["libraries"].items():
            T.append(f"{ln} & ${q['lambda_hat']:.3f}$ & ${q['lambda_LCB']:.3f}$ & ${q['lambda_UCB']:.3f}$ & ${q['uniform_worst']:.3f}$\\\\")
        T.append("\\bottomrule\n\\end{tabular}\n")
    if x2:
        T.append("% ---- X2: best-of-N success ----")
        T.append("\\begin{tabular}{@{}l" + "c" * len(x2["N"]) + "@{}}\n\\toprule")
        T.append("Selector & " + " & ".join(f"$N={n}$" for n in x2["N"]) + "\\\\\n\\midrule")
        for s, v in x2["success"].items():
            T.append(s + " & " + " & ".join(f"${x:.3f}$" for x in v) + "\\\\")
        T.append("\\bottomrule\n\\end{tabular}\n")
    if x3:
        T.append("% ---- X3: repair policies ----")
        T.append("\\begin{tabular}{@{}llccccc@{}}\n\\toprule\nModel & policy & repaired & overall pass & "
                 "calls & code revisit & oscillation\\\\\n\\midrule")
        for m, d in x3.items():
            for pol, q in d["policies"].items():
                T.append(f"{m} & {pol} & ${q['repaired']:.3f}$ & ${q['overall_pass']:.3f}$ & ${q['mean_calls']:.2f}$ & "
                         f"${q['frac_code_revisit']:.3f}$ & ${q['frac_oscillation']:.3f}$\\\\")
        T.append("\\bottomrule\n\\end{tabular}\n")
    return "\n".join(T)


def cost_report(phase, suffix=""):
    rows = load_jsonl(RESULTS / phase / f"ledger{suffix}.jsonl")
    by_exp = collections.defaultdict(float)
    by_model = collections.defaultdict(float)
    for r in rows:
        by_exp[r["exp"]] += r["cost"]
        by_model[r["model"]] += r["cost"]
    lines = [f"Phase {phase}: cap ${PHASES[phase]['cap']:.2f}; spent (new calls only) ${sum(by_exp.values()):.4f}"]
    lines += [f"  {e}: ${c:.4f} (sub-budget ${PHASES[phase].get(e, {}).get('budget', 0.05):.2f})"
              for e, c in sorted(by_exp.items())]
    lines += ["  by model:"] + [f"    {m}: ${c:.4f}" for m, c in sorted(by_model.items(), key=lambda x: -x[1])]
    return "\n".join(lines), dict(by_exp), dict(by_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["test", "final"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    suffix = "_dry" if a.dry_run else ""
    x1, x2, x3 = analyse_x1(a.phase, suffix), analyse_x2(a.phase, suffix), analyse_x3(a.phase, suffix)
    rep, by_exp, by_model = cost_report(a.phase, suffix)
    d = RESULTS / a.phase
    summary = dict(phase=a.phase, dry_run=a.dry_run, x1=x1, x2=x2, x3=x3,
                   cost=dict(by_experiment=by_exp, by_model=by_model))
    (d / f"summary{suffix}.json").write_text(json.dumps(summary, indent=1))
    (d / f"tables{suffix}.tex").write_text(tables(x1, x2, x3))
    (d / f"cost_report{suffix}.txt").write_text(rep)
    figures(a.phase, x2, x3, suffix)
    print(rep)
    if x1:
        print("\nX1 game values:", {k: (v["lambda_hat"], v["lambda_LCB"], v["lambda_UCB"]) for k, v in x1["libraries"].items()})
    if x2:
        print("X2 success at N =", x2["N"], {k: v for k, v in x2["success"].items()})
        print("   base accuracy", round(x2["base_accuracy"], 3), "blind AUROC", round(x2["blind_AUROC"], 3))
    if x3:
        for m, q in x3.items():
            print("X3", m, "initial pass", round(q["initial_pass_rate"], 3),
                  {p: (round(v["repaired"], 3), round(v["frac_code_revisit"], 3)) for p, v in q["policies"].items()})
    print(f"\nWrote {d / ('summary' + suffix + '.json')}, tables, figures and cost report.")


if __name__ == "__main__":
    main()
