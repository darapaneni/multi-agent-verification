# Author: Narayana Darapaneni
# Code for "Why Verification Fails in Multi-Agent LLM Systems" (2026). MIT License.
"""Figure 4 of the paper: (a) best-of-N success (X2), (b) repair policies pooled over repairers (X3).
Usage: python make_fig_llm.py results/final  (writes fig_llm.pdf)"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1] if len(sys.argv) > 1 else "results/final"
s = json.load(open(f"{d}/summary.json"))
rows = [json.loads(l) for l in open(f"{d}/x3_repair.jsonl")]
eps = [r for r in rows if r.get("policy") in ("P0", "P1", "P2")]

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
x2 = s["x2"]; N = x2["N"]
sty = {"oracle": ("#333333", "o", "oracle"), "solver": ("#4f9a5b", "s", "independent solver"),
       "judge": ("#3b6ea8", "^", "LLM judge"), "majority": ("#8a5bb0", "D", "majority vote"),
       "blind": ("#c0504d", "v", "blind judge"), "random": ("#888888", "x", "random")}
for k, (c, m, lab) in sty.items():
    ax[0].plot(N, x2["success"][k], marker=m, color=c, label=lab, lw=1.2, ms=4)
ax[0].set_xscale("log", base=2); ax[0].set_xticks(N); ax[0].set_xticklabels([str(n) for n in N])
ax[0].set_xlabel("candidates $N$"); ax[0].set_ylabel("success rate")
ax[0].set_title(f"(a) best-of-$N$ on GSM8K ({x2['n_problems']} problems)", fontsize=8)
ax[0].legend(fontsize=5.5, ncol=2, frameon=False, loc="upper left")

metrics = [("repaired", lambda r: r["final_pass"]), ("code state revisited", lambda r: r["revisits_code"] > 0),
           ("oscillating check", lambda r: (len(r["oscillating"]) if isinstance(r["oscillating"], list) else r["oscillating"]) > 0)]
cols = {"P0": "#c0504d", "P1": "#3b6ea8", "P2": "#4f9a5b"}
w = 0.26
for i, p in enumerate(["P0", "P1", "P2"]):
    e = [r for r in eps if r["policy"] == p]
    vals = [np.mean([f(r) for r in e]) for _, f in metrics]
    ax[1].bar(np.arange(3) + (i - 1) * w, vals, w, color=cols[p], label=p)
n = len([r for r in eps if r["policy"] == "P0"])
ax[1].set_xticks(range(3)); ax[1].set_xticklabels([m for m, _ in metrics], fontsize=7)
ax[1].set_ylim(0, 1); ax[1].set_ylabel(f"fraction of {n} episodes")
ax[1].set_title("(b) repair policies on HumanEval", fontsize=8); ax[1].legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig("fig_llm.pdf")
print("pooled episodes per policy:", n)
