# Author: Narayana Darapaneni
# Code for "Why Verification Fails in Multi-Agent LLM Systems" (2026). MIT License.
"""Recomputes the E6-E8 numbers quoted in Section 5.3 that are not printed by analyze.py.
Usage: python verify_llm_numbers.py results/test"""
import json, sys
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "results/test"
s = json.load(open(f"{d}/summary.json"))

# E6: mean Cohen's kappa and flag rates of each judge
pos = np.array(s["x1"]["counts"]["pos"]); neg = np.array(s["x1"]["counts"]["neg"])
for i, j in enumerate(s["x1"]["judges"]):
    k = np.mean([s["x1"]["agreement"][j][m]["kappa"] for m in s["x1"]["modes"]])
    T = np.array(s["x1"]["TPR"][i]); F = np.array(s["x1"]["FPR"][i])
    print(f"E6 {j}: mean kappa {k:.2f}; flag rate per mode {np.round((T*pos+F*neg)/(pos+neg), 3).tolist()}")
print("E6 valid half splits:", s["x1"]["n_splits"])

# E7: composition and within-problem AUROC (mean over mixed problems, bootstrap over problems)
rows = [json.loads(l) for l in open(f"{d}/x2_bestofn.jsonl")]
def num(a):
    try: return float(str(a).replace(",", "").strip())
    except ValueError: return None
comp = {"all": 0, "none": 0, "some": 0}; mixed = []
for r in rows:
    c = np.array([num(x["ans"]) == num(r["answer"]) for x in r["cands"]])
    key = "all" if c.all() else "none" if not c.any() else "some"; comp[key] += 1
    if key == "some": mixed.append((r, c))
print("E7 problems with all / none / some correct:", comp)
def auc(sc, c):
    sc = np.asarray(sc, float); p, n = sc[c], sc[~c]
    return ((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n))
rng = np.random.default_rng(0)
for key in ("blind", "judge"):
    vals = np.array([auc(r[key], c) for r, c in mixed])
    bs = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(2000)]
    print(f"E7 {key}: within-problem AUROC {vals.mean():.2f}, 95% CI {np.round(np.percentile(bs, [2.5, 97.5]), 2).tolist()}")

# E8: per-episode failing sets and code revisits
for r in (json.loads(l) for l in open(f"{d}/x3_repair.jsonl")):
    if r.get("policy") in ("P0", "P1", "P2"):
        print("E8", r["policy"], r["task"], "repaired" if r["final_pass"] else "not repaired",
              "calls", r["calls"], "code revisits", r["revisits_code"], "oscillating", r["oscillating"],
              "failing sets", [h["fail"] for h in r["hist"]])
