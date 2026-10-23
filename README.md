# Why Verification Fails in Multi-Agent LLM Systems

Code and recorded results for the paper

> **Why Verification Fails in Multi-Agent LLM Systems: A Geometric Theory of Blind Spots, Repair and Verifier Allocation**
> Narayana Darapaneni. 

The paper answers six questions raised by the MAST study of multi-agent failures
(Cemri et al., NeurIPS 2025 Datasets and Benchmarks): why checks miss errors, when combining
checks helps, when best-of-N stops helping, why fix-and-recheck loops cycle, when two scores
rule out task derailment and how to allocate a verification budget with a certificate.
This repository reproduces every number, table and figure of Section 5.

Narayana Darapaneni designed, implemented and ran all experiments in this repository.

## Contents

| Path | Experiment | Paper |
|---|---|---|
| `exp_e4_sd.py` | E1 to E4 (simulations) with standard deviations | Table 4, Figures 1 to 3; writes `results.json` and the figure PDFs |
| `exp_robust_repair.py` | E1 robustness check of Theorem 3.12 | Section 5.1; writes `results_robust_repair.json` |
| `exp_robust.py <MAST>/traces` | E1 multi-seed repeats; E5 bootstrap and threshold sensitivity | Section 5.2; writes `results_robust.json` |
| `exp_real.py <MAST>/traces` | E5: rule-based checks on the 61 human-annotated MAST traces, half splits, Algorithm 3 | Table 5; writes `results_real.json` |
| `llm_experiments/` | E6 to E8: LLM judges (X1), best-of-N on GSM8K (X2), repair loops on HumanEval (X3) | Table 6, Figure 4; see `llm_experiments/README.md` |
| `llm_experiments/results/final/` | Recorded model outputs of the final run reported in the paper | Section 5.3 |
| `llm_experiments/results/test/` | Recorded outputs of the low-cost pilot run that fixed the prompts | not reported |
| `companion_note/` | Companion note with the technical facts cited in Sections 3.2 and 3.5 | supplementary |

## Setup

Python 3.10 or newer.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

No data are redistributed here. Download them from their original sources:

- **MAST traces (E5, E6):** clone https://github.com/multi-agent-systems-failure-taxonomy/MAST.
  The 61 human-annotated traces are in its `traces` folder.
- **Full MAST-Data release (optional):** https://huggingface.co/datasets/mcemri/MAST-Data.
- **GSM8K and HumanEval (E7, E8):** downloaded automatically by `llm_experiments/common.py`.

## Reproducing the results

Simulations (no data, no API calls):
```
python exp_e4_sd.py
python exp_robust_repair.py
```

MAST trace analysis (no API calls):
```
python exp_real.py   path/to/MAST/traces
python exp_robust.py path/to/MAST/traces
```

Language-model experiments, from the recorded outputs (no API calls):
```
cd llm_experiments
python analyze.py --phase final
python verify_llm_numbers.py results/final
python make_fig_llm.py results/final
```

Re-running the language-model experiments from scratch needs an OpenAI API key and is
described step by step in `llm_experiments/README.md`, including the spending guard.
Keep the key in the environment variable `OPENAI_API_KEY`; never write it into a file.
Experiment X3 executes model-written Python code in a subprocess with a timeout. This is
not a sandbox: do not run it with administrator rights; prefer a virtual machine or
container.

All random seeds are fixed, so the scripts reproduce the reported numbers exactly.

## Citation

```
@article{Darapaneni2026Verification,
  title  = {Why Verification Fails in Multi-Agent {LLM} Systems: A Geometric Theory of
            Blind Spots, Repair and Verifier Allocation},
  author = {Darapaneni, Narayana},
  journal= {Submitted to Transactions on Machine Learning Research},
  year   = {2026}
}
```

## License

Code: MIT License (see `LICENSE`). The MAST data, GSM8K and HumanEval remain under their
own licenses.
