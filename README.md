# Language-model experiments (E6 to E8)

Designed, implemented and run by Narayana Darapaneni.

These scripts run the three language-model experiments of the paper: X1 is experiment E6, X2 is E7 and X3 is E8 (Section 5.3). They use the OpenAI API
and include a hard spending guard: about **$0.8 expected for the test phase (cap $2)** and
about **$4.7 expected for the final phase (cap $10)**.

| Experiment | What it tests in the paper | Data | Models (default) |
|---|---|---|---|
| **X1** LLM judges as verifiers | Verification modulus, blind spots, Proposition 4.1, Algorithm 3 (certified allocation), average detection vs worst-case soundness (Example 4.3) | 61 human-annotated MAST traces | test: gpt-5-nano, gpt-4o-mini, gpt-5.6-luna, gpt-4.1-mini, gpt-5-mini; final adds gpt-5.4-mini, gpt-5.6-terra, o3 |
| **X2** Best-of-N selection | Proposition 3.1 (sound selector) and Theorem 3.2 (blind-selector ceiling) | GSM8K test split | generator gpt-4o-mini; independent solver gpt-5.6-luna; judge and blind judge gpt-4o-mini |
| **X3** Verify-repair loops | Proposition 3.5 (repair can cycle) and Theorem 3.8 (a potential function gives termination) | HumanEval (tests split into single checks) | test: gpt-4o-mini; final adds gpt-5-nano |

## What each experiment measures

**X1. LLM judges as verifiers.** Each model reads a trace and says which of six failure
modes (FM-1.3, 2.1, 2.2, 2.3, 3.2, 3.3) are present.
- `analyze.py` compares these answers with the human labels. For each judge and mode it
  computes TPR, FPR and informedness J = (TPR − FPR)⁺.
- It then computes the game value λ̂ and the certified interval [λ_LCB, λ_UCB] of
  Algorithm 3, for three libraries: the LLM judges, the six programmatic verifiers W1–W6
  of the paper (experiment E5) and both together.
- It also reports held-out results over 500 random half splits, agreement with the human
  labels (Cohen's κ), self-consistency of repeated judgements and the cost per trace.
- The paper predicts that single verifiers have blind modes (a zero in their row) and that
  mixtures do better. It also predicts that the verifier with the best average detection
  can have zero worst-case soundness.

**X2. Best-of-N selection.** A cheap model writes 16 short candidate solutions per problem.
Six selectors pick one candidate:
- **oracle:** knows the true answer.
- **independent solver:** a sound verifier.
- **LLM judge:** sees the question and the candidate.
- **majority vote.**
- **blind judge:** sees only the candidate text, not the question.
- **random.**

Theorem 3.2 predicts that the blind selector stays near the single-sample accuracy for
every N, while sound selectors approach the oracle. `blind_AUROC` measures how blind the
blind judge really is (0.5 means fully blind).

**X3. Verify-repair loops.** Failing HumanEval solutions are repaired under three policies.
- **P0:** shows only the first failing check and accepts every edit. This is the usual
  multi-agent loop.
- **P1:** shows all failing checks and accepts every edit.
- **P2:** accepts an edit only if fewer checks fail and stops after two rejected
  proposals. This is the discrete analogue of Bregman repair.

The paper predicts revisited code states and oscillating checks under P0 and P1.
It predicts guaranteed termination with no oscillation under P2.

## Step-by-step procedure

### Step 0. Requirements
- Python **3.10 or newer**. Check with `python --version`.
- git. Check with `git --version`.
- About 1 GB of free disk space. The MAST repository is about 200 MB.
- An OpenAI API account with billing enabled.

### Step 1. Open a terminal in this folder
```
cd llm_experiments
```

### Step 2. Create a virtual environment and install the packages
Windows (PowerShell):
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
macOS / Linux:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3. Set the API key (and a second safety limit)
1. Create a key at https://platform.openai.com/api-keys. A separate project for these
   experiments limits exposure.
2. In that project's limits, set a monthly budget of about $12. This is a second safety
   net on top of the scripts' own guard.
3. Put the key in an environment variable. Never write it into a file.

Windows (PowerShell, current window only):
```
$env:OPENAI_API_KEY = "sk-..."
```
macOS / Linux:
```
export OPENAI_API_KEY="sk-..."
```

### Step 4. Offline rehearsal (free)
```
python run_all.py --phase test --dry-run
```
This downloads the data (MAST, GSM8K, HumanEval) and runs the whole pipeline with a fake
model. No API calls are made. It should end with "Done." The numbers it prints are
meaningless; only check that no step fails.

### Step 5. Check model access (costs less than $0.01)
```
python preflight.py --phase test
```
- Every model should print `OK`.
- If a model prints `FAILED` (for example, your account has no access to it), remove it
  from the lists in `PHASES` in `common.py`.
- Also check the price table `MODELS` in `common.py` against
  https://platform.openai.com/docs/pricing. The guard uses these prices.
- The GPT-5.6, GPT-5.5 and GPT-5.4 prices are the published post-July-2026 rates. The older
  models use their last known list prices.

### Step 6. Test phase (expected about $0.8, hard cap $2)
```
python run_all.py --phase test
```
- This takes about 20 to 40 minutes with the default 8 parallel requests.
- If you see rate-limit messages, use `--workers 4`.
- The script prints the spend after every block.
- Check `results/test/cost_report.txt`. Also compare it with the usage page of your
  OpenAI dashboard.
- Inspect `results/test` before running the final phase: check that the judgements parse
  and that the numbers are plausible. Fix prompts or model lists before the larger run.

### Step 6b. Full MAST dataset, free (no API cost)
`x0_mad_full.py` runs the programmatic verifiers of the paper (W1–W5; W6 needs
reference answers, which the full dataset does not have) on all 1,642 traces of the full MAST
dataset. Their labels come from an LLM judge. This step makes no API calls.
```
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download as d; print(d('mcemri/MAST-Data','MAD_full_dataset.json',repo_type='dataset',local_dir='data'))"
python x0_mad_full.py data/MAD_full_dataset.json
```
The download is about 200 MB. The results are written to `results/mad_full/`.

### Step 7. Final phase (expected about $4.7, hard cap $10)
```
python run_all.py --phase final
```
- Every call already made in the test phase is reused from the `cache` folder and costs
  nothing.
- This takes about 1 to 2 hours. If the run is interrupted, run the same command again.
  Finished calls are cached and are not paid twice.

### Step 8. Outputs
The folder **`results/final`** then contains:
- `summary.json`
- `tables.tex`
- `fig_x2_bestofn.pdf`
- `fig_x3_repair.pdf`
- `cost_report.txt`
- the three `.jsonl` files

The paper reports this run. `python verify_llm_numbers.py results/final` prints the numbers
quoted in Section 5.3 that `analyze.py` does not print. `python make_fig_llm.py results/final`
draws Figure 4.

## How the budget is enforced
- Before every call, the guard reserves the worst-case cost of that call: estimated input
  tokens times the input price, plus the maximum output tokens times the output price. It
  refuses the call if the experiment's sub-budget or the phase cap would be exceeded.
- After the call, it books the actual usage that the API reports (reasoning tokens are
  counted as output).
- Sub-budgets:
  - test phase: X1 $1.00, X2 $0.45, X3 $0.35, preflight $0.05
  - final phase: X1 $5.00, X2 $2.50, X3 $1.50
- Judges are run cheapest first, so if money runs out, the expensive judges are the ones
  dropped.
- To change the budgets, edit `cap` and the `budget` entries in `PHASES` in `common.py`.
- The guard uses list prices. Your real bill can be slightly lower (for example, through
  discounts on cached input).

## Safety note for X3
X3 executes Python code written by a language model. Each program runs in its own process
with a timeout, in a temporary folder. This is **not a sandbox**.
- Do not run the scripts as administrator.
- If possible, run X3 inside a virtual machine or a Docker container.
- To skip X3: `python run_all.py --phase final --skip x3`

## Troubleshooting
- **`OPENAI_API_KEY is not set`**: repeat Step 3 in the same terminal window.
- **`model ... NotFound` / `PermissionDenied`**: your account has no access to that model.
  Remove it from `PHASES`.
- **`reasoning effort` errors**: change `effort` in `PHASES` (for example to `"medium"`).
- **Many "unparsable" judgements in X1**: inspect `results/test/x1_judgements.jsonl`; the
  prompt may need adjusting for that model.
- **Running only one experiment**: `python x2_bestofn.py --phase test`, then
  `python analyze.py --phase test`.

## Files
- `common.py`: model registry and prices, phase configuration, budget guard, cached API
  wrapper, data loaders, offline mock
- `preflight.py`: cost estimate and model access check
- `x0_mad_full.py`: free analysis of the full MAST dataset (no API calls)
- `x1_judges.py`, `x2_bestofn.py`, `x3_repair.py`: the three experiments
- `analyze.py`: all statistics, LaTeX tables, figures and the cost report
- `run_all.py`: runs one phase end to end
