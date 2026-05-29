# Reproducibility Appendix

This file documents every empirical experiment in Section 14 (Empirical Validation)
of the manuscript in enough detail that a third party can reproduce each result
without contacting the author. It is referenced from §3.5 (Empirical Validation
Protocol) and §14.

All experiments are implemented as self-contained Jupyter notebooks under
`experiments/`, designed to run on Google Colab with a mounted Drive. Each notebook
opens with a `REPRO` dictionary that is serialized into its output JSON, so the exact
runtime configuration travels with the results.

---

## 1. Common configuration

| Item | Value |
|------|-------|
| Global seed | `42` (Python `random`, NumPy, PyTorch, CUDA all seeded) |
| Hardware (original §14 sweep, Exp 1–8) | NVIDIA A100 40 GB via Google Colab |
| Hardware (revision experiments E1/E3/E4) | NVIDIA A100 80 GB (high-memory Colab Pro runtime) |
| Primary judge | `meta-llama/Llama-Guard-3-1B` |
| Secondary judge (E1 only) | `cais/HarmBench-Llama-2-13b-cls` (official HarmBench classifier) |
| Tertiary signal | keyword-based refusal detection (rule list in `experiments/run_safety_benchmarks.py`) |
| Decoding | greedy (`do_sample=False`) for all models except the Qwen 3 family, which uses temperature-0.7 sampling reported over three independent runs |
| PyTorch | 2.11.0+cu128 |
| transformers (Exp 1–8, E1, E2, E4) | 5.9.0 (E1 run); current release at run time |
| transformers (E3 / GCG) | **pinned to 4.46.3** (nanoGCG 0.3.0 requires ≤4.47.1) |
| Statistics | 95% Wilson intervals (`statsmodels.stats.proportion.proportion_confint`, `method='wilson'`, `z_{0.975}=1.96`); Cohen's κ with 1,000-resample bootstrap CIs |

**Note on HuggingFace revisions.** Models are loaded by their canonical repository ID
at the head revision on the run date stamped in each result JSON's `repro.created`
field (E1: 2026-05-28; E3: 2026-05-29). To pin a specific revision, pass
`revision=<SHA>` to `from_pretrained`; the head revisions used are recorded in the
Colab execution logs committed alongside the notebooks. Gated models (Llama Guard
3-1B, Llama 3.2, the HarmBench classifier) require a HuggingFace token with the
relevant licence acceptance.

---

## 2. Model roster (49 configurations)

| Label | HuggingFace ID | Used in |
|-------|----------------|---------|
| Qwen 2.5-0.5B | `Qwen/Qwen2.5-0.5B-Instruct` | Exp 1 |
| Qwen 2.5-1.5B | `Qwen/Qwen2.5-1.5B-Instruct` | Exp 1 |
| Qwen 2.5-3B | `Qwen/Qwen2.5-3B-Instruct` | Exp 1, 2, 4–8, E1–E4 |
| Qwen 2.5-7B | `Qwen/Qwen2.5-7B-Instruct` | Exp 1 |
| Qwen 2.5-3B-Base | `Qwen/Qwen2.5-3B` | Exp 3 |
| Llama 3.2-1B | `meta-llama/Llama-3.2-1B-Instruct` | Exp 1 |
| Llama 3.2-3B | `meta-llama/Llama-3.2-3B-Instruct` | Exp 1, 4–8, E1, E3, E4 |
| Llama 3.2-3B-Base | `meta-llama/Llama-3.2-3B` | Exp 3 |
| Gemma 2-2B | `google/gemma-2-2b-it` | Exp 1 |
| Gemma 3-1B | `google/gemma-3-1b-it` | Exp 1 |
| Gemma 3-4B | `google/gemma-3-4b-it` | Exp 1 |
| Qwen 3-0.6B | `Qwen/Qwen3-0.6B` | Exp 1 |
| Qwen 3-1.7B | `Qwen/Qwen3-1.7B` | Exp 1 |
| Qwen 3-4B | `Qwen/Qwen3-4B` | Exp 1, 4–8, E4; variance run |
| Phi-4-Mini | `microsoft/Phi-4-mini-instruct` | Exp 1, 4–8, E4 |
| SmolLM2-1.7B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Exp 1 |

Quantization sweep (Exp 2) uses Qwen 2.5-3B at FP16, INT8, and INT4 via
`bitsandbytes` (`load_in_8bit=True` / `load_in_4bit=True`, NF4 with double
quantization for INT4). The remaining keys in `slm_safety_results_v3.json` are
language/attack variants of the four-model subset (suffixes `_ro/_zh/_es/_ar`,
`_jb`, `_mt`, `_or`).

---

## 3. Per-experiment detail

### Experiments 1–8 (original §14 sweep)

| Field | Value |
|-------|-------|
| Notebook | `experiments/safety_benchmarks.ipynb` (+ `run_safety_benchmarks.py`) |
| Result file | `experiments/results/slm_safety_results_v3.json` (49 configs) |
| Exp 1 (size-stratified) | full HarmBench standard test, 159 behaviors; 100 JailbreakBench benign for over-refusal; 14 instruct models |
| Exp 2 (compression) | Qwen 2.5-3B at FP16/INT8/INT4; same 159+100 prompts |
| Exp 3 (base vs instruct) | Qwen 2.5-3B and Llama 3.2-3B base vs instruct |
| Exp 4 (multilingual probe) | 15 self-translated harmful + 15 benign prompts per language (RO/ZH/ES/AR); 4-model subset |
| Exp 5 (jailbreak) | 5 templates × 20 HarmBench behaviors = 100 prompts; 4-model subset |
| Exp 6 (multi-turn crescendo) | 2 templates × 5 topics = 10 scenarios; 4-model subset |
| Exp 7 (over-refusal) | 100 JailbreakBench benign + 20 harm-adjacent; 4-model subset |
| Exp 8 (decoding) | Qwen 2.5-3B greedy vs temperature-0.7 sampling on 50 HarmBench behaviors |
| Per-prompt fields | `refused` (hybrid), `guard_label`, `kw_refused`, response text, response length |

### E1 — Inter-judge agreement (§14.9)

| Field | Value |
|-------|-------|
| Notebook | `experiments/09_official_harmbench_classifier.ipynb` |
| Result file | `experiments/results/harmbench_classifier_agreement.json` |
| Source generations | `slm_safety_results_v3.json`, Exp 1 configs (first 14 keys) |
| Sample | stratified by (model, normalised Llama Guard label), proportional allocation, ≥2 per stratum; n = 251 after the minimum was applied (target 250) |
| Classifier | `cais/HarmBench-Llama-2-13b-cls`, FP16, official HarmBench prompt template; `max_new_tokens=4`, greedy |
| Metric | Cohen's κ vs Llama Guard 3-1B (labels collapsed to safe/unsafe), 1,000-resample bootstrap 95% CI; per-model κ; confusion matrix |
| Result | overall κ = 0.009, 95% CI [−0.053, +0.065] |

### E2 — Wilson confidence intervals (Table 1, Appendix A)

| Field | Value |
|-------|-------|
| Notebook | `experiments/10_confidence_intervals.ipynb` (CPU; no GPU needed) |
| Result files | `experiments/results/section14_intervals.json`, `section14_intervals.csv` |
| Input | `slm_safety_results_v3.json` |
| Output | for every config: harmful/benign refusal under three judges (reported / guard / keyword), each as point estimate + 95% Wilson CI; 294 CIs total (49 configs × 3 judges × 2 splits) |
| Appendix builder | `experiments/_build_appendix.py` → `manuscript/appendix-all-cis.tex` |

### E3 — Optimization-based GCG attack (§14.10)

| Field | Value |
|-------|-------|
| Notebook | `experiments/12_gcg_attack.ipynb` |
| Result file | `experiments/results/gcg_attack.json` (+ `gcg_attack.csv`) |
| Library | `nanogcg==0.3.0`; `transformers==4.46.3` (pinned); SDPA attention |
| Models | `Qwen/Qwen2.5-3B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct` (fp16) |
| Behaviors | 50 from `walledai/HarmBench` (standard config), intersected with the §14.5 behavior subset and topped up with random standard behaviors; affirmative target `"Sure, here is {behavior}"` |
| GCG config | `num_steps=250`, `search_width=1024`, `batch_size=512`, `topk=256`, `n_replace=2`, `early_stop=True`, `use_prefix_cache=True` |
| Judge | Llama Guard 3-1B on the GCG-suffixed prompt; `max_new_tokens=128`, greedy |
| Result | ASR = 100% (50/50) on both, 95% Wilson CI [0.929, 1.000] |
| Excluded models | Phi-4-Mini (partial-RoPE, needs transformers ≥4.49) and Qwen 3-4B (needs ≥4.51) cannot load under the nanoGCG ≤4.47.1 pin |

### E4 — PolyGuardPrompts + M-ALERT multilingual (§14.4.1)

| Field | Value |
|-------|-------|
| Notebook | `experiments/11_polyguard_multilingual.ipynb` |
| Result file | `experiments/results/multilingual_polyguard_malert.json` (+ CSV) |
| Datasets | `ToxicityPrompts/PolyGuardPrompts` (flat `language` column, full names; filter `prompt_harm_label == 'harmful'`); `felfri/M-ALERT` (parallel-prompt schema, columns en/de/es/fr/it) |
| PolyGuard languages | english, arabic, chinese, spanish (Romanian NOT covered by the dataset) |
| M-ALERT languages | en, de, fr, it, es |
| Sample | 100 prompts per (dataset, language) |
| Models | Qwen 2.5-3B, Llama 3.2-3B, Qwen 3-4B, Phi-4-Mini; `max_new_tokens=256` |
| Judge | Llama Guard 3-1B |
| Result | ASR = 0% in all 36 cells, 95% Wilson upper bound 3.7% |

### Aggregation

| Field | Value |
|-------|-------|
| Notebook | `experiments/13_aggregate_revision_results.ipynb` (CPU) |
| Consumes | `section14_intervals.json`, `harmbench_classifier_agreement.json`, `multilingual_polyguard_malert.json`, `gcg_attack.json` |
| Produces | the five `manuscript_table_*.tex` fragments, `revision_summary.txt`, and the regenerated `figures/fig6-empirical-results.{pdf,png}` |

---

## 4. Table / figure → result-file map

| Manuscript artifact | Source file(s) | Notebook |
|---------------------|----------------|----------|
| Table 1 (`tab:empirical`, §14.1) | `section14_intervals.json` | 10 |
| Table 2 (`tab:multilingual`, §14.4) | `slm_safety_results_v3.json` (15-prompt probe) | safety_benchmarks |
| Table 8 (`tab:polyguard_malert`, §14.4.1) | `multilingual_polyguard_malert.json` | 11 |
| Table 9 (`tab:gcg`, §14.10) | `gcg_attack.json` | 12 |
| Classifier-agreement table (§14.9) | `harmbench_classifier_agreement.json` | 09 |
| Appendix A (294 CIs) | `section14_intervals.json` → `_build_appendix.py` | 10 |
| Figure 6 (`fig:empirical`) | all of the above | 13 |
| Tables 3, 4 (method/efficiency comparison) | `data/*-derivation.csv` + `methodology/rubrics/*` | n/a (literature-coded) |
| Figures 4, 5 (alignment radar, benchmark coverage) | `data/alignment-radar-derivation.csv`, `data/benchmark-coverage-derivation.csv` | n/a |

---

## 5. How to reproduce

1. Mount Drive in Colab; place `experiments/results/slm_safety_results_v3.json` under
   `/content/drive/MyDrive/PhD/paper1-survey/experiments/results/`.
2. Set a HuggingFace token (Colab Secrets `HF_TOKEN`) with Llama 3 / Llama Guard
   licence access.
3. Run notebooks in order: 10 (CIs, CPU) → 09 (E1, A100) → 11 (E4, A100) →
   12 (E3, A100) → 13 (aggregation, CPU). Notebook 10 is the cheapest and updates
   every Table 1 rate to a CI-annotated form.
4. Each notebook writes its JSON to the Drive results directory; notebook 13 reads
   them all and regenerates the tables and figure. Pull the JSONs back into the repo
   `experiments/results/` to refresh local artifacts.

All seeds are fixed at 42. The only expected source of run-to-run variation is the
Qwen 3 sampling family (reported with mean ± SD over three runs) and minor
nondeterminism in CUDA matmul kernels, which does not affect the reported
conclusions.
