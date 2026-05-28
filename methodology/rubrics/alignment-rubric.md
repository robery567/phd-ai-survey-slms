# Alignment Methods Comparison Rubric

This rubric defines the ordinal labels used in Table 3 (alignment comparison) of the manuscript. Every cell in Table 3 is derived from this rubric and the coded paper(s) supporting it. The supporting paper IDs are released in `data/table3-derivation.csv`.

Labels: `Very High`, `High`, `Medium`, `Low`, `Very Low`, `Minimal`.

## Memory Cost (lower is better for SLMs; the table reports memory-cost level, not "memory efficiency")

| Level | Operational criterion (FP16 reference + training, 7B model) |
|-------|------------------------------------------------------------|
| **Very High** | >40 GB. All four models simultaneously: policy + reference + reward + value. Examples: RLHF/PPO. |
| **High** | 20–40 GB. Two models: policy + reference. Examples: DPO, IPO. |
| **Medium** | 10–20 GB. One model + lightweight overhead (e.g., teacher inference for AI feedback). Examples: KTO, RLAIF. |
| **Low** | 4–10 GB. Single-model training without reference. Examples: ORPO, SimPO. |
| **Minimal** | <4 GB or no training cost (inference-only intervention). Examples: RepE. |

LoRA-based variants reduce these levels by approximately one tier each.

## Robustness (higher is better)

Aggregated from reported HarmBench / JailbreakBench / StrongREJECT results in the corpus paper(s) introducing or evaluating each method:

| Level | Operational criterion |
|-------|----------------------|
| **High** | ≥75% harmful-prompt refusal on HarmBench standard test, evaluated in ≥2 corpus papers; resists at least one optimization-based attack class. |
| **Medium** | 50–74% harmful-prompt refusal on HarmBench, or strong on one benchmark and middling on another. |
| **Low** | <50% harmful-prompt refusal on HarmBench, or robustness untested below 7B. |

When robustness scores are not directly available for a method, we report the modal level among DPO-family papers in the corpus that adapt or extend the method.

## SLM Fit (higher is better; composite)

Computed as a weighted composite of:
- 40% Memory Cost (inverted: Minimal=5, Low=4, Medium=3, High=2, Very High=1)
- 30% Training-data efficiency (5 = no preference data needed, 1 = >100k pairs needed)
- 30% Reported sub-7B compatibility (5 = explicitly evaluated and effective on ≤3B, 1 = only frontier-scale evaluation)

The resulting numeric score is binned to the same five-level ordinal scale.

## Provenance markers

Each cell in Table 3 carries one of three superscript markers, defined in §3.4 of the manuscript:

- **†** Directly extracted from a coded paper field (e.g., the memory cost is reported in the source paper).
- **‡** Aggregated by a defined rule across multiple coded papers in the corpus.
- **§** Expert synthesis by the author when the corpus does not provide a direct or aggregable answer.

§-marked cells are the most editorial; readers can locate them in the table and discount accordingly.
