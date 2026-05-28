# Efficiency–Safety Interaction Rubric

This rubric defines the labels used in Table 4 (efficiency–safety summary) of the manuscript.

## Safety Impact

| Label | Criterion |
|-------|-----------|
| **Preserved** | Reported safety degradation ≤2 percentage points on at least one safety benchmark (HarmBench, DecodingTrust, or JailbreakBench) at the stated compression level, in ≥1 corpus paper. |
| **Slightly reduced** | 2–5 pp degradation on at least one benchmark. |
| **Degraded** | >5 pp degradation, or qualitative report of "significant" safety loss in the source paper. |
| **Risk of removal** | Method does not directly degrade safety but enables removal under adversarial fine-tuning (Qi et al. 2024). |
| **Effective** | Method actively improves a measured safety dimension. |
| **Can improve** | Method has been reported to improve safety in some configurations and degrade it in others; net effect is configuration-dependent. |
| **Risky** | Method has reported failure modes that are unrelated to absolute degradation magnitude (e.g., distillation amplifying multilingual gaps). |

## Recommendation

The "Recommendation" column states the actionable guidance derived from the safety-impact label. Each recommendation is annotated with the same superscript marker convention (†/‡/§) defined in `alignment-rubric.md`.
