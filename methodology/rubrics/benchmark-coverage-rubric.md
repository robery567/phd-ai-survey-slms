# Benchmark Coverage Coding Rule (Figure 5)

This rubric defines the green/yellow/red coding used in the benchmark coverage matrix (Figure 5).

For each (benchmark × dimension) cell:

| Color | Operational criterion |
|-------|----------------------|
| **Green** (full coverage) | The benchmark explicitly evaluates the dimension as a primary axis: it has a dedicated subset, score, or table for that dimension in the original benchmark paper. |
| **Yellow** (partial coverage) | The dimension is touched on but treated as a secondary or partial measurement: e.g., toxicity scores reported as a sub-metric in a multi-dimensional benchmark, or a small sub-sample without statistical analysis. |
| **Red** (absent) | The dimension is not evaluated by the benchmark. |

The coding is performed against the benchmark's introducing paper. Subsequent extensions (e.g., translated re-releases) do not change the original benchmark's coding; they appear as separate entries in Table 2 if they meet the inclusion criteria.

## Dimensions evaluated

The 9 dimensions in Figure 5 are: Toxicity, Bias, Truthfulness, Adversarial robustness, Multilingual, Privacy, Over-refusal, Compression-aware, Size-stratified.

Per-cell evidence is released as `data/figure5-derivation.csv` listing the benchmark paper section that justifies each coding decision.
