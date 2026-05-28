# Methodology Supplementary Artifacts

This directory contains the search queries, screening logs, and coding rubrics referenced from the manuscript's Section 3 (Survey Methodology). Together they let a third party reproduce the literature review without contacting the author.

## Files

| File | Purpose | Manuscript reference |
|------|---------|---------------------|
| `search-queries.md` | Exact query strings per database, with date stamps and result counts | §3.2 |
| `screening-log.csv` | One row per candidate paper: source, decision, reason, axis assignment, both-pass coding decisions | §3.2, §3.3 |
| `rubrics/alignment-rubric.md` | Operational definitions for ordinal labels in Table 3 (alignment comparison) | §3.4 |
| `rubrics/efficiency-rubric.md` | Operational definitions for Table 4 (efficiency–safety) | §3.4 |
| `rubrics/benchmark-coverage-rubric.md` | Coding rule for green/yellow/red cells in Figure 5 | §3.4 |
| `reproducibility.md` | Per-experiment model IDs + revisions, quantization recipes, judge configs, decoding settings, seeds, hardware, software versions | §3.5 |
| `PRISMA-checklist.pdf` | Mapping of PRISMA-2020 items to manuscript sections | §3.2 |

Per-cell derivation CSVs supporting Tables 3, 4 and Figures 4, 5 live under `../data/`:

| File | Supports |
|------|----------|
| `data/alignment-comparison-derivation.csv` | Table 3 |
| `data/efficiency-safety-derivation.csv` | Table 4 |
| `data/alignment-radar-derivation.csv` | Figure 4 (rendered numbering: alignment radar) |
| `data/benchmark-coverage-derivation.csv` | Figure 5 (rendered numbering: benchmark coverage matrix) |

## Status

These artifacts are being prepared in tandem with the CSUR resubmission. The manuscript references them by filename; the files themselves are filled in across the revision workstream — see `../REVISION_PLAN.md` tasks M1.4, M1.5, M1.6, M1.9, M2.1–M2.5.
