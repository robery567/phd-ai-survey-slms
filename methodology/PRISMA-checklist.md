# PRISMA-2020 Checklist (adapted for a computing survey)

This checklist maps the PRISMA-2020 reporting items to where each is addressed in the
manuscript and the companion repository. The review is *inspired by* PRISMA-2020
(Page et al., 2021) and adapted for the fast-moving ML-research setting; items that do
not apply to a methods/benchmark survey (e.g., clinical effect-measure synthesis) are
marked **n/a** with a reason.

| # | PRISMA-2020 item | Addressed in | Notes |
|---|------------------|--------------|-------|
| 1 | Title identifies the report as a systematic review | Title | "A Systematic Survey of Methods, Evaluation, and Open Challenges" |
| 2 | Abstract: structured summary | Abstract | Objectives, scope (139 papers, 2022–2026), seven axes, empirical validation |
| 3 | Rationale | §1 Introduction | Why SLM safety is a distinct problem (capacity, compression, alignment tax, multilingual, deployment) |
| 4 | Objectives / research questions | §1, RQ1–RQ6 | Six explicit research questions |
| 5 | Eligibility criteria | §3.2 (Inclusion/Exclusion criteria) | Operational inclusion rule incl. 2026-preprint sub-criteria (a/b/c) |
| 6 | Information sources | §3.2 | arXiv, Google Scholar, Semantic Scholar, ACM DL, IEEE Xplore, ACL Anthology; Jan–Apr 2026 |
| 7 | Search strategy (full strings) | `methodology/search-queries.md` | Exact per-database queries, dates, raw counts |
| 8 | Selection process | §3.2, Figure 5 (PRISMA flow) | Title/abstract then full-text screening; single coder with two-pass mitigation (§3.3) |
| 9 | Data collection process | §3.3 (Coding Procedure) | Six extracted fields per paper; multi-axis assignment rule |
| 10 | Data items | §3.3 | Axes, method family, model scale, contribution, limitation, evidence quality |
| 11 | Risk of bias (per study) | §3.3, §15 | Evidence-quality field per paper; `\preliminary{}` flag for unreplicated 2026 preprints; single-coder limitation disclosed |
| 12 | Effect measures | **n/a** | Methods survey, not a meta-analysis of a common effect measure |
| 13 | Synthesis methods | §3.1, §3.3, §3.4 (rubrics + taxonomy derivation) | Qualitative comparative synthesis via documented rubrics; provenance markers |
| 14 | Reporting bias assessment | §15 (Limitations) | English-only, 7B threshold, preprint permissiveness, single-coder all disclosed |
| 15 | Certainty assessment | §3.4, evidence-tier tags | `\tagconsensus{}` / `\tagsingle{}` / `\tagour{}` distinguish maturity of claims |
| 16 | Study selection results (flow) | Figure 5 (PRISMA flow) | 451 → 392 (dedup) → 280 (title/abstract) → 139 (full-text) |
| 17 | Study characteristics | §4–§10, Tables 3–4 | Per-axis comparative tables |
| 18 | Risk of bias in studies | §3.3 + `methodology/screening-log.csv` | Evidence-quality coding per paper |
| 19 | Results of individual studies | §4–§13 narrative + comparative tables | |
| 20 | Results of syntheses | §4–§13, Figures 4–5 | Per-axis comparative analyses |
| 21 | Reporting biases | §15 | |
| 22 | Certainty of evidence | evidence-tier tags + `\preliminary{}` | |
| 23 | Discussion / limitations / conclusions | §15, §16 | |
| 24 | Registration and protocol | §3.1 + this file | Not pre-registered (typical for CS surveys); protocol documented post-hoc here and in §3 |
| 25 | Support / funding | Funding statement | |
| 26 | Competing interests | Competing Interests statement | |
| 27 | Availability of data, code, materials | §3, Data Availability | `methodology/` (queries, screening log, rubrics, reproducibility), `data/` (derivation CSVs), `experiments/` (notebooks + results JSONs) |

## Deduplication detail (PRISMA item 16)

The 451 → 392 step is the three-stage deduplication implemented in
`analysis/deduplicate.py`: DOI exact-match → arXiv-ID exact-match → normalized-title
Jaccard ≥ 0.92. The per-stage counts and the collapsed duplicate clusters are emitted
by that script for auditing.

## Multi-axis assignment (PRISMA items 8–9)

Some papers contribute to more than one taxonomic axis. A paper is assigned a second
axis only when it makes a direct, evaluated contribution there (not a passing
reference); the rule and worked example (Safe LoRA → training-time + efficiency) are
in §3.3. Per-paper axis assignments and the two-pass agreement flag are in
`methodology/screening-log.csv`.
