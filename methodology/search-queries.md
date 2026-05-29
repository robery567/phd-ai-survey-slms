# Search Queries

Exact query strings, databases, and date stamps for the systematic literature review
described in Section 3.2 (Search and Screening). The search window was January–April
2026.

Queries combine three term groups with boolean `AND` across groups and `OR` within a
group:

- **Group 1 — model scope:** `"small language model"`, `"SLM"`, `"efficient LLM"`,
  `"on-device model"`, `"edge LLM"`, `"compact language model"`
- **Group 2 — safety scope:** `"safety"`, `"alignment"`, `"harmlessness"`,
  `"jailbreak"`, `"toxicity"`, `"bias"`, `"refusal"`, `"red-teaming"`
- **Group 3 — technique scope:** `"RLHF"`, `"DPO"`, `"quantization safety"`,
  `"pruning alignment"`, `"interpretability safety"`, `"data curation"`,
  `"model editing"`, `"unlearning"`

The canonical query template is:

```
(G1) AND (G2) AND (G3?)
```

Group 3 is optional (`G3?`): the broad pass uses only G1 AND G2; technique-targeted
passes add a single G3 term to surface method-specific work that the broad pass
ranked too low.

## Per-database queries and counts

| # | Database | Fields / filter | Query (canonical form) | Date | Raw hits |
|---|----------|-----------------|------------------------|------|----------|
| 1 | arXiv | cs.CL, cs.AI, cs.LG, cs.CR; 2022–2026 | `(G1) AND (G2)` | 2026-01-14 | 188 |
| 2 | arXiv | same | `(G1) AND (G2) AND ("quantization" OR "pruning" OR "distillation")` | 2026-01-14 | 41 |
| 3 | arXiv | same | `(G1) AND ("model editing" OR "unlearning")` | 2026-01-15 | 22 |
| 4 | Google Scholar | 2022–2026 | `(G1) AND (G2)` | 2026-02-03 | 73 |
| 5 | Semantic Scholar | Computer Science | `(G1) AND (G2)` | 2026-02-05 | 58 |
| 6 | ACM Digital Library | full text | `(G1) AND (G2)` | 2026-02-19 | 24 |
| 7 | IEEE Xplore | metadata | `(G1) AND (G2)` | 2026-02-20 | 17 |
| 8 | ACL Anthology | all years ≥2022 | `(G1) AND (G2)` | 2026-03-04 | 28 |

Raw hits sum to 451 candidate records before deduplication, matching the count
reported in §3.2 and Figure 5 (PRISMA flow). After the three-stage deduplication
(see `analysis/deduplicate.py`), 392 unique records remained.

## Notes

- arXiv was queried via the public API (`http://export.arxiv.org/api/query`) with the
  category filter applied server-side and the boolean groups applied to the
  `all:` field.
- Google Scholar and Semantic Scholar were queried through their respective web/API
  front-ends; counts are the de-paginated totals captured on the date shown.
- Foundational pre-2022 works (RLHF, PPO, InstructGPT, etc.) were added by hand
  outside this search and are flagged as `foundational` in the screening log; they
  are not counted in the 451.
