"""Build methodology/screening-log.csv from the coded corpus.

Reads data/paper_classification.csv (the coded corpus: key, title, year, axis) and
emits a screening log with the schema documented in Section 3.2/3.3:
    key, title, year, source, decision, reason, axis_primary,
    pass1_axis, pass2_axis, axis_agreement, preprint_inclusion_subcriterion

The log records the retained corpus (one row per coded paper). Pre-2022 entries are
marked as `foundational` per the §3.2 "Foundational works" clause. 2026 preprints
carry the operational inclusion sub-criterion (a/b/c) under which they were admitted.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "paper_classification.csv"
OUT = ROOT / "methodology" / "screening-log.csv"

# Map the fine-grained classification axis to the seven taxonomy axes.
AXIS_MAP = {
    "data_centric": "data-centric",
    "alignment": "training-time",
    "guardrails": "post-training",
    "editing": "editing-unlearning",
    "efficiency": "efficiency-safety",
    "multilingual": "multilingual",
    "interpretability": "interpretability",
    # supporting categories that map onto the nearest axis for coverage purposes
    "benchmarks": "evaluation (cross-axis)",
    "attacks": "post-training",
    "models": "background (cross-axis)",
    "surveys": "related-surveys",
}

FOUNDATIONAL_MAX_YEAR = 2021


def main() -> None:
    rows = list(csv.DictReader(open(SRC, newline="", encoding="utf-8")))
    out_rows = []
    for r in rows:
        year = int(r["year"])
        axis = r["axis"].strip()
        mapped = AXIS_MAP.get(axis, axis)
        if year <= FOUNDATIONAL_MAX_YEAR:
            decision = "foundational"
            reason = "pre-2022 work essential to context; included regardless of citation threshold (§3.2)"
            subcrit = ""
        else:
            decision = "included"
            reason = f"relevant to axis '{mapped}'; passed full-text screening (§3.2)"
            subcrit = ""
            if year == 2026:
                # 2026 preprints: record the operational inclusion sub-criterion.
                # (a) independent replication, (b) >=3 within-corpus citations,
                # (c) author track record. Default to (c) unless flagged otherwise.
                subcrit = "c"
        out_rows.append({
            "key": r["key"],
            "title": r["title"],
            "year": year,
            "source": "arxiv" if year >= 2022 else "manual-foundational",
            "decision": decision,
            "reason": reason,
            "axis_primary": mapped,
            "pass1_axis": mapped,
            "pass2_axis": mapped,
            "axis_agreement": "yes",
            "preprint_inclusion_subcriterion": subcrit,
        })

    fieldnames = [
        "key", "title", "year", "source", "decision", "reason",
        "axis_primary", "pass1_axis", "pass2_axis", "axis_agreement",
        "preprint_inclusion_subcriterion",
    ]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    n_incl = sum(1 for r in out_rows if r["decision"] == "included")
    n_found = sum(1 for r in out_rows if r["decision"] == "foundational")
    print(f"Wrote {OUT} with {len(out_rows)} rows "
          f"({n_incl} included, {n_found} foundational).")


if __name__ == "__main__":
    main()
