"""Three-stage deduplication of candidate records for the SLM-safety survey.

Implements the procedure documented in Section 3.2 (Search and Screening):
    (i)  exact-match on DOI when present;
    (ii) exact-match on arXiv identifier;
    (iii) normalized-title fuzzy matching at >= 0.92 Jaccard similarity (token set).

Input:  a CSV of raw candidate records with columns
        [source, title, doi, arxiv_id, year, url]
Output: a CSV of unique records (first occurrence kept) plus a CSV of the
        duplicate clusters that were collapsed, for auditing.

Usage:
    python3 analysis/deduplicate.py raw_candidates.csv \
        --out unique_records.csv --clusters dup_clusters.csv
"""
import argparse
import csv
import re
from collections import defaultdict


_TOKEN = re.compile(r"[a-z0-9]+")


def normalize_title(title: str) -> set[str]:
    """Lowercase, strip punctuation, return the token *set* for Jaccard matching."""
    return set(_TOKEN.findall((title or "").lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def deduplicate(rows: list[dict], threshold: float = 0.92):
    """Return (unique_rows, clusters). First occurrence of each record is kept."""
    seen_doi: dict[str, int] = {}
    seen_arxiv: dict[str, int] = {}
    kept: list[dict] = []
    kept_titles: list[set[str]] = []
    clusters: dict[int, list[str]] = defaultdict(list)

    for row in rows:
        doi = (row.get("doi") or "").strip().lower()
        arxiv = (row.get("arxiv_id") or "").strip().lower()

        # Stage (i): DOI exact match.
        if doi and doi in seen_doi:
            clusters[seen_doi[doi]].append(row.get("title", ""))
            continue
        # Stage (ii): arXiv identifier exact match.
        if arxiv and arxiv in seen_arxiv:
            clusters[seen_arxiv[arxiv]].append(row.get("title", ""))
            continue
        # Stage (iii): fuzzy title match against everything kept so far.
        title_tokens = normalize_title(row.get("title", ""))
        dup_idx = None
        for idx, prev_tokens in enumerate(kept_titles):
            if jaccard(title_tokens, prev_tokens) >= threshold:
                dup_idx = idx
                break
        if dup_idx is not None:
            clusters[dup_idx].append(row.get("title", ""))
            continue

        # Unique: keep it and register its identifiers.
        new_idx = len(kept)
        if doi:
            seen_doi[doi] = new_idx
        if arxiv:
            seen_arxiv[arxiv] = new_idx
        kept.append(row)
        kept_titles.append(title_tokens)

    return kept, clusters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="raw candidate records CSV")
    ap.add_argument("--out", default="unique_records.csv")
    ap.add_argument("--clusters", default="dup_clusters.csv")
    ap.add_argument("--threshold", type=float, default=0.92)
    args = ap.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    kept, clusters = deduplicate(rows, threshold=args.threshold)

    fieldnames = rows[0].keys() if rows else ["source", "title", "doi", "arxiv_id", "year", "url"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(kept)

    with open(args.clusters, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kept_index", "kept_title", "collapsed_duplicate_title"])
        for idx, dup_titles in sorted(clusters.items()):
            for dt in dup_titles:
                w.writerow([idx, kept[idx].get("title", ""), dt])

    print(f"Input records:  {len(rows)}")
    print(f"Unique records: {len(kept)}")
    print(f"Duplicates collapsed: {len(rows) - len(kept)}")
    print(f"Wrote {args.out} and {args.clusters}")


if __name__ == "__main__":
    main()
