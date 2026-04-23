"""
Bibliography analysis: citation statistics, temporal distribution, venue breakdown.
Analyzes references.bib to produce survey coverage statistics.
"""
import re
from collections import Counter


def parse_bib(path="manuscript/references.bib"):
    with open(path) as f:
        text = f.read()

    entries = []
    for match in re.finditer(r'@(\w+)\{(\w+),\s*(.*?)\n\}', text, re.DOTALL):
        entry_type, key, body = match.groups()
        year_match = re.search(r'year\s*=\s*\{?(\d{4})\}?', body)
        title_match = re.search(r'title\s*=\s*\{(.+?)\}', body)
        venue_match = re.search(r'booktitle\s*=\s*\{(.+?)\}', body)
        journal_match = re.search(r'journal\s*=\s*\{(.+?)\}', body)

        entries.append({
            "key": key,
            "type": entry_type,
            "year": int(year_match.group(1)) if year_match else None,
            "title": title_match.group(1) if title_match else "",
            "venue": (venue_match or journal_match).group(1) if (venue_match or journal_match) else "unknown"
        })
    return entries


def analyze(entries):
    print(f"Total references: {len(entries)}")

    # Year distribution
    years = Counter(e["year"] for e in entries if e["year"])
    print(f"\nBy year:")
    for y in sorted(years):
        bar = "#" * years[y]
        print(f"  {y}: {years[y]:3d} {bar}")

    recent = sum(v for k, v in years.items() if k and k >= 2025)
    total = sum(years.values())
    print(f"\n2025-2026 recency: {recent}/{total} ({100*recent/total:.0f}%)")

    # Venue distribution
    venues = Counter()
    for e in entries:
        v = e["venue"].lower()
        if "neurips" in v or "neural information" in v:
            venues["NeurIPS"] += 1
        elif "icml" in v or "machine learning" in v and "international" in v:
            venues["ICML"] += 1
        elif "iclr" in v or "learning representations" in v:
            venues["ICLR"] += 1
        elif "acl" in v or "computational linguistics" in v and "association" in v:
            venues["ACL/EMNLP"] += 1
        elif "emnlp" in v or "empirical methods" in v:
            venues["ACL/EMNLP"] += 1
        elif "aaai" in v:
            venues["AAAI"] += 1
        elif "arxiv" in v:
            venues["arXiv preprint"] += 1
        else:
            venues["Other"] += 1

    print(f"\nBy venue category:")
    for venue, count in venues.most_common():
        print(f"  {venue}: {count}")

    # Type distribution
    types = Counter(e["type"] for e in entries)
    print(f"\nBy entry type:")
    for t, count in types.most_common():
        print(f"  {t}: {count}")


if __name__ == "__main__":
    entries = parse_bib()
    analyze(entries)
