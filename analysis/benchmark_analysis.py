"""
Safety benchmarks dataset and gap analysis.
Generates Table 3 (benchmark comparison) and gap analysis from the survey.
"""
import json
import csv
from collections import Counter

BENCHMARKS = [
    {"name": "RealToxicityPrompts", "year": 2020, "languages": ["en"], "scale": 100000,
     "dimension": "toxicity", "automated_eval": True, "slm_tested": False,
     "compression_aware": False, "key": "gehman2020realtoxicity"},
    {"name": "ToxiGen", "year": 2022, "languages": ["en"], "scale": 274000,
     "dimension": "implicit_toxicity", "automated_eval": True, "slm_tested": False,
     "compression_aware": False, "key": "hartvigsen2022toxigen"},
    {"name": "TruthfulQA", "year": 2022, "languages": ["en"], "scale": 817,
     "dimension": "truthfulness", "automated_eval": True, "slm_tested": True,
     "compression_aware": False, "key": "lin2022truthfulqa"},
    {"name": "BBQ", "year": 2022, "languages": ["en"], "scale": 58000,
     "dimension": "social_bias", "automated_eval": True, "slm_tested": False,
     "compression_aware": False, "key": "parrish2022bbq"},
    {"name": "DecodingTrust", "year": 2023, "languages": ["en"], "scale": None,
     "dimension": "comprehensive_8dim", "automated_eval": True, "slm_tested": False,
     "compression_aware": False, "key": "wang2023decodingtrust"},
    {"name": "HarmBench", "year": 2024, "languages": ["en"], "scale": 510,
     "dimension": "harmful_behaviors", "automated_eval": True, "slm_tested": True,
     "compression_aware": False, "key": "mazeika2024harmbench"},
    {"name": "SALAD-Bench", "year": 2024, "languages": ["en"], "scale": 30000,
     "dimension": "comprehensive_66cat", "automated_eval": True, "slm_tested": False,
     "compression_aware": False, "key": "li2024salad"},
    {"name": "XSafety", "year": 2024, "languages": ["en", "zh", "ar", "hi", "fr", "de", "es", "pt", "ru", "ja"],
     "scale": 14000, "dimension": "multilingual_safety", "automated_eval": True,
     "slm_tested": False, "compression_aware": False, "key": "wang2024xsafety"},
    {"name": "JailbreakBench", "year": 2024, "languages": ["en"], "scale": 100,
     "dimension": "jailbreak_robustness", "automated_eval": True, "slm_tested": True,
     "compression_aware": False, "key": "chao2024jailbreakbench"},
    {"name": "StrongREJECT", "year": 2024, "languages": ["en"], "scale": None,
     "dimension": "refusal_robustness", "automated_eval": True, "slm_tested": True,
     "compression_aware": False, "key": "souly2024strongreject"},
    {"name": "OR-Bench", "year": 2024, "languages": ["en"], "scale": 80000,
     "dimension": "over_refusal", "automated_eval": True, "slm_tested": False,
     "compression_aware": False, "key": "cui2024orbench"},
    {"name": "SORRY-Bench", "year": 2024, "languages": ["en"], "scale": 1000,
     "dimension": "refusal_45cat", "automated_eval": True, "slm_tested": False,
     "compression_aware": False, "key": "xie2024sorry"},
    {"name": "AgentHarm", "year": 2024, "languages": ["en"], "scale": None,
     "dimension": "agent_safety", "automated_eval": True, "slm_tested": False,
     "compression_aware": False, "key": "andriushchenko2024agentharm"},
    {"name": "AILuminate v1.0", "year": 2025, "languages": ["en", "fr"], "scale": 24000,
     "dimension": "comprehensive_12cat", "automated_eval": True, "slm_tested": False,
     "compression_aware": False, "key": "ailuminate2025"},
    {"name": "PolyGuardPrompts", "year": 2025, "languages": 17, "scale": 29000,
     "dimension": "multilingual_moderation", "automated_eval": True, "slm_tested": False,
     "compression_aware": False, "key": "polyguard2025"},
    {"name": "M-ALERT", "year": 2025, "languages": ["de", "fr", "it", "es", "pt"], "scale": 75000,
     "dimension": "cross_lingual_safety", "automated_eval": True, "slm_tested": False,
     "compression_aware": False, "key": "malert2025"},
    {"name": "SafeArena", "year": 2025, "languages": ["en"], "scale": 500,
     "dimension": "web_agent_safety", "automated_eval": True, "slm_tested": False,
     "compression_aware": False, "key": "safearena2025"},
    {"name": "HalluHard", "year": 2026, "languages": ["en"], "scale": 950,
     "dimension": "multi_turn_hallucination", "automated_eval": True, "slm_tested": False,
     "compression_aware": False, "key": "halluhard2026"},
    {"name": "ATBench", "year": 2026, "languages": ["en"], "scale": 500,
     "dimension": "agent_trajectory", "automated_eval": True, "slm_tested": False,
     "compression_aware": False, "key": "atbench2026"},
]


def gap_analysis():
    print("=" * 60)
    print("BENCHMARK GAP ANALYSIS FOR SLM SAFETY")
    print("=" * 60)

    # Language coverage
    all_langs = set()
    for b in BENCHMARKS:
        langs = b["languages"]
        if isinstance(langs, list):
            all_langs.update(langs)
    print(f"\nLanguages covered: {len(all_langs)}")
    print(f"  Covered: {sorted(all_langs)}")
    print(f"  Missing (examples): ro, pl, cs, hu, bg, uk, el, tr, ko, th, vi")

    # SLM-tested
    slm_tested = [b["name"] for b in BENCHMARKS if b["slm_tested"]]
    print(f"\nBenchmarks with SLM-specific results: {len(slm_tested)}/{len(BENCHMARKS)}")
    print(f"  {slm_tested}")

    # Compression-aware
    comp = [b["name"] for b in BENCHMARKS if b["compression_aware"]]
    print(f"\nCompression-aware benchmarks: {len(comp)}/{len(BENCHMARKS)}")

    # By dimension
    dims = Counter(b["dimension"] for b in BENCHMARKS)
    print(f"\nBenchmarks by dimension:")
    for dim, count in dims.most_common():
        print(f"  {dim}: {count}")

    # By year
    years = Counter(b["year"] for b in BENCHMARKS)
    print(f"\nBenchmarks by year:")
    for year in sorted(years):
        print(f"  {year}: {years[year]}")

    print(f"\nCritical gaps:")
    print(f"  1. No size-stratified analysis (1B vs 3B vs 7B)")
    print(f"  2. No compression-aware evaluation (FP16 vs INT4 vs INT3)")
    print(f"  3. Romanian and Eastern European languages uncovered")
    print(f"  4. Agent safety benchmarks assume frontier-scale models")
    print(f"  5. Over-refusal not characterized for SLMs")


def export(path="data/benchmarks.json"):
    with open(path, "w") as f:
        json.dump(BENCHMARKS, f, indent=2)
    print(f"\nExported {len(BENCHMARKS)} benchmarks to {path}")


if __name__ == "__main__":
    gap_analysis()
    export()
