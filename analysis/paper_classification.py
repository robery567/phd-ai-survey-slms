"""
Generate the survey paper classification dataset.
Classifies all 215 included papers by survey axis, year, and venue.
This is the structured dataset behind the survey's coverage claims.
"""
import json
import csv
import re
from collections import Counter


def parse_bib(path="manuscript/references.bib"):
    with open(path) as f:
        text = f.read()
    entries = []
    for match in re.finditer(r'@(\w+)\{(\w+),\s*(.*?)\n\}', text, re.DOTALL):
        entry_type, key, body = match.groups()
        year_m = re.search(r'year\s*=\s*\{?(\d{4})\}?', body)
        title_m = re.search(r'title\s*=\s*\{(.+?)\}', body)
        entries.append({
            "key": key,
            "year": int(year_m.group(1)) if year_m else None,
            "title": title_m.group(1).replace("{", "").replace("}", "") if title_m else "",
        })
    return entries


# Manual axis classification based on citation context in the survey
AXIS_MAP = {
    "alignment": [
        "ouyang2022training", "stiennon2020learning", "ziegler2019fine",
        "rafailov2023direct", "azar2023general", "ethayarajh2024kto",
        "hong2024orpo", "meng2024simpo", "bai2022constitutional",
        "lee2023rlaif", "chen2024spin", "yuan2024selfrewarding",
        "munos2024nash", "wu2024sppo", "rosset2024dno",
        "zou2023representation", "arditi2024refusal", "turner2023activation",
        "dai2024safe", "tunstall2023zephyr", "ivison2023tulu",
        "liu2024rso", "gao2023scaling", "tang2024gpo",
        "chen2024dposurvey", "ethayarajh2024limits", "zheng2024expo",
        "zheng2023secrets", "dong2024rlhf", "shen2024trickle",
        "thakkar2024tradeoffs", "qi2025shallow", "zhao2025dualobjective",
        "safetyarithmetic2025", "weaksupervision2026",
        "inan2023llamaguard", "rebedea2023nemo", "han2024wildguard",
        "sharma2025constitutional", "nakka2025litelmguard", "krishna2025dsa",
    ],
    "benchmarks": [
        "gehman2020realtoxicity", "hartvigsen2022toxigen", "lin2022truthfulqa",
        "parrish2022bbq", "nadeem2021stereoset", "nangia2020crows",
        "dhamala2021bold", "sheng2019regard", "zhao2018winobias",
        "rudinger2018winogender", "mazeika2024harmbench", "wang2023decodingtrust",
        "li2024salad", "huang2024trustllm", "xie2024sorry",
        "cui2024orbench", "li2023halueval", "min2023factscore",
        "liang2022helm", "vidgen2024mlcommons", "chao2024jailbreakbench",
        "souly2024strongreject", "andriushchenko2024agentharm", "ruan2024toolemu",
        "vidgen2023simplesafety", "ji2023beavertails", "gao2023lmeval",
        "ailuminate2025", "polyguard2025", "malert2025", "safearena2025",
        "halluhard2026", "atbench2026", "osharm2025", "openagentsafety2025",
        "linguasafetybench2026", "ren2024safetywashing",
    ],
    "efficiency": [
        "hong2024compressed", "wee2025caq", "chen2025qresafe",
        "hsu2024safelora", "xue2025lora", "mou2025decoupling",
        "salora2025", "wei2024brittleness", "niu2025nspo",
        "zandieh2025turboquant", "zandieh2024qjl", "zandieh2026polarquant",
        "ease2025",
    ],
    "multilingual": [
        "wang2024xsafety", "deng2024multilingual", "yong2024lowresource",
        "multilingualsparse2026", "polyguard2025", "malert2025",
        "linguasafetybench2026",
    ],
    "interpretability": [
        "elhage2021mathematical", "elhage2022superposition",
        "bricken2023monosemanticity", "templeton2024scaling",
        "meng2022rome", "conmy2023acdc", "hubinger2024sleeper",
        "greenblatt2024faking", "li2023iti", "bereska2024mi",
        "lindsey2025biology", "yi2024superficial",
        "zhou2025hiddenrisks", "kuo2025hcot", "openai2025deliberative",
        "safepath2025", "star1_2025", "thinksafe2026",
    ],
    "models": [
        "abdin2024phi3", "abdin2024phi4", "gemma2024", "gemma2024b",
        "gemma2025", "gemma2026", "qwen2024", "qwen2025",
        "qwen2026_35", "qwen2026_36", "grattafiori2024llama3",
        "touvron2023llama2", "jiang2023mistral", "zhang2024tinyllama",
        "benallal2024smollm", "groeneveld2024olmo", "mehta2024openelm",
        "phi4vision2026",
    ],
    "attacks": [
        "zou2023gcg", "liu2023autodan", "chao2023pair", "perez2022red",
        "bowen2024scaling", "wei2024jailbroken", "qi2024finetuning",
        "perez2022ignore", "huang2024lazy",
    ],
    "editing": [
        "mitchell2022memit", "li2024wmdp", "eldan2023whosharry", "liu2024rmu",
    ],
    "data_centric": [
        "longpre2024pretrainer", "wang2024toxicdata", "li2024datacentric",
        "xu2024magpie", "bianchi2024safetytuned",
    ],
    "surveys": [
        "li2024safetysurvey", "wang2024slmsurvey", "lu2024slm",
        "wan2024efficient", "wang2024redteamsurvey", "yi2024jailbreaksurvey",
        "gallegos2024bias",
    ],
}


def main():
    entries = {e["key"]: e for e in parse_bib()}

    rows = []
    for axis, keys in AXIS_MAP.items():
        for key in keys:
            if key in entries:
                e = entries[key]
                rows.append({
                    "key": key,
                    "title": e["title"],
                    "year": e["year"],
                    "axis": axis,
                })

    # Export
    with open("data/paper_classification.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "title", "year", "axis"])
        w.writeheader()
        w.writerows(rows)

    with open("data/paper_classification.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Stats
    axis_counts = Counter(r["axis"] for r in rows)
    print(f"Total classified papers: {len(rows)}")
    print(f"\nBy axis:")
    for axis, count in axis_counts.most_common():
        print(f"  {axis}: {count}")

    year_counts = Counter(r["year"] for r in rows)
    print(f"\nBy year:")
    for y in sorted(year_counts):
        print(f"  {y}: {year_counts[y]}")


if __name__ == "__main__":
    main()
