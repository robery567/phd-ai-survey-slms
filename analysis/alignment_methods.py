"""
Alignment methods dataset and comparative analysis.
Generates Table 2 (alignment comparison) from the survey.
"""
import json
import csv

METHODS = [
    {
        "method": "RLHF/PPO",
        "family": "reward-based",
        "reward_model": True,
        "paired_data": True,
        "reference_model": True,
        "compute": "high",
        "memory_gb_7b": 56,
        "slm_fit": "low",
        "robustness": "high",
        "key_paper": "ouyang2022training",
        "year": 2022,
        "strengths": [
            "Most robust alignment when well-executed",
            "Explicit reward model captures nuanced preferences",
            "KL constraint prevents catastrophic forgetting"
        ],
        "weaknesses": [
            "4 models in memory simultaneously",
            "Reward hacking via overoptimization",
            "Complex implementation with many hyperparameters"
        ]
    },
    {
        "method": "DPO",
        "family": "direct-preference",
        "reward_model": False,
        "paired_data": True,
        "reference_model": True,
        "compute": "low",
        "memory_gb_7b": 28,
        "slm_fit": "high",
        "robustness": "medium",
        "key_paper": "rafailov2023direct",
        "year": 2023,
        "strengths": [
            "Simple supervised learning formulation",
            "Stable training, half the memory of RLHF",
            "LoRA-compatible for parameter efficiency"
        ],
        "weaknesses": [
            "Reference model doubles memory",
            "Bypasses rather than removes toxic capabilities",
            "Shallow alignment — first few tokens only"
        ]
    },
    {
        "method": "IPO",
        "family": "direct-preference",
        "reward_model": False,
        "paired_data": True,
        "reference_model": True,
        "compute": "low",
        "memory_gb_7b": 28,
        "slm_fit": "high",
        "robustness": "medium",
        "key_paper": "azar2023general",
        "year": 2023,
        "strengths": [
            "Squared loss prevents overfitting",
            "More stable on small/noisy datasets"
        ],
        "weaknesses": [
            "Slightly lower peak performance than DPO",
            "Still requires reference model"
        ]
    },
    {
        "method": "KTO",
        "family": "direct-preference",
        "reward_model": False,
        "paired_data": False,
        "reference_model": True,
        "compute": "low",
        "memory_gb_7b": 28,
        "slm_fit": "high",
        "robustness": "medium",
        "key_paper": "ethayarajh2024kto",
        "year": 2024,
        "strengths": [
            "Binary feedback — dramatically cheaper data",
            "Grounded in prospect theory"
        ],
        "weaknesses": [
            "Lower quality than paired preference methods",
            "Theoretical grounding not empirically validated as mechanism"
        ]
    },
    {
        "method": "ORPO",
        "family": "reference-free",
        "reward_model": False,
        "paired_data": True,
        "reference_model": False,
        "compute": "very_low",
        "memory_gb_7b": 14,
        "slm_fit": "very_high",
        "robustness": "medium",
        "key_paper": "hong2024orpo",
        "year": 2024,
        "strengths": [
            "No reference model — lowest memory",
            "Single-phase SFT+alignment training"
        ],
        "weaknesses": [
            "Merged objective can create tension",
            "Less fine-grained safety control"
        ]
    },
    {
        "method": "SimPO",
        "family": "reference-free",
        "reward_model": False,
        "paired_data": True,
        "reference_model": False,
        "compute": "very_low",
        "memory_gb_7b": 14,
        "slm_fit": "very_high",
        "robustness": "medium",
        "key_paper": "meng2024simpo",
        "year": 2024,
        "strengths": [
            "Reference-free with sequence-level reward",
            "Memory-efficient"
        ],
        "weaknesses": [
            "Implicit reward may miss fine-grained safety distinctions"
        ]
    },
    {
        "method": "RLAIF",
        "family": "self-alignment",
        "reward_model": True,
        "paired_data": True,
        "reference_model": True,
        "compute": "medium",
        "memory_gb_7b": 28,
        "slm_fit": "medium",
        "robustness": "medium",
        "key_paper": "lee2023rlaif",
        "year": 2023,
        "strengths": [
            "No human labels needed",
            "Leverages frontier model judgment"
        ],
        "weaknesses": [
            "Bounded by teacher model quality",
            "Propagates teacher blind spots"
        ]
    },
    {
        "method": "RepE",
        "family": "representation",
        "reward_model": False,
        "paired_data": False,
        "reference_model": False,
        "compute": "minimal",
        "memory_gb_7b": 0,
        "slm_fit": "very_high",
        "robustness": "low",
        "key_paper": "zou2023representation",
        "year": 2023,
        "strengths": [
            "Zero training cost",
            "Works on quantized models",
            "Interpretable geometric mechanism"
        ],
        "weaknesses": [
            "Coarse safety signal — single direction",
            "Untested on sub-3B models",
            "Over-refusal if steering too aggressive"
        ]
    }
]


def export_csv(path="data/alignment_methods.csv"):
    fields = ["method", "family", "reward_model", "paired_data",
              "reference_model", "compute", "memory_gb_7b",
              "slm_fit", "robustness", "key_paper", "year"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in METHODS:
            w.writerow({k: m[k] for k in fields})
    print(f"Exported {len(METHODS)} methods to {path}")


def export_json(path="data/alignment_methods.json"):
    with open(path, "w") as f:
        json.dump(METHODS, f, indent=2)
    print(f"Exported {len(METHODS)} methods to {path}")


def print_comparison():
    print(f"\n{'Method':<12} {'Compute':<10} {'Mem(GB)':<8} {'SLM Fit':<10} {'Robust.':<8}")
    print("-" * 52)
    for m in METHODS:
        print(f"{m['method']:<12} {m['compute']:<10} {m['memory_gb_7b']:<8} "
              f"{m['slm_fit']:<10} {m['robustness']:<8}")


if __name__ == "__main__":
    export_csv()
    export_json()
    print_comparison()
