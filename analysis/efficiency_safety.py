"""
Efficiency-safety interaction analysis.
Generates Table 4 (efficiency impacts) from the survey.
"""
import json

TECHNIQUES = [
    {
        "technique": "4-bit quantization (GPTQ/AWQ)",
        "safety_impact": "preserved",
        "capability_impact": "minimal (<1% degradation)",
        "memory_reduction": "4x",
        "evidence": "hong2024compressed",
        "recommendation": "Safe for deployment; evaluate post-quantization",
        "slm_relevance": "Primary deployment format for SLMs"
    },
    {
        "technique": "3-bit quantization",
        "safety_impact": "degraded",
        "capability_impact": "moderate (invisible from perplexity)",
        "memory_reduction": "5.3x",
        "evidence": "hong2024compressed",
        "recommendation": "Use CAQ or Q-resafe; re-evaluate safety",
        "slm_relevance": "Needed for sub-4GB deployment"
    },
    {
        "technique": "2.5-3.5 bit (TurboQuant)",
        "safety_impact": "unknown (quality-neutral but safety untested)",
        "capability_impact": "near-neutral at 3.5 bits",
        "memory_reduction": "4.6-6.4x",
        "evidence": "zandieh2025turboquant",
        "recommendation": "Promising but requires safety-specific evaluation",
        "slm_relevance": "Could enable smartphone deployment of 7B models"
    },
    {
        "technique": "LoRA fine-tuning (general)",
        "safety_impact": "risk of safety removal",
        "capability_impact": "task-specific improvement",
        "memory_reduction": "N/A (training efficiency)",
        "evidence": "qi2024finetuning",
        "recommendation": "Use Safe LoRA to preserve safety",
        "slm_relevance": "Primary fine-tuning method for SLMs"
    },
    {
        "technique": "LoRA safety alignment",
        "safety_impact": "effective (rank-1 suffices)",
        "capability_impact": "minimal with NSPO",
        "memory_reduction": "N/A",
        "evidence": "xue2025lora",
        "recommendation": "Recommended for SLM alignment",
        "slm_relevance": "Most practical alignment approach"
    },
    {
        "technique": "Pruning (WANDA, ≤20%)",
        "safety_impact": "can improve jailbreak resistance",
        "capability_impact": "moderate degradation",
        "memory_reduction": "1.25x",
        "evidence": "wei2024brittleness",
        "recommendation": "Safety-preserving at moderate levels",
        "slm_relevance": "Counter-intuitive safety improvement"
    },
    {
        "technique": "Pruning (>20%)",
        "safety_impact": "degraded",
        "capability_impact": "significant degradation",
        "memory_reduction": ">1.25x",
        "evidence": "wei2024brittleness",
        "recommendation": "Avoid for safety-critical applications",
        "slm_relevance": "Safety-critical regions are only ~3% of params"
    },
    {
        "technique": "Knowledge distillation (safety)",
        "safety_impact": "risky — can worsen multilingual safety",
        "capability_impact": "generally positive",
        "memory_reduction": "model-dependent",
        "evidence": "survey analysis",
        "recommendation": "Validate multilingual safety post-distillation",
        "slm_relevance": "Common SLM training approach"
    },
    {
        "technique": "Null-space alignment (NSPO)",
        "safety_impact": "state-of-the-art (zero capability cost)",
        "capability_impact": "mathematically zero",
        "memory_reduction": "N/A",
        "evidence": "niu2025nspo",
        "recommendation": "Best practice for safety alignment",
        "slm_relevance": "Eliminates alignment tax"
    }
]


def export(path="data/efficiency_safety.json"):
    with open(path, "w") as f:
        json.dump(TECHNIQUES, f, indent=2)
    print(f"Exported {len(TECHNIQUES)} techniques to {path}")


def print_summary():
    print(f"\n{'Technique':<35} {'Safety Impact':<35} {'Recommendation'}")
    print("-" * 105)
    for t in TECHNIQUES:
        print(f"{t['technique']:<35} {t['safety_impact']:<35} {t['recommendation']}")


if __name__ == "__main__":
    export()
    print_summary()
