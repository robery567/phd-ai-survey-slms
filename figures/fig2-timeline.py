"""Generate Figure 2: Timeline of alignment methods and safety milestones 2020-2026.

Redesigned for portrait orientation (no manual rotation in LaTeX):
- vertical (top-to-bottom) timeline with year labels on the left
- four parallel lanes: alignment methods, safety benchmarks, models, attacks/MI
- fits naturally as a page-width figure on a portrait page
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Data: (year_float, label, category) ---
events = [
    # Alignment methods
    (2020.0, "Learning to Summarize\n(RLHF foundations)", "alignment"),
    (2022.2, "InstructGPT (RLHF/PPO)", "alignment"),
    (2022.9, "Constitutional AI (RLAIF)", "alignment"),
    (2023.4, "DPO", "alignment"),
    (2023.8, "RepE (activation steering)", "alignment"),
    (2023.9, "IPO", "alignment"),
    (2024.1, "KTO", "alignment"),
    (2024.2, "ORPO", "alignment"),
    (2024.3, "SimPO", "alignment"),
    (2024.4, "SPIN / Self-Rewarding", "alignment"),
    (2024.5, "Nash-LHF / SPPO", "alignment"),
    (2025.0, "NSPO (null-space)", "alignment"),
    (2025.5, "Rank-1 LoRA safety", "alignment"),
    (2026.1, "ThinkSafe", "alignment"),

    # Safety benchmarks
    (2020.5, "RealToxicityPrompts", "benchmark"),
    (2022.0, "TruthfulQA / BBQ", "benchmark"),
    (2022.3, "ToxiGen", "benchmark"),
    (2023.5, "DecodingTrust", "benchmark"),
    (2023.9, "Llama Guard", "benchmark"),
    (2024.1, "HarmBench", "benchmark"),
    (2024.3, "XSafety / SALAD-Bench", "benchmark"),
    (2024.5, "JailbreakBench / StrongREJECT", "benchmark"),
    (2024.8, "SORRY-Bench / TrustLLM", "benchmark"),
    (2025.4, "PolyGuardPrompts / M-ALERT", "benchmark"),
    (2026.0, "HalluHard / ATBench", "benchmark"),

    # Small models
    (2023.4, "Phi-1 (1.3B)", "model"),
    (2023.6, "Llama 2 (7B)", "model"),
    (2023.8, "Mistral 7B", "model"),
    (2024.0, "Gemma (2B/7B)", "model"),
    (2024.2, "Phi-3 Mini (3.8B)", "model"),
    (2024.3, "Llama 3.2 (1B/3B)", "model"),
    (2024.5, "Gemma 2 / Qwen 2.5", "model"),
    (2024.9, "SmolLM2 (1.7B)", "model"),
    (2025.2, "Gemma 3 / Qwen 3", "model"),
    (2026.1, "Gemma 4 / Qwen 3.6", "model"),

    # Key attacks / MI findings
    (2023.6, "GCG attack", "attack"),
    (2023.8, "AutoDAN / PAIR", "attack"),
    (2024.0, "Sleeper Agents", "attack"),
    (2024.4, "Refusal direction discovery", "attack"),
    (2024.9, "Alignment faking", "attack"),
    (2025.6, "Shallow alignment (Qi+)", "attack"),
]

colors = {"alignment": "#2196F3", "benchmark": "#4CAF50",
          "model": "#FF9800", "attack": "#F44336"}
labels = {"alignment": "Alignment Methods", "benchmark": "Safety Benchmarks",
          "model": "Small Language Models", "attack": "Attacks & MI Findings"}

# Lane x-positions (left to right): alignment, benchmark, model, attack
lanes = {"alignment": 1, "benchmark": 2, "model": 3, "attack": 4}

# Sort events by year so vertical staggering looks clean
events_sorted = sorted(events, key=lambda e: e[0])

fig, ax = plt.subplots(figsize=(11, 14))

# Year axis (vertical, top-to-bottom)
years = list(range(2020, 2027))
y_min, y_max = 2019.7, 2026.6
for year in years:
    ax.axhline(y=year, color="lightgray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.text(0.3, year, str(year), ha="right", va="center",
            fontsize=12, fontweight="bold")

# Lane headers
for cat, x in lanes.items():
    ax.text(x, y_min - 0.15, labels[cat], ha="center", va="top",
            fontsize=11, fontweight="bold", color=colors[cat])

# Track stagger offsets per (lane, year-bucket) to avoid label overlap
stagger_map = {}
for y_year, label, cat in events_sorted:
    bucket = (cat, round(y_year * 4) / 4)  # 0.25-year buckets
    stagger = stagger_map.get(bucket, 0)
    stagger_map[bucket] = stagger + 1
    x = lanes[cat] + (stagger * 0.18 - 0.1)

    ax.plot(x, y_year, "o", color=colors[cat], markersize=6, zorder=3)
    # label slightly offset to the right of the marker
    ax.text(x + 0.07, y_year, label, ha="left", va="center",
            fontsize=8, color=colors[cat],
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="white", edgecolor=colors[cat],
                      alpha=0.9, linewidth=0.5))

# Lane lines
for cat, x in lanes.items():
    ax.plot([x, x], [y_min, y_max], color=colors[cat],
            linewidth=1.2, alpha=0.5, zorder=1)

# Legend (top-right)
patches = [mpatches.Patch(color=colors[k], label=labels[k])
           for k in ["alignment", "benchmark", "model", "attack"]]
ax.legend(handles=patches, loc="upper right",
          fontsize=10, framealpha=0.92, ncol=1)

ax.set_xlim(0.4, 5.1)
ax.set_ylim(y_max + 0.2, y_min - 0.6)  # invert: 2020 on top, 2026 on bottom
ax.set_title(
    "Evolution of Safety Alignment Methods, Benchmarks, and "
    "Small Language Models (2020-2026)",
    fontsize=12, fontweight="bold", pad=20,
)
ax.axis("off")

plt.tight_layout()
plt.savefig("figures/fig2-timeline.pdf", dpi=300, bbox_inches="tight")
plt.savefig("figures/fig2-timeline.png", dpi=300, bbox_inches="tight")
print("Saved fig2-timeline.pdf and fig2-timeline.png")
