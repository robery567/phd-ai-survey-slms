"""Generate Figure 2: Timeline of alignment methods and safety milestones 2020-2026."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Data ---
events = [
    # (year_float, label, category)
    # Alignment methods
    (2020.0, "Learning to Summarize\n(RLHF foundations)", "alignment"),
    (2022.2, "InstructGPT\n(RLHF/PPO)", "alignment"),
    (2022.9, "Constitutional AI\n(RLAIF)", "alignment"),
    (2023.4, "DPO", "alignment"),
    (2023.8, "RepE\n(activation steering)", "alignment"),
    (2023.9, "IPO", "alignment"),
    (2024.1, "KTO", "alignment"),
    (2024.2, "ORPO", "alignment"),
    (2024.3, "SimPO", "alignment"),
    (2024.4, "SPIN / Self-Rewarding", "alignment"),
    (2024.5, "Nash-LHF / SPPO", "alignment"),
    (2025.0, "Null-space alignment\n(NSPO)", "alignment"),
    (2025.5, "Rank-1 LoRA safety", "alignment"),

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

    # Key attacks/findings
    (2023.6, "GCG attack", "attack"),
    (2023.8, "AutoDAN / PAIR", "attack"),
    (2024.0, "Sleeper Agents", "attack"),
    (2024.4, "Refusal direction\ndiscovery", "attack"),
    (2024.9, "Alignment faking", "attack"),
]

colors = {"alignment": "#2196F3", "benchmark": "#4CAF50", "model": "#FF9800", "attack": "#F44336"}
labels = {"alignment": "Alignment Methods", "benchmark": "Safety Benchmarks",
          "model": "Small Models", "attack": "Attacks & MI Findings"}
y_offsets = {"alignment": 1.8, "benchmark": 0.9, "model": -0.9, "attack": -1.8}

fig, ax = plt.subplots(figsize=(16, 7))

# Timeline axis
ax.axhline(y=0, color='gray', linewidth=1.5, zorder=1)
for year in range(2020, 2027):
    ax.axvline(x=year, color='lightgray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.text(year, -2.5, str(year), ha='center', fontsize=11, fontweight='bold')

# Plot events
for i, (x, label, cat) in enumerate(events):
    y_base = y_offsets[cat]
    # Stagger within category to avoid overlap
    same_cat = [(xx, ll) for xx, ll, cc in events[:i] if cc == cat and abs(xx - x) < 0.3]
    y = y_base + len(same_cat) * 0.4 * (1 if y_base > 0 else -1)

    ax.plot(x, 0, 'o', color=colors[cat], markersize=5, zorder=3)
    ax.plot([x, x], [0, y], color=colors[cat], linewidth=0.8, alpha=0.6, zorder=2)
    ax.text(x, y, label, ha='center', va='bottom' if y > 0 else 'top',
            fontsize=6.5, color=colors[cat], fontweight='medium',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=colors[cat], alpha=0.8, linewidth=0.5))

# Legend
patches = [mpatches.Patch(color=colors[k], label=labels[k]) for k in ["alignment", "benchmark", "model", "attack"]]
ax.legend(handles=patches, loc='upper left', fontsize=9, framealpha=0.9)

ax.set_xlim(2019.7, 2026.3)
ax.set_ylim(-4.5, 4.5)
ax.set_title("Evolution of Safety Alignment Methods, Benchmarks, and Small Language Models (2020–2026)",
             fontsize=12, fontweight='bold', pad=15)
ax.axis('off')

plt.tight_layout()
plt.savefig("figures/fig2-timeline.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/fig2-timeline.png", dpi=300, bbox_inches='tight')
print("Saved fig2-timeline.pdf and fig2-timeline.png")
