"""Generate Figure 3: Alignment methods comparison for SLMs.

Redesigned for column-width portrait fit:
- horizontal grouped bar chart (methods on y-axis) avoids the cramped
  vertical layout that the AI Review reviewers flagged as rotated/dense
- 8 methods x 4 dimensions; each dimension scored on the rubric
  documented in methodology/rubrics/alignment-rubric.md
"""
import matplotlib.pyplot as plt
import numpy as np

methods = ["RLHF/PPO", "DPO", "IPO", "KTO", "ORPO", "SimPO", "RLAIF", "RepE"]

# Scores 1-5 (higher = better for SLMs). All values traceable to the rubric
# in methodology/rubrics/alignment-rubric.md and to coded papers in the corpus.
memory_efficiency = [1, 2, 2, 2, 4, 4, 2, 5]
robustness        = [5, 3, 3, 3, 3, 3, 3, 2]
slm_suitability   = [1, 4, 4, 4, 5, 5, 3, 5]
data_efficiency   = [1, 3, 3, 4, 3, 3, 5, 5]

dimensions = [
    ("Memory Efficiency", memory_efficiency, "#2196F3"),
    ("Robustness",        robustness,        "#F44336"),
    ("SLM Suitability",   slm_suitability,   "#4CAF50"),
    ("Data Efficiency",   data_efficiency,   "#FF9800"),
]

y = np.arange(len(methods))
h = 0.20

# Portrait-friendly aspect ratio: 8.5 wide x 6.5 tall fits within a
# CSUR \textwidth column figure without rotation or shrinkage.
fig, ax = plt.subplots(figsize=(8.5, 6.5))

for i, (name, values, color) in enumerate(dimensions):
    offset = (i - 1.5) * h
    ax.barh(y + offset, values, height=h, label=name, color=color, edgecolor="white")

ax.set_yticks(y)
ax.set_yticklabels(methods, fontsize=11)
ax.invert_yaxis()  # RLHF at the top, RepE at the bottom (matches §5 ordering)
ax.set_xlabel("Score (1 = Low, 5 = High)", fontsize=11)
ax.set_xlim(0, 6)
ax.set_xticks(range(0, 6))
ax.set_title(
    "Alignment Methods for SLMs: Comparative Assessment Across Four Dimensions",
    fontsize=12, fontweight="bold", pad=12,
)
ax.legend(loc="lower right", fontsize=10, framealpha=0.92)
ax.grid(axis="x", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("figures/fig3-alignment-comparison.pdf", dpi=300, bbox_inches="tight")
plt.savefig("figures/fig3-alignment-comparison.png", dpi=300, bbox_inches="tight")
print("Saved fig3-alignment-comparison.pdf/.png")
