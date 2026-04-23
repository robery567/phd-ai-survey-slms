"""Generate Figure 3: Alignment methods comparison for SLMs.
Compares methods across memory cost, robustness, and SLM suitability.
"""
import matplotlib.pyplot as plt
import numpy as np

methods = ["RLHF/PPO", "DPO", "IPO", "KTO", "ORPO", "SimPO", "RLAIF", "RepE"]

# Scores 1-5 (higher = better for SLMs)
memory_efficiency = [1, 2, 2, 2, 4, 4, 2, 5]   # inverse of memory cost
robustness        = [5, 3, 3, 3, 3, 3, 3, 2]
slm_suitability   = [1, 4, 4, 4, 5, 5, 3, 5]
data_efficiency    = [1, 3, 3, 4, 3, 3, 5, 5]   # how little data needed

x = np.arange(len(methods))
w = 0.2

fig, ax = plt.subplots(figsize=(12, 5))
bars1 = ax.bar(x - 1.5*w, memory_efficiency, w, label='Memory Efficiency', color='#2196F3')
bars2 = ax.bar(x - 0.5*w, robustness, w, label='Robustness', color='#F44336')
bars3 = ax.bar(x + 0.5*w, slm_suitability, w, label='SLM Suitability', color='#4CAF50')
bars4 = ax.bar(x + 1.5*w, data_efficiency, w, label='Data Efficiency', color='#FF9800')

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('Score (1=Low, 5=High)', fontsize=11)
ax.set_title('Comparative Assessment of Alignment Methods for Small Language Models', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 6)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("figures/fig3-alignment-comparison.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/fig3-alignment-comparison.png", dpi=300, bbox_inches='tight')
print("Saved fig3-alignment-comparison.pdf/.png")
