"""Generate Figure 6: Empirical safety results across 19 model configurations."""
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), gridspec_kw={'width_ratios': [5, 1.2, 1]})

# --- Experiment 1: Size-stratified safety ---
labels_1 = [
    'Qwen2.5\n0.5B', 'Qwen2.5\n1.5B', 'Qwen2.5\n3B', 'Qwen2.5\n7B',
    'Llama3.2\n1B', 'Llama3.2\n3B',
    'Gemma2\n2B', 'Gemma3\n1B', 'Gemma3\n4B',
    'Qwen3\n0.6B', 'Qwen3\n1.7B', 'Qwen3\n4B',
    'Phi-4\n3.8B', 'SmolLM2\n1.7B',
]
vals_1 = [86.7, 93.3, 100, 100, 73.3, 60, 93.3, 86.7, 0, 40, 86.7, 93.3, 100, 46.7]
colors_1 = (
    ['#1565C0', '#1976D2', '#2196F3', '#42A5F5'] +  # Qwen 2.5
    ['#C62828', '#E53935'] +                          # Llama
    ['#2E7D32', '#43A047', '#66BB6A'] +               # Gemma
    ['#6A1B9A', '#8E24AA', '#AB47BC'] +               # Qwen 3
    ['#E65100'] +                                      # Phi-4
    ['#455A64']                                        # SmolLM2
)

ax1 = axes[0]
x1 = np.arange(len(labels_1))
bars = ax1.bar(x1, vals_1, color=colors_1, edgecolor='white', linewidth=0.5)
ax1.set_ylabel('Harmful Prompt Refusal Rate (%)', fontsize=10)
ax1.set_title('Exp 1: Size-Stratified Safety', fontsize=11, fontweight='bold')
ax1.set_xticks(x1)
ax1.set_xticklabels(labels_1, fontsize=7.5, rotation=0)
ax1.set_ylim(0, 115)
ax1.axhline(y=100, color='green', linestyle='--', alpha=0.3)
for bar, val in zip(bars, vals_1):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# --- Experiment 2: Quantization ---
labels_2 = ['fp16', 'int8', 'int4']
vals_2 = [100, 100, 100]
colors_2 = ['#2196F3', '#64B5F6', '#90CAF9']

ax2 = axes[1]
x2 = np.arange(len(labels_2))
bars2 = ax2.bar(x2, vals_2, color=colors_2, edgecolor='white', linewidth=0.5)
ax2.set_title('Exp 2: Quantization\n(Qwen2.5-3B)', fontsize=11, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(labels_2, fontsize=9)
ax2.set_ylim(0, 115)
ax2.axhline(y=100, color='green', linestyle='--', alpha=0.3)
for bar, val in zip(bars2, vals_2):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.set_yticklabels([])

# --- Experiment 3: Base vs Instruct ---
labels_3 = ['Qwen2.5\n3B-Base', 'Llama3.2\n3B-Base']
vals_3 = [26.7, 0]
colors_3 = ['#FF9800', '#FFCC80']

ax3 = axes[2]
x3 = np.arange(len(labels_3))
bars3 = ax3.bar(x3, vals_3, color=colors_3, edgecolor='white', linewidth=0.5)
ax3.set_title('Exp 3: Base\n(Unaligned)', fontsize=11, fontweight='bold')
ax3.set_xticks(x3)
ax3.set_xticklabels(labels_3, fontsize=8)
ax3.set_ylim(0, 115)
ax3.axhline(y=100, color='green', linestyle='--', alpha=0.3)
for bar, val in zip(bars3, vals_3):
    ax3.text(bar.get_x() + bar.get_width()/2, max(val, 2) + 1,
             f'{val:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.set_yticklabels([])

plt.suptitle('Empirical Safety Evaluation — Harmful Prompt Refusal Rate (n=15 harmful, n=10 benign)',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig("figures/fig6-empirical-results.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/fig6-empirical-results.png", dpi=300, bbox_inches='tight')
print("Saved fig6-empirical-results.pdf/.png")
