"""Generate Figure 6: Empirical safety results across model sizes."""
import matplotlib.pyplot as plt
import numpy as np

# Results from our experiments
models = ["0.5B\nInstruct", "1.5B\nInstruct", "3B\nInstruct", "3B\nBase"]
harmful_refusal = [93.3, 96.7, 100.0, 26.7]
benign_refusal = [0.0, 10.0, 0.0, 0.0]

x = np.arange(len(models))
w = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Harmful refusal rate (higher = safer)
bars1 = ax1.bar(x, harmful_refusal, w*2, color=['#2196F3', '#1976D2', '#0D47A1', '#FF9800'])
ax1.set_ylabel('Harmful Prompt Refusal Rate (%)', fontsize=11)
ax1.set_title('Safety: Harmful Prompt Refusal', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
ax1.set_ylim(0, 110)
ax1.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='Perfect safety')
ax1.legend(fontsize=9)
for bar, val in zip(bars1, harmful_refusal):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Right: Over-refusal rate (lower = better)
bars2 = ax2.bar(x, benign_refusal, w*2, color=['#4CAF50', '#F44336', '#4CAF50', '#4CAF50'])
ax2.set_ylabel('Benign Prompt Refusal Rate (%)', fontsize=11)
ax2.set_title('Usability: Over-Refusal on Benign Prompts', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=10)
ax2.set_ylim(0, 25)
for bar, val in zip(bars2, benign_refusal):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Qwen 2.5 Family — Safety Probe Evaluation (n=15 harmful, n=10 benign)',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig("figures/fig6-empirical-results.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/fig6-empirical-results.png", dpi=300, bbox_inches='tight')
print("Saved fig6-empirical-results.pdf/.png")
