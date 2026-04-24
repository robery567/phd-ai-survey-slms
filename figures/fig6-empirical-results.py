"""Generate Figure 6: Empirical safety results across 48 model configurations (v2)."""
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# === Row 1: Exp 1 (size-stratified), Exp 2 (quant), Exp 3 (base), Exp 4 (multilingual) ===

# Exp 1: Size-stratified
ax = axes[0, 0]
labels = ['Q2.5\n0.5B','Q2.5\n1.5B','Q2.5\n3B','Q2.5\n7B',
          'Ll3.2\n1B','Ll3.2\n3B','Gem2\n2B','Gem3\n1B','Gem3\n4B',
          'Q3\n0.6B','Q3\n1.7B','Q3\n4B','Phi4\n3.8B','Smol\n1.7B']
harmful = [70,100,96.7,96.7, 83.3,96.7, 100,100,46.7, 86.7,90,100, 90,76.7]
benign =  [53.3,66.7,73.3,66.7, 73.3,53.3, 46.7,66.7,20, 53.3,60,53.3, 66.7,66.7]
x = np.arange(len(labels))
ax.bar(x - 0.2, harmful, 0.4, label='Harmful↑', color='#1976D2', alpha=0.85)
ax.bar(x + 0.2, benign, 0.4, label='Benign↓', color='#FF9800', alpha=0.85)
ax.set_ylabel('Refusal Rate (%)', fontsize=9)
ax.set_title('Exp 1: Size-Stratified', fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5)
ax.set_ylim(0, 115); ax.legend(fontsize=7, loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Exp 2: Quantization
ax = axes[0, 1]
labels2 = ['fp16', 'int8', 'int4']
h2 = [96.7, 100, 96.7]; b2 = [73.3, 73.3, 53.3]
x2 = np.arange(3)
ax.bar(x2 - 0.2, h2, 0.4, color='#1976D2', alpha=0.85)
ax.bar(x2 + 0.2, b2, 0.4, color='#FF9800', alpha=0.85)
ax.set_title('Exp 2: Quantization\n(Qwen2.5-3B)', fontsize=10, fontweight='bold')
ax.set_xticks(x2); ax.set_xticklabels(labels2, fontsize=9)
ax.set_ylim(0, 115); ax.grid(axis='y', alpha=0.3)
for i, (h, b) in enumerate(zip(h2, b2)):
    ax.text(i, 108, f'S:{h-b:.0f}%', ha='center', fontsize=7, fontweight='bold')

# Exp 3: Base vs Instruct
ax = axes[0, 2]
labels3 = ['Q2.5-3B\nBase','Ll3.2-3B\nBase']
h3 = [80, 83.3]; b3 = [60, 93.3]
x3 = np.arange(2)
ax.bar(x3 - 0.2, h3, 0.4, color='#1976D2', alpha=0.85)
ax.bar(x3 + 0.2, b3, 0.4, color='#FF9800', alpha=0.85)
ax.set_title('Exp 3: Base\n(Unaligned)', fontsize=10, fontweight='bold')
ax.set_xticks(x3); ax.set_xticklabels(labels3, fontsize=8)
ax.set_ylim(0, 115); ax.grid(axis='y', alpha=0.3)
for i, (h, b) in enumerate(zip(h3, b3)):
    ax.text(i, 108, f'S:{h-b:.0f}%', ha='center', fontsize=7, fontweight='bold',
            color='#F44336' if h-b < 0 else '#333')

# Exp 4: Multilingual heatmap
ax = axes[0, 3]
models4 = ['Q2.5-3B', 'Ll3.2-3B', 'Q3-4B', 'Phi4']
langs = ['EN', 'RO', 'ZH', 'ES', 'AR']
scores4 = np.array([
    [23.3, 0, 20, 26.7, 20],
    [43.3, 26.7, 46.7, 46.7, 46.7],
    [46.7, 33.3, 53.3, 26.7, 40],
    [23.3, 13.3, 33.3, 0, 26.7],
])
im = ax.imshow(scores4, cmap='RdYlGn', vmin=-10, vmax=60, aspect='auto')
ax.set_xticks(range(5)); ax.set_xticklabels(langs, fontsize=8)
ax.set_yticks(range(4)); ax.set_yticklabels(models4, fontsize=8)
ax.set_title('Exp 4: Multilingual\n(Safety Score)', fontsize=10, fontweight='bold')
for i in range(4):
    for j in range(5):
        ax.text(j, i, f'{scores4[i,j]:.0f}', ha='center', va='center', fontsize=7, fontweight='bold')

# === Row 2: Exp 5 (jailbreak), Exp 6 (HarmBench), Exp 7 (over-refusal), Exp 8 (decoding) ===

# Exp 5: Jailbreak
ax = axes[1, 0]
models5 = ['Q2.5-3B', 'Ll3.2-3B', 'Q3-4B', 'Phi4']
direct = [96.7, 96.7, 100, 90]  # from Exp 1
jb =     [93.3, 80, 96.7, 96.7]
x5 = np.arange(4)
ax.bar(x5 - 0.2, direct, 0.4, label='Direct', color='#1976D2', alpha=0.85)
ax.bar(x5 + 0.2, jb, 0.4, label='Jailbreak', color='#F44336', alpha=0.85)
ax.set_title('Exp 5: Jailbreak\nResistance', fontsize=10, fontweight='bold')
ax.set_xticks(x5); ax.set_xticklabels(models5, fontsize=8)
ax.set_ylim(0, 115); ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
ax.set_ylabel('Harmful Refusal (%)', fontsize=9)

# Exp 6: HarmBench
ax = axes[1, 1]
hb = [100, 100, 100, 100]
ax.bar(x5, hb, 0.6, color='#4CAF50', alpha=0.85)
ax.set_title('Exp 6: HarmBench\nIndirect Attacks', fontsize=10, fontweight='bold')
ax.set_xticks(x5); ax.set_xticklabels(models5, fontsize=8)
ax.set_ylim(0, 115); ax.grid(axis='y', alpha=0.3)
ax.text(1.5, 105, '100% refusal across all models', ha='center', fontsize=8, style='italic')

# Exp 7: Over-refusal
ax = axes[1, 2]
or_rates = [70, 50, 70, 60]
ax.bar(x5, or_rates, 0.6, color='#FF9800', alpha=0.85)
ax.set_title('Exp 7: Over-Refusal\n(Borderline, lower=better)', fontsize=10, fontweight='bold')
ax.set_xticks(x5); ax.set_xticklabels(models5, fontsize=8)
ax.set_ylim(0, 100); ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(or_rates):
    ax.text(i, v + 2, f'{v}%', ha='center', fontsize=8, fontweight='bold')

# Exp 8: Decoding
ax = axes[1, 3]
labels8 = ['Greedy\nHarmful','Greedy\nBenign','Sample\nHarmful','Sample\nBenign']
vals8 = [96.7, 73.3, 96.7, 13.3]
colors8 = ['#1976D2', '#FF9800', '#1976D2', '#4CAF50']
ax.bar(range(4), vals8, color=colors8, alpha=0.85)
ax.set_title('Exp 8: Decoding\n(Qwen2.5-3B)', fontsize=10, fontweight='bold')
ax.set_xticks(range(4)); ax.set_xticklabels(labels8, fontsize=7.5)
ax.set_ylim(0, 115); ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(vals8):
    ax.text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=8, fontweight='bold')
ax.annotate('Score: 23%', xy=(0.5, 85), fontsize=8, ha='center', color='#666')
ax.annotate('Score: 83%', xy=(2.5, 85), fontsize=8, ha='center', color='#4CAF50', fontweight='bold')

plt.suptitle('Empirical Safety Evaluation — 48 Configurations, 8 Experiments (v2)',
             fontsize=12, y=1.01, fontweight='bold')
plt.tight_layout()
plt.savefig("figures/fig6-empirical-results.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/fig6-empirical-results.png", dpi=300, bbox_inches='tight')
print("Saved fig6-empirical-results.pdf/.png")
