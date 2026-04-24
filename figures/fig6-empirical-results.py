"""Generate Figure 6: v3 HarmBench results across 49 configurations."""
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# Exp 1: HarmBench size-stratified
ax = axes[0, 0]
labels = ['Q2.5\n0.5B','Q2.5\n1.5B','Q2.5\n3B','Q2.5\n7B',
          'Ll3.2\n1B','Ll3.2\n3B','Gem2\n2B','Gem3\n1B','Gem3\n4B',
          'Q3\n0.6B','Q3\n1.7B','Q3\n4B','Phi4\n3.8B','Smol\n1.7B']
harmful = [49.7,38.4,41.5,38.4, 40.9,40.9, 40.3,49.1,40.3, 44.0,45.3,40.9, 42.8,39.6]
benign =  [54.0,36.0,42.0,47.0, 40.0,48.0, 40.0,50.0,45.0, 48.0,40.0,41.0, 48.0,50.0]
x = np.arange(len(labels))
ax.bar(x-0.2, harmful, 0.4, label='Harmful↑', color='#1976D2', alpha=0.85)
ax.bar(x+0.2, benign, 0.4, label='Benign↓', color='#FF9800', alpha=0.85)
ax.set_ylabel('Refusal Rate (%)', fontsize=9)
ax.set_title('Exp 1: HarmBench (159)', fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5)
ax.set_ylim(0, 70); ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=0.8)

# Exp 2: Quantization
ax = axes[0, 1]
h2 = [47.8, 45.9, 36.5]; b2 = [41.0, 45.0, 41.0]
x2 = np.arange(3)
ax.bar(x2-0.2, h2, 0.4, color='#1976D2', alpha=0.85)
ax.bar(x2+0.2, b2, 0.4, color='#FF9800', alpha=0.85)
ax.set_title('Exp 2: Quantization\n(Qwen2.5-3B)', fontsize=10, fontweight='bold')
ax.set_xticks(x2); ax.set_xticklabels(['fp16','int8','int4'], fontsize=9)
ax.set_ylim(0, 70); ax.grid(axis='y', alpha=0.3)

# Exp 3: Base
ax = axes[0, 2]
h3 = [44.7, 38.4]; b3 = [33.0, 53.0]
x3 = np.arange(2)
ax.bar(x3-0.2, h3, 0.4, color='#1976D2', alpha=0.85)
ax.bar(x3+0.2, b3, 0.4, color='#FF9800', alpha=0.85)
ax.set_title('Exp 3: Base\n(Unaligned)', fontsize=10, fontweight='bold')
ax.set_xticks(x3); ax.set_xticklabels(['Q2.5-3B\nBase','Ll3.2-3B\nBase'], fontsize=8)
ax.set_ylim(0, 70); ax.grid(axis='y', alpha=0.3)

# Exp 4: Multilingual heatmap
ax = axes[0, 3]
scores4 = np.array([
    [-0.5, -6.7, 13.3, 20.0, 6.7],
    [-7.1, -6.7, -6.7, 26.7, 13.3],
    [-0.1, -13.3, 6.7, -13.3, 6.7],
    [-5.2, 6.7, 0.0, 0.0, -13.3],
])
im = ax.imshow(scores4, cmap='RdYlGn', vmin=-20, vmax=30, aspect='auto')
ax.set_xticks(range(5)); ax.set_xticklabels(['EN','RO','ZH','ES','AR'], fontsize=8)
ax.set_yticks(range(4)); ax.set_yticklabels(['Q2.5-3B','Ll3.2-3B','Q3-4B','Phi4'], fontsize=8)
ax.set_title('Exp 4: Multilingual\n(Safety Score %)', fontsize=10, fontweight='bold')
for i in range(4):
    for j in range(5):
        ax.text(j, i, f'{scores4[i,j]:.0f}', ha='center', va='center', fontsize=7, fontweight='bold')

# Exp 5: Jailbreak
ax = axes[1, 0]
models5 = ['Q2.5-3B','Ll3.2-3B','Q3-4B','Phi4']
direct = [41.5, 40.9, 40.9, 42.8]
jb = [42.0, 44.0, 44.0, 43.0]
x5 = np.arange(4)
ax.bar(x5-0.2, direct, 0.4, label='Direct', color='#1976D2', alpha=0.85)
ax.bar(x5+0.2, jb, 0.4, label='Jailbreak', color='#F44336', alpha=0.85)
ax.set_title('Exp 5: Jailbreak', fontsize=10, fontweight='bold')
ax.set_xticks(x5); ax.set_xticklabels(models5, fontsize=8)
ax.set_ylim(0, 70); ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
ax.set_ylabel('Harmful Refusal (%)', fontsize=9)

# Exp 6: Multi-turn
ax = axes[1, 1]
mt = [10, 60, 10, 50]
ax.bar(x5, mt, 0.6, color=['#F44336','#4CAF50','#F44336','#FF9800'], alpha=0.85)
ax.set_title('Exp 6: Multi-Turn\nCrescendo', fontsize=10, fontweight='bold')
ax.set_xticks(x5); ax.set_xticklabels(models5, fontsize=8)
ax.set_ylim(0, 80); ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(mt):
    ax.text(i, v+2, f'{v}%', ha='center', fontsize=8, fontweight='bold')

# Exp 7: Over-refusal
ax = axes[1, 2]
or_rates = [42, 42, 40, 53]
ax.bar(x5, or_rates, 0.6, color='#FF9800', alpha=0.85)
ax.set_title('Exp 7: Over-Refusal\n(JBB benign, lower=better)', fontsize=10, fontweight='bold')
ax.set_xticks(x5); ax.set_xticklabels(models5, fontsize=8)
ax.set_ylim(0, 70); ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(or_rates):
    ax.text(i, v+1, f'{v}%', ha='center', fontsize=8, fontweight='bold')

# Exp 8: Decoding + Variance
ax = axes[1, 3]
ax.bar([0,1], [41.5, 42.0], 0.6, color=['#1976D2','#9C27B0'], alpha=0.85)
ax.errorbar(0, 41.5, yerr=1.8, fmt='none', color='black', capsize=5, linewidth=2)
ax.set_title('Exp 8: Decoding &\nVariance', fontsize=10, fontweight='bold')
ax.set_xticks([0,1]); ax.set_xticklabels(['Greedy\n(Q2.5-3B)','Sampling\n(Q2.5-3B)'], fontsize=8)
ax.set_ylim(0, 60); ax.grid(axis='y', alpha=0.3)
ax.text(0, 44, '41.5±1.8%', ha='center', fontsize=8, fontweight='bold')
ax.text(1, 44, '42.0%', ha='center', fontsize=8, fontweight='bold')

plt.suptitle('HarmBench Safety Evaluation — 49 Configs, 8 Experiments, Llama Guard 3-1B Judge',
             fontsize=12, y=1.01, fontweight='bold')
plt.tight_layout()
plt.savefig("figures/fig6-empirical-results.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/fig6-empirical-results.png", dpi=300, bbox_inches='tight')
print("Saved")
