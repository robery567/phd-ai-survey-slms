"""Generate Figure 6: v3 HarmBench results across 49 configurations.

Redesigned for full-page portrait layout — no rotation needed.
Uses a 4×2 grid (4 rows, 2 columns), compact enough to fit on one
A4 page (2.5 cm margins) with its caption visible below.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# --- Global style ---
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'axes.titlesize': 8.5,
    'axes.titlepad': 4,
    'axes.labelsize': 7.5,
    'axes.labelpad': 3,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 7,
    'xtick.major.pad': 2,
    'ytick.major.pad': 2,
    'legend.fontsize': 6.5,
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette — accessible, publication-quality
C_HARMFUL = '#2166AC'   # strong blue
C_BENIGN  = '#F4A582'   # warm salmon
C_JAILBRK = '#B2182B'   # deep red
C_MULTI   = '#4DAF4A'   # green
C_ACCENT  = '#984EA3'   # purple
C_WARN    = '#FF7F00'   # orange

fig, axes = plt.subplots(4, 2, figsize=(6.8, 8.0))
fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.04,
                    hspace=0.52, wspace=0.34)

w = 0.33  # bar width

# ═══════════════════════════════════════════════════════════════════
# (a) Exp 1 — left half of 14 models
# ═══════════════════════════════════════════════════════════════════
ax = axes[0, 0]
labels_1a = ['Q2.5\n0.5B','Q2.5\n1.5B','Q2.5\n3B','Q2.5\n7B',
             'Ll3.2\n1B','Ll3.2\n3B','Gem2\n2B']
harmful_1a = [49.7, 38.4, 41.5, 38.4, 40.9, 40.9, 40.3]
benign_1a  = [54.0, 36.0, 42.0, 47.0, 40.0, 48.0, 40.0]
x = np.arange(len(labels_1a))
ax.bar(x - w/2, harmful_1a, w, label='Harmful ref. ↑', color=C_HARMFUL, alpha=0.9)
ax.bar(x + w/2, benign_1a,  w, label='Benign ref. ↓',  color=C_BENIGN,  alpha=0.9)
ax.set_ylabel('Refusal Rate (%)')
ax.set_title('(a) Exp 1: HarmBench — Qwen 2.5 / Llama / Gemma 2', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels_1a, fontsize=5.5)
ax.set_ylim(0, 68)
ax.axhline(y=50, color='grey', linestyle=':', alpha=0.5, linewidth=0.6)
ax.legend(loc='upper right', framealpha=0.9, handlelength=1.2)
ax.grid(axis='y', alpha=0.12)

# ═══════════════════════════════════════════════════════════════════
# (b) Exp 1 — right half of 14 models
# ═══════════════════════════════════════════════════════════════════
ax = axes[0, 1]
labels_1b = ['Gem3\n1B','Gem3\n4B','Q3\n0.6B','Q3\n1.7B','Q3\n4B',
             'Phi4\n3.8B','Smol\n1.7B']
harmful_1b = [49.1, 40.3, 44.0, 45.3, 40.9, 42.8, 39.6]
benign_1b  = [50.0, 45.0, 48.0, 40.0, 41.0, 48.0, 50.0]
x = np.arange(len(labels_1b))
ax.bar(x - w/2, harmful_1b, w, label='Harmful ref. ↑', color=C_HARMFUL, alpha=0.9)
ax.bar(x + w/2, benign_1b,  w, label='Benign ref. ↓',  color=C_BENIGN,  alpha=0.9)
ax.set_title('(b) Exp 1: HarmBench — Gemma 3 / Qwen 3 / Others', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels_1b, fontsize=5.5)
ax.set_ylim(0, 68)
ax.axhline(y=50, color='grey', linestyle=':', alpha=0.5, linewidth=0.6)
ax.legend(loc='upper right', framealpha=0.9, handlelength=1.2)
ax.grid(axis='y', alpha=0.12)

# ═══════════════════════════════════════════════════════════════════
# (c) Exp 2: Quantization
# ═══════════════════════════════════════════════════════════════════
ax = axes[1, 0]
h2 = [47.8, 45.9, 36.5]; b2 = [41.0, 45.0, 41.0]
x2 = np.arange(3)
ax.bar(x2 - w/2, h2, w, color=C_HARMFUL, alpha=0.9)
ax.bar(x2 + w/2, b2, w, color=C_BENIGN,  alpha=0.9)
ax.set_title('(c) Exp 2: Quantization (Qwen 2.5-3B)', fontweight='bold')
ax.set_xticks(x2); ax.set_xticklabels(['FP16', 'INT8', 'INT4'], fontsize=8)
ax.set_ylim(0, 60); ax.set_ylabel('Refusal Rate (%)')
ax.grid(axis='y', alpha=0.12)
ax.annotate('−11.3 pp', xy=(2, 36.5), xytext=(2.25, 50),
            fontsize=6.5, color=C_JAILBRK, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_JAILBRK, lw=1.0))

# ═══════════════════════════════════════════════════════════════════
# (d) Exp 3: Base vs Instruct
# ═══════════════════════════════════════════════════════════════════
ax = axes[1, 1]
h3 = [44.7, 38.4]; b3 = [33.0, 53.0]
scores3 = [11.7, -14.6]
x3 = np.arange(2)
ax.bar(x3 - w/2, h3, w, label='Harmful ↑', color=C_HARMFUL, alpha=0.9)
ax.bar(x3 + w/2, b3, w, label='Benign ↓',  color=C_BENIGN,  alpha=0.9)
for i, s in enumerate(scores3):
    clr = C_MULTI if s > 0 else C_JAILBRK
    ax.text(i, max(h3[i], b3[i]) + 1.5, f'Score: {s:+.1f}%', ha='center',
            fontsize=6.5, fontweight='bold', color=clr)
ax.set_title('(d) Exp 3: Base vs. Instruct (Unaligned)', fontweight='bold')
ax.set_xticks(x3); ax.set_xticklabels(['Qwen 2.5-3B Base', 'Llama 3.2-3B Base'], fontsize=7)
ax.set_ylim(0, 68); ax.legend(loc='upper left', framealpha=0.9, handlelength=1.2)
ax.grid(axis='y', alpha=0.12)

# ═══════════════════════════════════════════════════════════════════
# (e) Exp 4: Multilingual heatmap
# ═══════════════════════════════════════════════════════════════════
ax = axes[2, 0]
scores4 = np.array([
    [-0.5, -6.7, 13.3, 20.0, 6.7],
    [-7.1, -6.7, -6.7, 26.7, 13.3],
    [-0.1, -13.3, 6.7, -13.3, 6.7],
    [-5.2, 6.7, 0.0, 0.0, -13.3],
])
im = ax.imshow(scores4, cmap='RdYlGn', vmin=-20, vmax=30, aspect='auto')
ax.set_xticks(range(5)); ax.set_xticklabels(['EN', 'RO', 'ZH', 'ES', 'AR'], fontsize=7.5)
ax.set_yticks(range(4)); ax.set_yticklabels(['Q2.5-3B', 'Ll3.2-3B', 'Q3-4B', 'Phi-4'], fontsize=6.5)
ax.set_title('(e) Exp 4: Multilingual Safety Score (%)', fontweight='bold')
for i in range(4):
    for j in range(5):
        val = scores4[i, j]
        txt_color = 'white' if abs(val) > 12 else 'black'
        ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                fontsize=6.5, fontweight='bold', color=txt_color)

# ═══════════════════════════════════════════════════════════════════
# (f) Exp 5: Jailbreak
# ═══════════════════════════════════════════════════════════════════
ax = axes[2, 1]
models5 = ['Q2.5-3B', 'Ll3.2-3B', 'Q3-4B', 'Phi-4']
direct = [41.5, 40.9, 40.9, 42.8]
jb     = [42.0, 44.0, 44.0, 43.0]
x5 = np.arange(4)
ax.bar(x5 - w/2, direct, w, label='Direct', color=C_HARMFUL, alpha=0.9)
ax.bar(x5 + w/2, jb,     w, label='Jailbreak', color=C_JAILBRK, alpha=0.9)
ax.set_title('(f) Exp 5: Jailbreak Resistance', fontweight='bold')
ax.set_xticks(x5); ax.set_xticklabels(models5, fontsize=7)
ax.set_ylim(0, 58); ax.set_ylabel('Harmful Refusal (%)')
ax.legend(loc='upper right', framealpha=0.9, handlelength=1.2)
ax.grid(axis='y', alpha=0.12)

# ═══════════════════════════════════════════════════════════════════
# (g) Exp 6: Multi-turn crescendo
# ═══════════════════════════════════════════════════════════════════
ax = axes[3, 0]
mt = [10, 60, 10, 50]
colors_mt = [C_JAILBRK, C_MULTI, C_JAILBRK, C_WARN]
ax.bar(x5, mt, 0.50, color=colors_mt, alpha=0.9)
ax.set_title('(g) Exp 6: Multi-Turn Crescendo Attack', fontweight='bold')
ax.set_xticks(x5); ax.set_xticklabels(models5, fontsize=7)
ax.set_ylim(0, 76); ax.set_ylabel('Refusal Rate (%)')
ax.grid(axis='y', alpha=0.12)
for i, v in enumerate(mt):
    ax.text(i, v + 1.5, f'{v}%', ha='center', fontsize=7, fontweight='bold', color=colors_mt[i])
ax.axhline(y=30, color='grey', linestyle=':', alpha=0.35, linewidth=0.6)
ax.text(3.4, 26, 'High risk ↓', fontsize=5.5, color='grey', ha='right', style='italic')

# ═══════════════════════════════════════════════════════════════════
# (h) Exp 7: Over-refusal
# ═══════════════════════════════════════════════════════════════════
ax = axes[3, 1]
or_rates = [42, 42, 40, 53]
ax.bar(x5, or_rates, 0.50, color=C_WARN, alpha=0.85,
       label='Over-refusal rate (↓ better)')
ax.set_title('(h) Exp 7: Over-Refusal (JBB Benign)', fontweight='bold')
ax.set_xticks(x5); ax.set_xticklabels(models5, fontsize=7)
ax.set_ylim(0, 68); ax.set_ylabel('Benign Refusal Rate (%)')
ax.grid(axis='y', alpha=0.12)
for i, v in enumerate(or_rates):
    ax.text(i, v + 1.2, f'{v}%', ha='center', fontsize=7, fontweight='bold', color='#5D4037')
ax.axhline(y=50, color='grey', linestyle=':', alpha=0.35, linewidth=0.6)
ax.text(3.4, 51, '50%', fontsize=5.5, color='grey', ha='right', style='italic')
ax.legend(loc='upper left', framealpha=0.9, handlelength=1.2)

# ═══════════════════════════════════════════════════════════════════
# Super-title
# ═══════════════════════════════════════════════════════════════════
fig.suptitle('HarmBench Safety Evaluation — 49 Configurations, 8 Experiments, Llama Guard 3-1B Judge',
             fontsize=9.5, fontweight='bold')

plt.savefig("figures/fig6-empirical-results.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/fig6-empirical-results.png", dpi=300, bbox_inches='tight')
print("Saved fig6-empirical-results.pdf and .png")
