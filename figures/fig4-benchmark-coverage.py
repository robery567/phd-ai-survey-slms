"""Generate Figure 4: Safety benchmark coverage gap analysis.
Shows which safety dimensions are covered by which benchmarks,
highlighting gaps for SLMs.
"""
import matplotlib.pyplot as plt
import numpy as np

benchmarks = [
    "RealToxicity.", "ToxiGen", "TruthfulQA", "BBQ",
    "DecodingTrust", "HarmBench", "SALAD-Bench", "XSafety",
    "JailbreakB.", "StrongREJECT", "OR-Bench", "SORRY-Bench",
    "AILuminate", "PolyGuard", "M-ALERT", "SafeArena",
    "HalluHard", "ATBench"
]

dimensions = [
    "Toxicity", "Bias", "Truthful.", "Jailbreak",
    "Over-refusal", "Multilingual", "Agent", "SLM-tested",
    "Compress.-aware"
]

# Coverage matrix: 1 = covered, 0.5 = partial, 0 = not covered
# Rows = benchmarks, Cols = dimensions
data = np.array([
    # Tox  Bias Truth Jail  ORef  Multi Agent SLM   Comp
    [1,    0,   0,    0,    0,    0,    0,    0,    0],    # RealToxicity
    [1,    0.5, 0,    0,    0,    0,    0,    0,    0],    # ToxiGen
    [0,    0,   1,    0,    0,    0,    0,    1,    0],    # TruthfulQA
    [0,    1,   0,    0,    0,    0,    0,    0,    0],    # BBQ
    [1,    1,   0.5,  1,    0,    0,    0,    0,    0],    # DecodingTrust
    [1,    0,   0,    1,    0,    0,    0,    1,    0],    # HarmBench
    [1,    1,   0,    1,    0,    0,    0,    0,    0],    # SALAD-Bench
    [1,    0,   0,    0.5,  0,    1,    0,    0,    0],    # XSafety
    [0,    0,   0,    1,    0,    0,    0,    1,    0],    # JailbreakBench
    [0,    0,   0,    1,    0.5,  0,    0,    1,    0],    # StrongREJECT
    [0,    0,   0,    0,    1,    0,    0,    0,    0],    # OR-Bench
    [0,    0,   0,    0.5,  1,    0,    0,    0,    0],    # SORRY-Bench
    [1,    1,   0,    1,    0,    0.5,  0,    0,    0],    # AILuminate
    [1,    0,   0,    0.5,  0,    1,    0,    0,    0],    # PolyGuard
    [1,    0,   0,    0.5,  0,    1,    0,    0,    0],    # M-ALERT
    [0,    0,   0,    0,    0,    0,    1,    0,    0],    # SafeArena
    [0,    0,   1,    0,    0,    0,    0,    0,    0],    # HalluHard
    [0,    0,   0,    0,    0,    0,    1,    0,    0],    # ATBench
])

fig, ax = plt.subplots(figsize=(10, 9))

cmap = plt.cm.RdYlGn
im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

ax.set_xticks(np.arange(len(dimensions)))
ax.set_yticks(np.arange(len(benchmarks)))
ax.set_xticklabels(dimensions, fontsize=9, rotation=45, ha='right')
ax.set_yticklabels(benchmarks, fontsize=9)

# Add text annotations
for i in range(len(benchmarks)):
    for j in range(len(dimensions)):
        val = data[i, j]
        if val == 1:
            text = "●"
        elif val == 0.5:
            text = "◐"
        else:
            text = ""
        ax.text(j, i, text, ha="center", va="center", fontsize=10,
                color="white" if val > 0.6 else "black")

# Highlight the gap columns
for j in [7, 8]:  # SLM-tested, Compression-aware
    ax.add_patch(plt.Rectangle((j-0.5, -0.5), 1, len(benchmarks),
                               fill=False, edgecolor='red', linewidth=2, linestyle='--'))

ax.set_title('Safety Benchmark Coverage Matrix\n'
             '(● = covered, ◐ = partial, red border = critical SLM gaps)',
             fontsize=12, fontweight='bold', pad=15)

plt.colorbar(im, ax=ax, label='Coverage', shrink=0.6, pad=0.02)
plt.tight_layout()
plt.savefig("figures/fig4-benchmark-coverage.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/fig4-benchmark-coverage.png", dpi=300, bbox_inches='tight')
print("Saved fig4-benchmark-coverage.pdf/.png")
