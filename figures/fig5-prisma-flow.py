"""Generate Figure 5: PRISMA-inspired systematic review flow diagram."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

box_style = dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5)
exc_style = dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1.5)
fin_style = dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1.5)

# Identification
ax.text(5, 9.2, 'Identification', fontsize=13, fontweight='bold', ha='center', color='#1565C0')
ax.text(5, 8.5, 'Records identified through\ndatabase searching\n(n = 450)', fontsize=10,
        ha='center', va='center', bbox=box_style)

# Arrow
ax.annotate('', xy=(5, 7.6), xytext=(5, 7.9), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

# Screening
ax.text(5, 7.3, 'Screening', fontsize=13, fontweight='bold', ha='center', color='#1565C0')
ax.text(5, 6.6, 'Records after title/abstract\nscreening\n(n = 280)', fontsize=10,
        ha='center', va='center', bbox=box_style)

# Exclusion box 1
ax.text(8.5, 7.5, 'Excluded (n = 170)\n- Not SLM-relevant\n- Duplicate entries\n- No English abstract',
        fontsize=8, ha='center', va='center', bbox=exc_style)
ax.annotate('', xy=(7.2, 7.5), xytext=(6.8, 8.2), arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.2))

# Arrow
ax.annotate('', xy=(5, 5.7), xytext=(5, 6.0), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

# Eligibility
ax.text(5, 5.4, 'Eligibility', fontsize=13, fontweight='bold', ha='center', color='#1565C0')
ax.text(5, 4.7, 'Full-text articles assessed\nfor eligibility\n(n = 280)', fontsize=10,
        ha='center', va='center', bbox=box_style)

# Exclusion box 2
ax.text(8.5, 5.3, 'Excluded (n = 65)\n- Models >13B only\n- No method contribution\n- Below citation threshold',
        fontsize=8, ha='center', va='center', bbox=exc_style)
ax.annotate('', xy=(7.2, 5.3), xytext=(6.8, 4.9), arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.2))

# Arrow
ax.annotate('', xy=(5, 3.7), xytext=(5, 4.0), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

# Included
ax.text(5, 3.4, 'Included', fontsize=13, fontweight='bold', ha='center', color='#2E7D32')
ax.text(5, 2.5, 'Studies included in\nsystematic survey\n(n = 215)', fontsize=11,
        ha='center', va='center', bbox=fin_style)

# Breakdown
ax.text(5, 1.2, 'Alignment (68)  |  Benchmarks (42)  |  Efficiency (31)\n'
                 'Multilingual (18)  |  Interpretability (24)  |  Models (32)',
        fontsize=9, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', edgecolor='#999', linewidth=1))

plt.tight_layout()
plt.savefig("figures/fig5-prisma-flow.pdf", dpi=300, bbox_inches='tight')
plt.savefig("figures/fig5-prisma-flow.png", dpi=300, bbox_inches='tight')
print("Saved fig5-prisma-flow.pdf/.png")
