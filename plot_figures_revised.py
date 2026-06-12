"""
论文配图生成脚本（出版质量）。

  Figure 2: 三条件对照实验 F1 对比柱状图（IBM + Indian 数据集）
  Figure 3: 阈值敏感性曲线（F1、Precision、Recall、G-Mean）

用法：
  python plot_figures_revised.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Publication style (compact, paper-appropriate) ─────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 7.5
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['axes.unicode_minus'] = False

PALETTE = {
    "ensemble": "#0F4D92",    # deep blue — hero method
    "smote":    "#E53935",    # red — baseline
    "grey":     "#767676",
    "light_grey": "#E8E8E8",
}

OUTPUT_DIR = Path(__file__).parent / 'results' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Three-condition controlled comparison
# ══════════════════════════════════════════════════════════════════════════════

# Data from Table 1 and Table 2 of the paper (mean F1, no error bars shown here)
ibm_ensemble = [0.4410, 0.5296, 0.5296]
ibm_smote    = [0.3960, 0.5072, 0.3960]

indian_ensemble = [0.6828, 0.6838, 0.6838]
indian_smote    = [0.6705, 0.6777, 0.6705]

# Significance levels for the Δ annotations
# IBM:  cond1 p=0.015(**),  cond2 p=0.026(*),   cond3 p<0.001(***)
# Indian: cond1 p<0.001(***) cond2 p=0.010(**), cond3 p<0.001(***)
ibm_sig   = ['**', '*', '***']
indian_sig = ['***', '**', '***']

conditions = ['Condition 1:\nEqual τ=0.50',
              'Condition 2:\nFair τ* each',
              'Condition 3:\nAsymmetric τ*']

x = np.arange(3)
w = 0.30   # bar width
gap = 0.04 # gap between paired bars

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

def plot_panel(ax, ens, smt, sig_list, title, ylims, delta_y_offsets):
    """Draw one dataset panel."""
    bars_e = ax.bar(x - gap - w/2, ens, w, color=PALETTE['ensemble'],
                    edgecolor='none', zorder=2)
    bars_s = ax.bar(x + gap + w/2, smt, w, color=PALETTE['smote'],
                    edgecolor='none', zorder=2)

    ax.set_title(title, fontsize=8.5, fontweight='bold', pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=7)
    ax.set_ylim(ylims)
    ax.set_ylabel('F1 Score', fontsize=7.5)

    # Δ annotation between each pair
    for i in range(3):
        delta = ens[i] - smt[i]
        mid_y = max(ens[i], smt[i]) + delta_y_offsets[i]
        sig = sig_list[i]
        color = PALETTE['grey'] if sig in ('*', '**', 'n.s.') else PALETTE['smote']
        ax.annotate(f'Δ={delta:.3f}\n({sig})',
                    xy=(x[i], mid_y),
                    fontsize=6.5, ha='center', va='bottom',
                    color=color,
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='#FAFAFA', edgecolor='#CCCCCC',
                              alpha=0.85))

    # Value labels on bars
    for bar, val in zip(bars_e, ens):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=6.5, fontweight='bold', color=PALETTE['ensemble'])
    for bar, val in zip(bars_s, smt):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=6.5, fontweight='bold', color=PALETTE['smote'])

    ax.legend([bars_e, bars_s], ['Heterogeneous Ensemble', 'SMOTE+LightGBM'],
              loc='lower right', fontsize=6.5)
    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5, zorder=0)

plot_panel(axes[0], ibm_ensemble, ibm_smote, ibm_sig,
           'IBM HR Attrition',
           ylims=(0.30, 0.62),
           delta_y_offsets=(0.048, 0.028, 0.018))

plot_panel(axes[1], indian_ensemble, indian_smote, indian_sig,
           'Indian HR Attrition',
           ylims=(0.58, 0.76),
           delta_y_offsets=(0.012, 0.010, 0.008))

fig.suptitle(
    'Three-Condition Controlled Comparison: Heterogeneous Ensemble vs SMOTE+LightGBM (F1)',
    fontsize=9, fontweight='bold', y=1.01)

plt.tight_layout(pad=1.5, w_pad=2.0)
for fmt in ('svg', 'pdf', 'png'):
    fig.savefig(OUTPUT_DIR / f'three_condition_comparison.{fmt}',
                dpi=600 if fmt == 'png' else None,
                bbox_inches='tight')
plt.close(fig)
print(f"[OK] Figure 2 saved to {OUTPUT_DIR}/three_condition_comparison.*")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Threshold sensitivity curves
# ══════════════════════════════════════════════════════════════════════════════

thresholds = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
f1         = np.array([0.4654, 0.5267, 0.5442, 0.5106, 0.4410, 0.3797, 0.3171])
precision  = np.array([0.3318, 0.4393, 0.5203, 0.6008, 0.6603, 0.7541, 0.8217])
recall     = np.array([0.7973, 0.6699, 0.5792, 0.4523, 0.3402, 0.2584, 0.2037])
gmean      = np.array([0.5835, 0.6700, 0.6966, 0.6638, 0.5684, 0.4733, 0.3857])

fig, ax = plt.subplots(figsize=(6.5, 4.0))

tau_star = 0.189

line_kw = dict(lw=1.6, marker='o', markersize=5, markeredgecolor='white',
               markeredgewidth=0.8)

ax.plot(thresholds, f1,        color=PALETTE['ensemble'], label='F1',         **line_kw)
ax.plot(thresholds, precision, color='#2E9E44',            label='Precision',  **line_kw)
ax.plot(thresholds, recall,    color='#E53935',            label='Recall',     **line_kw)
ax.plot(thresholds, gmean,     color='#9A4D8E',            label='G-Mean', linestyle='--',
        lw=1.4, marker='D', markersize=4.5,
        markeredgecolor='white', markeredgewidth=0.8)

# τ* vertical reference line
ax.axvline(tau_star, color=PALETTE['grey'], linewidth=1.0,
           linestyle=':', alpha=0.6)

# Highlight the optimal point (F1 at τ≈0.20)
idx_star = np.argmin(np.abs(thresholds - tau_star))
ax.plot(thresholds[idx_star], f1[idx_star], 'o',
        markersize=11, markerfacecolor='white',
        markeredgecolor=PALETTE['ensemble'], markeredgewidth=2.0,
        zorder=5)

ax.annotate(f'τ*≈{tau_star:.2f}',
            xy=(tau_star, f1[idx_star]),
            xytext=(tau_star + 0.06, f1[idx_star] - 0.04),
            fontsize=7, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=PALETTE['ensemble'],
                            lw=1.0),
            color=PALETTE['ensemble'])

ax.set_xlabel('Decision Threshold τ', fontsize=7.5)
ax.set_ylabel('Score', fontsize=7.5)
ax.set_xlim(0.08, 0.72)
ax.set_ylim(0.28, 0.85)
ax.legend(loc='center right', fontsize=6.8, columnspacing=0.8)
ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)

fig.suptitle('Threshold Sensitivity Analysis (IBM Dataset)',
             fontsize=8.5, fontweight='bold', y=1.01)

plt.tight_layout(pad=1.2)
for fmt in ('svg', 'pdf', 'png'):
    fig.savefig(OUTPUT_DIR / f'threshold_sensitivity.{fmt}',
                dpi=600 if fmt == 'png' else None,
                bbox_inches='tight')
plt.close(fig)
print(f"[OK] Figure 3 saved to {OUTPUT_DIR}/threshold_sensitivity.*")
