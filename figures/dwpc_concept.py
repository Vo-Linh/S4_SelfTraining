"""DWPC — Dual-Witness Pseudo-label Correction. Headline concept figure.

Top:    pipeline — teacher pseudo-label cross-examined by two witnesses.
Middle: the 2x2 agreement decision table.
Bottom: the two mutual-rescue cases that justify the combination:
        (L) Witness A hallucinates a flip -> Witness B vetoes (structure implausible).
        (R) Witness B would delete a rare object -> Witness A protects (appearance rare).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'mathtext.fontset': 'dejavusans',
})

C_TEACH = '#D62728'   # teacher (biased)
C_APP   = '#FF7F0E'   # Witness A appearance / prototype
C_STR   = '#1F77B4'   # Witness B structure / topology
C_FLIP  = '#2CA02C'   # flip / recover
C_KEEP  = '#7F7F7F'   # keep
C_VETO  = '#9467BD'   # abstain / veto
C_RARE  = '#1F77B4'
C_HEAD  = '#7F7F7F'

fig = plt.figure(figsize=(16.5, 13.0), facecolor='white')
gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 1.15],
                      hspace=0.34, wspace=0.16,
                      left=0.04, right=0.975, top=0.92, bottom=0.055)
ax_pipe = fig.add_subplot(gs[0, :])
ax_tab  = fig.add_subplot(gs[1, :])
ax_L    = fig.add_subplot(gs[2, 0])
ax_R    = fig.add_subplot(gs[2, 1])

for ax in (ax_pipe, ax_tab):
    ax.axis('off')


# ==========================================================================
# ROW 1 — Pipeline
# ==========================================================================
ax_pipe.set_xlim(0, 100); ax_pipe.set_ylim(0, 30)
ax_pipe.set_title("(a)  Cross-examination pipeline — one biased teacher, two independent witnesses",
                  loc='left', fontweight='bold', fontsize=12, pad=4)


def box(ax, x, y, w, h, title, body, fc, ec, tcolor=None, tsize=10.5, bsize=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.25,rounding_size=0.8",
                 facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=2))
    ax.text(x + w/2, y + h - 1.6, title, ha='center', va='top',
            fontsize=tsize, fontweight='bold', color=tcolor or ec, zorder=3)
    if body:
        ax.text(x + w/2, y + h - 4.4, body, ha='center', va='top',
                fontsize=bsize, color='#222', zorder=3)


# Teacher
box(ax_pipe, 1, 11, 16, 9, "EMA Teacher",
    r"pseudo-label $\hat{y}_i$" + "\n(biased + noisy)", '#FFEFEF', C_TEACH)

# Witness A
box(ax_pipe, 26, 20, 22, 9, "Witness A — Appearance",
    r"prototype likelihood $p(f_i\!\mid\!c)$" + "\n+ rare-prior ratio  $\\Lambda_i$",
    '#FFF3E0', C_APP)
ax_pipe.text(37, 18.4, "domain-SENSITIVE", ha='center', fontsize=7.6,
             color=C_APP, style='italic')

# Witness B
box(ax_pipe, 26, 2, 22, 9, "Witness B — Structure",
    r"plausibility $s_i(c)$: area, conn.," + "\nco-occur, containment",
    '#E6F2FA', C_STR)
ax_pipe.text(37, 0.6, "domain-INVARIANT", ha='center', fontsize=7.6,
             color=C_STR, style='italic', fontweight='bold')

# Agreement decision
box(ax_pipe, 57, 11, 19, 9, "Agreement\ndecision", "", '#FFFCE5', '#D6B656',
    tcolor='#8A6D00', tsize=11)

# Corrected pseudo-label + student
box(ax_pipe, 84, 11, 15, 9, "Student", r"$\mathcal{L}_{\rm ssl}$ on $\tilde{y}_i$",
    'white', '#444', tsize=11)

# Arrows
ax_pipe.add_patch(FancyArrowPatch((17, 17), (26, 24), arrowstyle='-|>',
                  mutation_scale=15, color=C_APP, lw=2.0,
                  connectionstyle='arc3,rad=0.15', zorder=1))
ax_pipe.add_patch(FancyArrowPatch((17, 14), (26, 6.5), arrowstyle='-|>',
                  mutation_scale=15, color=C_STR, lw=2.0,
                  connectionstyle='arc3,rad=-0.15', zorder=1))
ax_pipe.add_patch(FancyArrowPatch((48, 24), (57, 17), arrowstyle='-|>',
                  mutation_scale=15, color=C_APP, lw=2.0,
                  connectionstyle='arc3,rad=-0.15', zorder=1))
ax_pipe.add_patch(FancyArrowPatch((48, 6.5), (57, 14), arrowstyle='-|>',
                  mutation_scale=15, color=C_STR, lw=2.0,
                  connectionstyle='arc3,rad=0.15', zorder=1))
ax_pipe.add_patch(FancyArrowPatch((76, 15.5), (84, 15.5), arrowstyle='-|>',
                  mutation_scale=16, color='#444', lw=2.2, zorder=1))
ax_pipe.text(80, 17.0, r"$\tilde{y}_i, w_i$", ha='center', fontsize=8.5, color='#444')


# ==========================================================================
# ROW 2 — 2x2 decision table
# ==========================================================================
ax_tab.set_xlim(0, 100); ax_tab.set_ylim(0, 30)
ax_tab.set_title("(b)  The decision is the AGREEMENT pattern of the two witnesses",
                 loc='left', fontweight='bold', fontsize=12, pad=4)

# Table geometry
x0, y0, cw, ch = 24, 2, 33, 11
# Column headers (Witness B)
ax_tab.text(x0 + cw*0.5, 27.2, "Witness B: structure PLAUSIBLE",
            ha='center', fontsize=9.4, color=C_STR, fontweight='bold')
ax_tab.text(x0 + cw*1.5, 27.2, "Witness B: structure IMPLAUSIBLE",
            ha='center', fontsize=9.4, color=C_STR, fontweight='bold')
# Row headers (Witness A)
ax_tab.text(x0 - 1.5, y0 + ch*1.5, "Witness A:\nsays RARE $c^*$",
            ha='right', va='center', fontsize=9.4, color=C_APP, fontweight='bold')
ax_tab.text(x0 - 1.5, y0 + ch*0.5, "Witness A:\nno rare evidence",
            ha='right', va='center', fontsize=9.4, color=C_APP, fontweight='bold')

cells = [
    # (col, row, label, sublabel, color, facecolor)
    (0, 1, "FLIP → rare", "both agree: recover\nrare-class pixel (budgeted)", C_FLIP, '#E8F5E9'),
    (1, 1, "ABSTAIN", "appearance may be\nhallucination → down-weight", C_VETO, '#F0E8F7'),
    (0, 0, "KEEP", "reliable → full-weight\npseudo-label", C_KEEP, '#F2F2F2'),
    (1, 0, "DOWN-WEIGHT", "structurally noisy →\nreduce loss weight", '#C2882B', '#FBF1E0'),
]
for (cx, ry, lab, sub, col, fcol) in cells:
    rx = x0 + cx*cw; ryy = y0 + ry*ch
    ax_tab.add_patch(FancyBboxPatch((rx, ryy), cw, ch,
                     boxstyle="round,pad=0.15,rounding_size=0.4",
                     facecolor=fcol, edgecolor=col, linewidth=1.8, zorder=2))
    ax_tab.text(rx + cw/2, ryy + ch - 1.8, lab, ha='center', va='top',
                fontsize=11, fontweight='bold', color=col, zorder=3)
    ax_tab.text(rx + cw/2, ryy + ch - 4.7, sub, ha='center', va='top',
                fontsize=8.4, color='#333', zorder=3)

# Highlight the flip cell (the headline action)
ax_tab.add_patch(FancyBboxPatch((x0, y0+ch), cw, ch,
                 boxstyle="round,pad=0.15,rounding_size=0.4",
                 facecolor='none', edgecolor=C_FLIP, linewidth=3.0,
                 linestyle='-', zorder=4))

ax_tab.text(92, 15,
            "Flip only on\nDUAL agreement.\n\nProtect a pixel if\nEITHER witness\nvouches for it.",
            ha='center', va='center', fontsize=9.0, color='#222',
            bbox=dict(boxstyle='round,pad=0.4', fc='#FFFCE5',
                      ec='#D6B656', lw=1.0))


# ==========================================================================
# ROW 3 LEFT — Mutual rescue case 1: A hallucinates, B vetoes
# ==========================================================================
def mini_scene(ax, which):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#888'); s.set_linewidth(0.9)
    # background vegetation
    ax.add_patch(Rectangle((0, 0), 10, 10, facecolor='#5FA052', alpha=0.65,
                           edgecolor='none'))


ax_L.set_title("(c)  Witness A hallucinates → Witness B VETOES",
               loc='left', fontweight='bold', fontsize=11, pad=6)
mini_scene(ax_L, 'L')
# a shadow pixel that "looks like" water (appearance), isolated (structure-bad)
ax_L.scatter([5.0], [6.0], s=240, marker='s', c='#3F77BB',
             edgecolors=C_TEACH, linewidths=2.2, zorder=5)
ax_L.annotate("shadow pixel:\nappearance ≈ water,\nbut isolated (1 px)",
              xy=(5.0, 6.0), xytext=(0.4, 8.6), fontsize=8.2, color='#222',
              bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#888', lw=0.7),
              arrowprops=dict(arrowstyle='->', color='#555', lw=0.9))
# Witness A vote
ax_L.text(1.0, 3.2, "Witness A: FLIP→water ✗",
          fontsize=9.0, color=C_APP, fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.22', fc='#FFF3E0', ec=C_APP, lw=0.9))
# Witness B veto
ax_L.text(1.0, 1.4, "Witness B: implausible → VETO",
          fontsize=9.0, color=C_STR, fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.22', fc='#E6F2FA', ec=C_STR, lw=0.9))
# result
ax_L.text(7.7, 1.4, "no flip\n(ProDA would\nflip wrongly)", ha='center',
          fontsize=8.0, color=C_VETO, fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.22', fc='#F0E8F7', ec=C_VETO, lw=0.9))


# ==========================================================================
# ROW 3 RIGHT — Mutual rescue case 2: B deletes rare, A protects
# ==========================================================================
ax_R.set_title("(d)  Witness B would delete rare object → Witness A PROTECTS",
               loc='left', fontweight='bold', fontsize=11, pad=6)
mini_scene(ax_R, 'R')
# a small TRUE pond: small (structure-bad under min-area) but water-appearance (A-good)
ax_R.scatter([5.2], [5.6], s=150, marker='o', c='#3F77BB',
             edgecolors='black', linewidths=1.2, zorder=5)
ax_R.annotate("small TRUE pond:\nfails min-area,\nbut appearance = water",
              xy=(5.2, 5.6), xytext=(0.4, 8.6), fontsize=8.2, color='#222',
              bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#888', lw=0.7),
              arrowprops=dict(arrowstyle='->', color='#555', lw=0.9))
# Witness B would delete
ax_R.text(1.0, 3.2, "Witness B: min-area → DELETE ✗",
          fontsize=8.6, color=C_STR, fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.22', fc='#E6F2FA', ec=C_STR, lw=0.9))
# Witness A protects (gate)
ax_R.text(1.0, 1.4, "Witness A: rare evidence → GATE OFF",
          fontsize=8.6, color=C_APP, fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.22', fc='#FFF3E0', ec=C_APP, lw=0.9))
ax_R.text(7.8, 1.4, "pond kept\n(clDice/Kervadec\nwould delete)", ha='center',
          fontsize=8.0, color=C_FLIP, fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.22', fc='#E8F5E9', ec=C_FLIP, lw=0.9))


# ==========================================================================
# Title + legend
# ==========================================================================
fig.suptitle(
    "Dual-Witness Pseudo-label Correction (DWPC): an appearance witness and a domain-invariant structure witness\n"
    "cross-examine the teacher — each repairs the other's failure mode (ProDA hallucinates; topology losses delete rare objects)",
    fontsize=12.6, fontweight='bold', y=0.985)

handles = [
    mpatches.Patch(facecolor='#FFEFEF', edgecolor=C_TEACH, lw=1.4, label='EMA teacher (biased + noisy)'),
    mpatches.Patch(facecolor='#FFF3E0', edgecolor=C_APP, lw=1.4, label='Witness A — appearance / prototype likelihood (P1, ProDA-lineage)'),
    mpatches.Patch(facecolor='#E6F2FA', edgecolor=C_STR, lw=1.4, label='Witness B — domain-invariant structure (TARS)'),
    mpatches.Patch(facecolor='#E8F5E9', edgecolor=C_FLIP, lw=1.4, label='flip / protect (rare-class recovered)'),
    mpatches.Patch(facecolor='#F0E8F7', edgecolor=C_VETO, lw=1.4, label='abstain / veto'),
]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8.6,
           frameon=True, framealpha=0.96, edgecolor='#888',
           bbox_to_anchor=(0.5, -0.002))

out = '/home/ubuntu/S4_SelfTraining/figures/dwpc_concept.png'
plt.savefig(out, dpi=170, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"saved {out}")
