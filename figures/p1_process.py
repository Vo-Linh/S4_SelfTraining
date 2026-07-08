"""P1 pseudo-label inversion — step-by-step process diagram.

Shows what happens to a single pixel as it flows through the pipeline:
  1. EMA teacher prediction (biased)
  2. Prototype album lookup
  3. Per-class likelihood computation
  4. Likelihood-ratio test with rare-class prior
  5. Safety gate (rare-target? margin? flip-budget?)
  6. Output pseudo-label to student

Designed to read left-to-right, top-to-bottom, with two parallel paths
(teacher branch vs. prototype branch) converging at the decision gate.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np

# ---------------- Colors ----------------
C_TEACHER = '#D62728'   # red — biased teacher
C_PROTO   = '#1F77B4'   # blue — prototype album (rare)
C_HEAD    = '#888888'   # grey — head class
C_RARE    = '#1F77B4'   # blue — rare class
C_FLIP    = '#2CA02C'   # green — flip outcome
C_KEEP    = '#999999'   # grey — keep outcome
C_TEXT    = '#222222'
C_BG_BOX  = '#F8F8F8'

# ---------------- Figure ----------------
fig, ax = plt.subplots(figsize=(17, 10), facecolor='white')
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.set_aspect('auto')
ax.axis('off')

# ---------------- Helpers ----------------
def step_box(ax, x, y, w, h, title, body, *, fc=C_BG_BOX, ec='#666',
             title_color=C_TEXT, body_color=C_TEXT, title_size=11,
             body_size=9.0, lw=1.2):
    """Rounded rectangle with a bold title and body text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.4,rounding_size=0.8",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 1.4, title,
            ha='center', va='top', fontsize=title_size, fontweight='bold',
            color=title_color, zorder=3)
    ax.text(x + w/2, y + h - 4.0, body,
            ha='center', va='top', fontsize=body_size, color=body_color,
            zorder=3)

def arrow(ax, p1, p2, *, color='#444', lw=2.0, ls='-', label=None,
          label_off=(0, 0.6), label_size=9, label_color=None,
          curve=0.0):
    a = FancyArrowPatch(
        p1, p2,
        arrowstyle='-|>', mutation_scale=18,
        color=color, lw=lw, linestyle=ls,
        connectionstyle=f'arc3,rad={curve}',
        zorder=4,
    )
    ax.add_patch(a)
    if label is not None:
        mx = (p1[0] + p2[0]) / 2 + label_off[0]
        my = (p1[1] + p2[1]) / 2 + label_off[1]
        ax.text(mx, my, label, ha='center', va='center',
                fontsize=label_size, color=label_color or color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.18', fc='white',
                          ec=color, lw=0.6, alpha=0.95),
                zorder=5)

def step_number(ax, x, y, n, *, color='#444'):
    c = Circle((x, y), 1.4, facecolor=color, edgecolor='black',
               linewidth=0.7, zorder=6)
    ax.add_patch(c)
    ax.text(x, y, str(n), ha='center', va='center',
            fontsize=10.5, fontweight='bold', color='white', zorder=7)

# ---------------- Title strip ----------------
ax.text(50, 58.5,
        "P1 — Pseudo-label inversion pipeline (per pixel)",
        ha='center', va='center', fontsize=14, fontweight='bold',
        color=C_TEXT)
ax.text(50, 56.2,
        "Two opinions are asked in parallel; if they disagree strongly AND the alternative is a rare class, FLIP.",
        ha='center', va='center', fontsize=10, color='#444', style='italic')

# ==========================================================================
# STEP 1 — Input pixel feature
# ==========================================================================
step_box(ax, 2, 38, 14, 12,
         "Input",
         "One pixel from an\nunlabeled image\n\nfeature  $f_i$",
         fc='white', ec='#444', title_size=12)
step_number(ax, 3.5, 49.0, 1)

# ==========================================================================
# STEP 2a — EMA teacher (top branch)
# ==========================================================================
step_box(ax, 22, 44, 18, 12,
         "EMA Teacher (biased)",
         "predicts class + confidence\n"
         r"$\hat{y}_i = \mathrm{Building}$" + "\n"
         r"$\hat{c}_i = 0.97$",
         fc='#FFEFEF', ec=C_TEACHER, title_color=C_TEACHER,
         title_size=11)
step_number(ax, 23.5, 55.0, 2)
ax.text(31, 41.8, "(but this is WRONG —\nthe pixel is truly Water)",
        ha='center', va='top', fontsize=8.5,
        color=C_TEACHER, style='italic')

# ==========================================================================
# STEP 2b — Prototype album (bottom branch)
# ==========================================================================
step_box(ax, 22, 22, 18, 13,
         "Prototype Album",
         "DynamicAnchor bank\n"
         r"$\{\mu_1,\mu_2,\mu_3\}\!:\,\kappa=\mathrm{Building}$" + "\n"
         r"$\{\mu_4\}\!:\,\kappa=\mathrm{Water}$" + "\n"
         "quality gates  $q_k$",
         fc='#E6F2FA', ec=C_PROTO, title_color=C_PROTO,
         title_size=11)
step_number(ax, 23.5, 34.0, 2)
ax.text(31, 19.8, "(independent geometric view\nof what each class looks like)",
        ha='center', va='top', fontsize=8.5,
        color=C_PROTO, style='italic')

# Arrow from input → both branches
arrow(ax, (16, 46.5), (22, 50.0), color=C_TEACHER, lw=2.2,
      label="(a) ask teacher", label_off=(0, 1.4),
      label_size=8.5, curve=0.10)
arrow(ax, (16, 43.5), (22, 30.0), color=C_PROTO, lw=2.2,
      label="(b) ask album", label_off=(0, -1.4),
      label_size=8.5, curve=-0.15)

# ==========================================================================
# STEP 3 — Likelihood per class
# ==========================================================================
step_box(ax, 46, 22, 20, 13,
         "Per-class likelihood",
         r"$p(f_i\mid c) = \sum_k \pi_k\,\mathcal{N}(f_i;\mu_k,\Sigma_c)$" + "\n\n"
         r"$p(f_i\mid\mathrm{Building}) = \mathbf{LOW}$" + "\n"
         r"$p(f_i\mid\mathrm{Water})    = \mathbf{HIGH}$",
         fc='#E6F2FA', ec=C_PROTO, title_color=C_PROTO,
         title_size=11, body_size=9.0)
step_number(ax, 47.5, 34.0, 3)

arrow(ax, (40, 28.5), (46, 28.5), color=C_PROTO, lw=2.2)

# ==========================================================================
# STEP 4 — Likelihood ratio test
# ==========================================================================
step_box(ax, 46, 42, 20, 13,
         "Likelihood ratio  $\\Lambda$",
         r"$\Lambda = \log\dfrac{p(f\mid\mathrm{Water})}{p(f\mid\mathrm{Building})}$"
         + r" $+\;\log\dfrac{\pi^{\mathrm{rare}}_{\mathrm{Water}}}{\pi^{\mathrm{rare}}_{\mathrm{Building}}}$"
         + "\n\n"
         "rare-class prior gives\nthe minority a boost",
         fc='#E6F2FA', ec=C_PROTO, title_color=C_PROTO,
         title_size=11, body_size=8.7)
step_number(ax, 47.5, 54.0, 4)

# arrows: teacher → ratio, likelihood → ratio
arrow(ax, (40, 50.0), (46, 48.5), color=C_TEACHER, lw=2.2,
      label=r"$\hat{y}_i, \hat{c}_i$", label_off=(0.5, 1.2),
      label_size=9, curve=0.0)
arrow(ax, (56, 35), (56, 42), color=C_PROTO, lw=2.2,
      label="combine", label_off=(2.5, 0), label_size=9, curve=0.0)

# ==========================================================================
# STEP 5 — Decision gate (3 conditions)
# ==========================================================================
step_box(ax, 72, 36, 24, 18,
         "Decision gate",
         r"$\Lambda > \tau_{\mathrm{flip}}$?    (strong disagreement)" + "\n"
         r"argmax $\in \mathcal{C}_{\mathrm{rare}}$?    (target is rare)" + "\n"
         r"flip budget left?    ($\leq \beta_{\mathrm{flip}} \cdot |U|$)" + "\n\n"
         "ALL three must be YES",
         fc='#FFF8E7', ec='#D6B656', title_color='#8A6D00',
         title_size=11, body_size=9.0)
step_number(ax, 73.5, 53.0, 5)

arrow(ax, (66, 48.5), (72, 47.0), color='#444', lw=2.4,
      label=r"$\Lambda$, $\hat{y}_i$",
      label_off=(0, 1.4), label_size=9)

# ==========================================================================
# STEP 6 — Outcomes (two branches)
# ==========================================================================
# YES branch — FLIP
step_box(ax, 72, 18, 24, 12,
         "OUTCOME — YES: FLIP",
         r"$\tilde{y}_i \leftarrow \mathrm{Water}$" + "\n"
         "rare-class pixel\nRECOVERED",
         fc='#E8F5E9', ec=C_FLIP, title_color=C_FLIP,
         title_size=11)
# NO branch — KEEP
step_box(ax, 72, 2, 24, 12,
         "OUTCOME — NO: KEEP",
         r"$\tilde{y}_i \leftarrow \hat{y}_i$ (unchanged)" + "\n"
         "fall back to standard\nreliability weighting",
         fc='#F0F0F0', ec=C_KEEP, title_color='#555',
         title_size=11)

arrow(ax, (84, 36), (84, 30), color=C_FLIP, lw=2.6,
      label="ALL YES  →", label_off=(6, 0), label_size=9,
      label_color=C_FLIP)
arrow(ax, (84, 36), (84, 14), color=C_KEEP, lw=2.2, ls='--',
      label="any NO  →", label_off=(6, -7.5), label_size=9,
      label_color=C_KEEP, curve=0.25)

# ==========================================================================
# STEP 7 — Student training
# ==========================================================================
# Single output box at the far left bottom, fed by both outcomes
step_box(ax, 2, 8, 14, 14,
         "Student training",
         r"$\mathcal{L}_{\mathrm{ssl}}$ on" + "\n"
         r"corrected $\tilde{y}_i$" + "\n\n"
         "(flipped pixels train\non the RIGHT label)",
         fc='white', ec='#444', title_size=12)
step_number(ax, 3.5, 21.0, 6)

# Curved arrow from FLIP outcome → student (down-left)
arrow(ax, (72, 24), (16, 17), color=C_FLIP, lw=2.6,
      curve=0.25,
      label="corrected label", label_off=(-1, 3),
      label_size=9, label_color=C_FLIP)
# Curved arrow from KEEP outcome → student (also feeds in)
arrow(ax, (72, 8), (16, 12), color=C_KEEP, lw=1.8, ls='--',
      curve=-0.18)

# ==========================================================================
# Side panel — geometric intuition (feature space mini-view)
# ==========================================================================
# Small inset showing the geometry
inset_x, inset_y, inset_w, inset_h = 42, 2, 24, 14
ax.add_patch(FancyBboxPatch(
    (inset_x, inset_y), inset_w, inset_h,
    boxstyle="round,pad=0.4,rounding_size=0.6",
    facecolor='#FAFAFA', edgecolor='#888', linewidth=0.9, zorder=1))
ax.text(inset_x + inset_w/2, inset_y + inset_h - 1.0,
        "Geometric intuition",
        ha='center', va='top', fontsize=10, fontweight='bold', color='#444')

# inside this inset, sketch a tiny feature space
# Coordinates within the inset
def ix(u): return inset_x + 2 + u * (inset_w - 4)
def iy(v): return inset_y + 2 + v * (inset_h - 4)

# Head cloud (top-left of inset)
ax.add_patch(Ellipse((ix(0.30), iy(0.65)), inset_w*0.30, inset_h*0.35,
                     facecolor=C_HEAD, alpha=0.18, edgecolor='none', zorder=2))
# Rare cloud (bottom-right of inset)
ax.add_patch(Ellipse((ix(0.78), iy(0.30)), inset_w*0.18, inset_h*0.22,
                     facecolor=C_RARE, alpha=0.28, edgecolor='none', zorder=2))
# Stars (prototypes)
ax.scatter([ix(0.22), ix(0.30), ix(0.40)],
           [iy(0.72), iy(0.62), iy(0.70)],
           s=90, marker='*', c=C_HEAD,
           edgecolors='black', linewidths=0.5, zorder=4)
ax.scatter([ix(0.78)], [iy(0.30)],
           s=110, marker='*', c=C_RARE,
           edgecolors='black', linewidths=0.5, zorder=4)
# Victim pixel
vx, vy = ix(0.60), iy(0.40)
ax.scatter([vx], [vy], s=55, c=C_RARE,
           edgecolors=C_TEACHER, linewidths=1.5, zorder=5)
ax.text(vx + 0.4, vy + 0.9, "victim", fontsize=7.5,
        color=C_TEACHER, fontweight='bold')

# distance lines
ax.plot([vx, ix(0.40)], [vy, iy(0.70)], color=C_HEAD,
        linestyle=':', linewidth=1.2, alpha=0.7, zorder=3)
ax.plot([vx, ix(0.78)], [vy, iy(0.30)], color=C_RARE,
        linestyle='-', linewidth=2.0, alpha=0.85, zorder=3)
# Tiny labels
ax.text(ix(0.50), iy(0.58), "far", fontsize=7,
        color=C_HEAD, style='italic')
ax.text(ix(0.68), iy(0.30), "close", fontsize=7,
        color=C_RARE, style='italic', fontweight='bold')
# Annotation
ax.text(inset_x + inset_w/2, inset_y + 0.9,
        "victim pixel is geometrically close to the rare prototype",
        ha='center', va='bottom', fontsize=7.8, color='#444')

# ==========================================================================
# Legend strip at very bottom
# ==========================================================================
legend_handles = [
    mpatches.Patch(facecolor='#FFEFEF', edgecolor=C_TEACHER, lw=1.4,
                   label='Teacher branch (biased classifier head)'),
    mpatches.Patch(facecolor='#E6F2FA', edgecolor=C_PROTO, lw=1.4,
                   label='Prototype-album branch (geometric)'),
    mpatches.Patch(facecolor='#FFF8E7', edgecolor='#D6B656', lw=1.4,
                   label='Decision gate (3 safety conditions)'),
    mpatches.Patch(facecolor='#E8F5E9', edgecolor=C_FLIP, lw=1.4,
                   label='FLIP outcome (label changed to rare class)'),
    mpatches.Patch(facecolor='#F0F0F0', edgecolor=C_KEEP, lw=1.4,
                   label='KEEP outcome (label unchanged, fall back to weighting)'),
    Line2D([0], [0], color=C_TEACHER, lw=2.2,
           label="teacher signal flow"),
    Line2D([0], [0], color=C_PROTO, lw=2.2,
           label="prototype signal flow"),
    Line2D([0], [0], color=C_FLIP, lw=2.6,
           label="flip-path (label corrected)"),
    Line2D([0], [0], color=C_KEEP, lw=2.0, ls='--',
           label="keep-path (label unchanged)"),
]

leg = ax.legend(
    handles=legend_handles, loc='lower left',
    bbox_to_anchor=(0.0, -0.06), ncol=3,
    fontsize=8.6, frameon=True, fancybox=True, framealpha=1.0,
    edgecolor='#888',
    title='Legend — colors, branches, signal types',
    title_fontsize=9.5,
    handletextpad=0.7, columnspacing=1.2, borderpad=0.7,
)

plt.tight_layout()
out = '/home/ubuntu/S4_SelfTraining/figures/p1_process.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"saved {out}")
