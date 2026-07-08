"""Feature-space visualization of the P1 pseudo-label inversion concept.

3 panels:
  (a) The setup: head-class cloud (many pixels) + rare-class cloud (few pixels),
      4 prototypes, EMA teacher's decision boundary biased toward the rare class.
  (b) What reweighting does: keeps the wrong label at full or low weight, never recovers.
  (c) What P1 does: prototype-density likelihood ratio flips the label, recovering the pixel.

Includes a unified legend mapping every glyph and color to its semantic meaning.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Ellipse, Circle
import numpy as np

# ---------- Reproducibility ----------
rng = np.random.default_rng(7)

# ---------- Class data ----------
# Head class: many pixels in upper-left
head_center = np.array([3.2, 6.8])
head_cov    = np.array([[1.6, 0.4], [0.4, 1.2]])
head_pts    = rng.multivariate_normal(head_center, head_cov, size=70)

# Rare class: few pixels in lower-right
rare_center = np.array([8.0, 3.0])
rare_cov    = np.array([[0.45, 0.05], [0.05, 0.35]])
rare_pts    = rng.multivariate_normal(rare_center, rare_cov, size=12)

# Prototypes
head_protos = np.array([[2.4, 7.3], [3.5, 6.4], [4.1, 7.5]])  # 3 prototypes for head
rare_proto  = np.array([[8.0, 3.0]])                          # 1 prototype for rare

# Victim pixel: truly rare-class, sits in the rare cloud but on the WRONG
# side of the teacher's biased decision boundary
victim = np.array([6.6, 4.0])

# ---------- Colors ----------
C_HEAD   = '#888888'
C_RARE   = '#1F77B4'
C_BNDY   = '#D62728'  # teacher boundary
C_FLIP   = '#2CA02C'  # flip arrow
C_VICTIM = '#D62728'
C_PANEL_BG = '#FAFAFA'

# ---------- Figure ----------
fig = plt.figure(figsize=(18, 7.6), facecolor='white')
# Reserve bottom strip for the legend; top for 3 panels
gs = fig.add_gridspec(2, 3, height_ratios=[10, 1.2], hspace=0.10, wspace=0.10)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
ax_legend = fig.add_subplot(gs[1, :])
ax_legend.axis('off')

for ax in axes:
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_facecolor(C_PANEL_BG)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('feature dim 1', fontsize=9, labelpad=2)
    ax.set_ylabel('feature dim 2', fontsize=9, labelpad=2)
    for s in ax.spines.values():
        s.set_color('#999'); s.set_linewidth(0.8)


# ---------- Helper: draw the static feature-space layout shared by all 3 panels ----------
def draw_base_layout(ax, show_boundary=True):
    # Head-class soft cloud (ellipse)
    head_ellipse = Ellipse(head_center, width=4.6, height=3.4, angle=18,
                            facecolor=C_HEAD, edgecolor=C_HEAD,
                            alpha=0.15, linewidth=0)
    ax.add_patch(head_ellipse)
    # Rare-class soft cloud
    rare_ellipse = Ellipse(rare_center, width=2.2, height=1.8, angle=10,
                            facecolor=C_RARE, edgecolor=C_RARE,
                            alpha=0.18, linewidth=0)
    ax.add_patch(rare_ellipse)

    # Head pixels
    ax.scatter(head_pts[:, 0], head_pts[:, 1], s=14, c=C_HEAD,
               alpha=0.55, edgecolors='none', label='Building pixels (head)')
    # Rare pixels
    ax.scatter(rare_pts[:, 0], rare_pts[:, 1], s=18, c=C_RARE,
               alpha=0.75, edgecolors='none', label='Water pixels (rare)')

    # Prototypes (stars)
    ax.scatter(head_protos[:, 0], head_protos[:, 1], s=240, marker='*',
               c=C_HEAD, edgecolors='black', linewidths=0.8, zorder=5)
    ax.scatter(rare_proto[:, 0], rare_proto[:, 1], s=300, marker='*',
               c=C_RARE, edgecolors='black', linewidths=0.8, zorder=5)
    # Prototype labels
    for i, p in enumerate(head_protos, start=1):
        ax.annotate(rf'$\mu_{i}$', xy=(p[0]+0.18, p[1]+0.18),
                    fontsize=10, color='black')
    ax.annotate(r'$\mu_4$', xy=(rare_proto[0, 0]+0.20, rare_proto[0, 1]+0.20),
                fontsize=10, color='black')

    # Class cloud title labels
    ax.text(2.0, 9.2, 'Head class: Building\n(high frequency)',
            fontsize=9.5, color=C_HEAD, fontweight='bold')
    ax.text(7.2, 1.4, 'Rare class: Water\n(low frequency)',
            fontsize=9.5, color=C_RARE, fontweight='bold')

    # Teacher decision boundary: a biased curve pulled toward the rare cloud.
    # Drawn as a quadratic through a few control points.
    if show_boundary:
        xs = np.linspace(0.5, 10.5, 200)
        # Bias the boundary to cut the rare cluster — passes close to (7, 4.5)
        ys = 0.07 * (xs - 5.5)**2 + 4.2
        ax.plot(xs, ys, color=C_BNDY, linestyle='--', linewidth=1.7,
                label="Teacher's decision boundary (biased)", zorder=3)
        # Inline label directly on the boundary curve
        ax.text(0.7, 4.7, "EMA teacher\ndecision boundary",
                fontsize=8.2, color=C_BNDY, fontweight='bold',
                rotation=-18,
                bbox=dict(boxstyle='round,pad=0.18', fc='white',
                          ec=C_BNDY, lw=0.6, alpha=0.92))


# =========================================================================
# Panel A: The Setup
# =========================================================================
axA = axes[0]
draw_base_layout(axA)

# Victim pixel — drawn larger and circled
axA.scatter(victim[0], victim[1], s=80, c=C_RARE, marker='o',
            edgecolors=C_VICTIM, linewidths=2.0, zorder=6)
axA.add_patch(Circle(victim, 0.45, facecolor='none',
                     edgecolor=C_VICTIM, linewidth=2.0, linestyle='-', zorder=6))

# Annotation: teacher boundary bias
axA.annotate("Teacher boundary pulled\ntoward rare class\n(class-imbalance bias)",
             xy=(6.0, 5.0), xytext=(0.3, 8.4),
             fontsize=9, color=C_BNDY,
             arrowprops=dict(arrowstyle='->', color=C_BNDY, lw=1.2,
                             connectionstyle='arc3,rad=0.2'))

# Annotation: the victim pixel
axA.annotate("Victim pixel:\n true = Water\n teacher = Building\n $\\hat{c}=0.97$",
             xy=(victim[0]+0.4, victim[1]+0.1), xytext=(8.7, 6.6),
             fontsize=8.6, color=C_VICTIM,
             bbox=dict(boxstyle='round,pad=0.3', fc='white',
                       ec=C_VICTIM, lw=1.0),
             arrowprops=dict(arrowstyle='->', color=C_VICTIM, lw=1.2,
                             connectionstyle='arc3,rad=-0.2'))

# Side-of-boundary labels
axA.text(2.3, 4.5, "teacher: Building", fontsize=8.5, color=C_HEAD,
         style='italic', alpha=0.9)
axA.text(9.0, 8.4, "teacher: Water", fontsize=8.5, color=C_RARE,
         style='italic', alpha=0.9)

axA.set_title("(a) Setup — the EMA teacher's biased boundary leaves\n"
              "rare-class pixels on the wrong side, predicted with HIGH confidence",
              fontsize=11, fontweight='bold', pad=12)


# =========================================================================
# Panel B: What Reweighting Does
# =========================================================================
axB = axes[1]
draw_base_layout(axB)

# Victim pixel highlighted same way
axB.scatter(victim[0], victim[1], s=80, c=C_RARE, marker='o',
            edgecolors=C_VICTIM, linewidths=2.0, zorder=6)
axB.add_patch(Circle(victim, 0.45, facecolor='none',
                     edgecolor=C_VICTIM, linewidth=2.0, zorder=6))
axB.annotate("victim pixel", xy=(victim[0]+0.4, victim[1]-0.1),
             xytext=(victim[0]+1.2, victim[1]-0.8),
             fontsize=8.6, color=C_VICTIM,
             arrowprops=dict(arrowstyle='->', color=C_VICTIM, lw=1.0))

# Case boxes
case_A = (
    "Case A — high confidence ($\\hat{c}=0.97$):\n"
    "  reweighting KEEPS the wrong label\n"
    "  at full weight → confirmation bias"
)
case_B = (
    "Case B — borderline ($\\hat{c}=0.75$):\n"
    "  reweighting downgrades →\n"
    "  rare pixel silently LOST"
)
axB.text(0.4, 1.7, case_A, fontsize=8.2,
         bbox=dict(boxstyle='round,pad=0.4', fc='#FFEFEF',
                   ec=C_VICTIM, lw=0.9))
axB.text(0.4, 0.4, case_B, fontsize=8.2,
         bbox=dict(boxstyle='round,pad=0.4', fc='#FFF6E0',
                   ec='#D6B656', lw=0.9))

# Big "NO RECOVERY" stamp
axB.text(5.5, 9.2, "✗  NO RECOVERY — the wrong label persists",
         fontsize=11.5, color='#B40000', fontweight='bold',
         ha='center',
         bbox=dict(boxstyle='round,pad=0.4', fc='white',
                   ec='#B40000', lw=1.4))

axB.set_title("(b) Reweighting cannot fix the label —\nit only adjusts how loudly the WRONG label speaks",
              fontsize=11, fontweight='bold', pad=12)


# =========================================================================
# Panel C: What P1 Does — Flip via Likelihood Ratio
# =========================================================================
axC = axes[2]
draw_base_layout(axC, show_boundary=False)
# Still show the teacher boundary as faded (for context) so the reader can see what's being overruled
xs = np.linspace(0.5, 10.5, 200)
ys = 0.07 * (xs - 5.5)**2 + 4.2
axC.plot(xs, ys, color=C_BNDY, linestyle='--', linewidth=1.0, alpha=0.35, zorder=2)

# Victim pixel
axC.scatter(victim[0], victim[1], s=80, c=C_RARE, marker='o',
            edgecolors=C_VICTIM, linewidths=2.0, zorder=6)
axC.add_patch(Circle(victim, 0.45, facecolor='none',
                     edgecolor=C_VICTIM, linewidth=2.0, zorder=6))

# Distance lines from victim to nearest head prototype + to rare prototype
nearest_head = head_protos[np.argmin(np.linalg.norm(head_protos - victim, axis=1))]
axC.plot([victim[0], nearest_head[0]], [victim[1], nearest_head[1]],
         color=C_HEAD, linestyle=':', linewidth=1.4, alpha=0.7, zorder=4)
axC.plot([victim[0], rare_proto[0, 0]], [victim[1], rare_proto[0, 1]],
         color=C_RARE, linestyle='-', linewidth=2.2, alpha=0.85, zorder=4)

# Likelihood contours (concentric ellipses) around each prototype that the victim sees
for r, alpha_v in [(0.7, 0.30), (1.3, 0.18), (2.0, 0.10)]:
    axC.add_patch(Ellipse(rare_proto[0], width=2*r, height=2*r*0.75,
                           angle=0, facecolor=C_RARE, alpha=alpha_v,
                           edgecolor='none', zorder=1))
for r, alpha_v in [(1.0, 0.10), (2.0, 0.06)]:
    axC.add_patch(Ellipse(nearest_head, width=2*r, height=2*r*0.75,
                           angle=0, facecolor=C_HEAD, alpha=alpha_v,
                           edgecolor='none', zorder=1))

# Likelihood labels
axC.text(nearest_head[0]-2.0, nearest_head[1]-1.0,
         r"$p(f \mid \mathrm{Building})$" + "\n= LOW",
         fontsize=8.5, color=C_HEAD, fontweight='bold')
axC.text(rare_proto[0, 0]+0.55, rare_proto[0, 1]-1.4,
         r"$p(f \mid \mathrm{Water})$" + "\n= HIGH",
         fontsize=8.5, color=C_RARE, fontweight='bold')

# Likelihood ratio formula box
formula = (
    r"$\Lambda(\mathrm{Water} \mid \mathrm{Building}) = "
    r"\log\frac{p(f\mid\mathrm{Water})}{p(f\mid\mathrm{Building})}"
    r" + \log\frac{\pi^{\mathrm{rare}}_{\mathrm{Water}}}{\pi^{\mathrm{rare}}_{\mathrm{Building}}}$"
    "\n"
    r"$> \tau_{\mathrm{flip}}\ \Rightarrow$ FLIP"
)
axC.text(0.3, 8.7, formula, fontsize=8.6,
         bbox=dict(boxstyle='round,pad=0.4', fc='white',
                   ec=C_FLIP, lw=1.2))

# Curved FLIP arrow from "Building" label to "Water"
old_label_xy = np.array([victim[0]-1.4, victim[1]+1.6])
new_label_xy = np.array([victim[0]+1.4, victim[1]+1.6])
axC.text(old_label_xy[0]-0.5, old_label_xy[1]+0.1, "Building\n(teacher)",
         fontsize=8.5, color=C_HEAD,
         bbox=dict(boxstyle='round,pad=0.25', fc='white',
                   ec=C_HEAD, lw=0.8))
axC.text(new_label_xy[0]-0.2, new_label_xy[1]+0.1, "Water\n(flipped)",
         fontsize=8.5, color=C_RARE, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.25', fc='#E6F2FA',
                   ec=C_RARE, lw=1.2))

flip_arrow = FancyArrowPatch(
    (old_label_xy[0]+0.7, old_label_xy[1]+0.3),
    (new_label_xy[0]-0.1, new_label_xy[1]+0.3),
    arrowstyle='-|>', mutation_scale=22,
    connectionstyle='arc3,rad=-0.5',
    color=C_FLIP, lw=2.6, zorder=7
)
axC.add_patch(flip_arrow)
axC.text((old_label_xy[0]+new_label_xy[0])/2, new_label_xy[1]+1.05,
         "FLIP", fontsize=10.5, color=C_FLIP, fontweight='bold',
         ha='center')

# Bottom annotation
axC.text(5.5, 0.5, "Student trained on CORRECT label.  Rare-class pixel RECOVERED.",
         fontsize=9.5, color=C_FLIP, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.35', fc='#E8F5E9',
                   ec=C_FLIP, lw=1.0))

axC.set_title("(c) P1 — Bayesian likelihood ratio flips the label,\n"
              "recovering the mislabeled rare-class pixel",
              fontsize=11, fontweight='bold', pad=12)


# =========================================================================
# Unified legend (bottom strip)
# =========================================================================
legend_handles = [
    # Pixels
    Line2D([0], [0], marker='o', color='none', markerfacecolor=C_HEAD,
           markeredgecolor='none', markersize=8,
           label='Head-class pixel (Building)'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=C_RARE,
           markeredgecolor='none', markersize=8,
           label='Rare-class pixel (Water)'),
    # Prototypes
    Line2D([0], [0], marker='*', color='none', markerfacecolor=C_HEAD,
           markeredgecolor='black', markeredgewidth=0.8, markersize=15,
           label=r'Head-class prototype  $\mu_1,\mu_2,\mu_3$'),
    Line2D([0], [0], marker='*', color='none', markerfacecolor=C_RARE,
           markeredgecolor='black', markeredgewidth=0.8, markersize=15,
           label=r'Rare-class prototype  $\mu_4$'),
    # Class clouds (filled patches)
    mpatches.Patch(facecolor=C_HEAD, alpha=0.22, edgecolor='none',
                   label='Head-class feature cloud'),
    mpatches.Patch(facecolor=C_RARE, alpha=0.28, edgecolor='none',
                   label='Rare-class feature cloud'),
    # Boundary
    Line2D([0], [0], color=C_BNDY, linestyle='--', linewidth=1.7,
           label="EMA teacher's decision boundary (biased by class imbalance)"),
    # Victim pixel marker
    Line2D([0], [0], marker='o', color='none', markerfacecolor=C_RARE,
           markeredgecolor=C_VICTIM, markeredgewidth=2.0, markersize=11,
           label='Victim pixel (mislabeled by teacher)'),
    # Distance lines (panel c)
    Line2D([0], [0], color=C_HEAD, linestyle=':', linewidth=1.6,
           label=r'Distance: victim $\to$ head prototype  (panel c)'),
    Line2D([0], [0], color=C_RARE, linestyle='-', linewidth=2.2,
           label=r'Distance: victim $\to$ rare prototype  (panel c)'),
    # Density contours
    mpatches.Patch(facecolor=C_HEAD, alpha=0.18, edgecolor='none',
                   label=r'$p(f\mid\mathrm{Building})$ density contour (panel c)'),
    mpatches.Patch(facecolor=C_RARE, alpha=0.30, edgecolor='none',
                   label=r'$p(f\mid\mathrm{Water})$ density contour (panel c)'),
    # Flip arrow
    Line2D([0], [0], color=C_FLIP, linewidth=2.8,
           marker='>', markersize=10, markevery=[-1],
           label='FLIP operation: relabel pixel from teacher class to rare class'),
]

ax_legend.legend(
    handles=legend_handles,
    loc='center',
    ncol=4,
    fontsize=8.6,
    frameon=True,
    fancybox=True,
    framealpha=1.0,
    edgecolor='#888',
    title='Legend — symbols, colors, and lines',
    title_fontsize=10,
    handletextpad=0.7,
    columnspacing=1.2,
    borderpad=0.8,
)

# ---------- Overall title ----------
fig.suptitle(
    "Pseudo-label inversion for rare-class recovery — feature-space view",
    fontsize=13.5, fontweight='bold', y=0.995
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = '/home/ubuntu/S4_SelfTraining/figures/p1_concept.png'
plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"saved {out_path}")
