"""P6 — why it lives in a DIFFERENT space than feature-space methods.

The central scientific claim of P6:
   Teacher noise on rare classes is often INVISIBLE in feature space
   (the noisy pixel's feature looks like a legitimate rare-class feature),
   but becomes a clear OUTLIER in 'constraint-satisfaction space'
   (size, connectivity, neighbor-compatibility, containment).

Three panels:
  (a) FEATURE SPACE — feature-based / prototype methods see the noisy 'water'
      pixel sitting INSIDE the water cluster. It is indistinguishable from a
      true water pixel. Feature reliability cannot separate them.
  (b) CONSTRAINT-SATISFACTION SPACE — the SAME pixels re-plotted on
      structural axes (component-area  vs  neighbor-compatibility). Now the
      noisy pixel is a clear outlier: tiny isolated component, incompatible
      neighbors. P6 separates what feature space could not.
  (c) THE MAPPING — schematic showing each pixel carries TWO descriptions:
      a feature vector f (continuous appearance) and a structure vector s
      (discrete label-map properties). P6 adds the s-axis that feature
      methods ignore.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'mathtext.fontset': 'dejavusans',
})

C_HEAD = '#7F7F7F'
C_RARE = '#1F77B4'
C_NOISE = '#D62728'   # the noisy / hallucinated pixel
C_TRUE = '#2CA02C'
C_OK = '#2CA02C'

rng = np.random.default_rng(7)

fig = plt.figure(figsize=(17, 6.4), facecolor='white')
gs = fig.add_gridspec(1, 3, wspace=0.22, left=0.04, right=0.985,
                      top=0.84, bottom=0.13)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])


def style(ax):
    for s in ax.spines.values():
        s.set_color('#888'); s.set_linewidth(0.9)


def panel_title(ax, letter, body):
    ax.set_title(f"({letter})  {body}", loc='left', fontweight='bold',
                 fontsize=11.5, pad=9)


# ==========================================================================
# PANEL A — FEATURE SPACE (what prototype / feature methods see)
# ==========================================================================
panel_title(axA, 'a', "Feature space — the noise is INVISIBLE here")
axA.set_xlim(0, 10); axA.set_ylim(0, 10)
axA.set_aspect('equal'); style(axA)
axA.set_xticks([]); axA.set_yticks([])
axA.set_xlabel("feature dim 1"); axA.set_ylabel("feature dim 2")

# Head cluster (big, many points)
head_c = np.array([3.0, 7.0])
head = rng.multivariate_normal(head_c, [[1.3, 0.2], [0.2, 1.0]], size=60)
# Rare cluster (small, few points)
rare_c = np.array([7.3, 3.3])
rare = rng.multivariate_normal(rare_c, [[0.5, 0.05], [0.05, 0.4]], size=11)

axA.add_patch(Ellipse(head_c, 4.6, 3.6, angle=12, facecolor=C_HEAD,
                      alpha=0.15, edgecolor='none'))
axA.add_patch(Ellipse(rare_c, 2.4, 2.0, angle=10, facecolor=C_RARE,
                      alpha=0.20, edgecolor='none'))
axA.scatter(head[:, 0], head[:, 1], s=20, c=C_HEAD, alpha=0.6,
            edgecolors='none')
axA.scatter(rare[:, 0], rare[:, 1], s=26, c=C_RARE, alpha=0.8,
            edgecolors='none')

# Prototype stars
axA.scatter([head_c[0]], [head_c[1]], s=260, marker='*', c=C_HEAD,
            edgecolors='black', linewidths=0.8, zorder=5)
axA.scatter([rare_c[0]], [rare_c[1]], s=300, marker='*', c=C_RARE,
            edgecolors='black', linewidths=0.8, zorder=5)

# THE NOISY PIXEL: a hallucinated "water" pixel whose FEATURE sits right
# inside the water cluster -> looks perfectly legitimate
noise_feat = rare_c + np.array([0.15, 0.1])
axA.scatter([noise_feat[0]], [noise_feat[1]], s=150, marker='X',
            c=C_NOISE, edgecolors='black', linewidths=1.2, zorder=7)
axA.annotate("noisy 'water' pixel\n(teacher hallucination)\n"
             r"feature $\approx$ true water" + "\nIMPOSSIBLE to flag here",
             xy=(noise_feat[0], noise_feat[1]),
             xytext=(2.4, 1.2), fontsize=8.6, color=C_NOISE,
             bbox=dict(boxstyle='round,pad=0.30', fc='white',
                       ec=C_NOISE, lw=0.9),
             arrowprops=dict(arrowstyle='->', color=C_NOISE, lw=1.1,
                             connectionstyle='arc3,rad=0.2'))

# Class labels
axA.text(1.2, 9.0, "Head class", color=C_HEAD, fontsize=10, fontweight='bold')
axA.text(6.0, 1.6, "Rare class (water)", color=C_RARE, fontsize=10,
         fontweight='bold')

axA.text(0.3, 0.3,
         "Feature / prototype methods:\nnoisy pixel is INSIDE the cluster\n→ high feature reliability → kept",
         fontsize=8.4, color='#7A1F1F',
         bbox=dict(boxstyle='round,pad=0.3', fc='#FCEAEA', ec=C_NOISE, lw=0.8))


# ==========================================================================
# PANEL B — CONSTRAINT-SATISFACTION SPACE (what P6 sees)
# ==========================================================================
panel_title(axB, 'b', "Constraint space — the SAME pixel is now an OUTLIER")
axB.set_xlim(0, 10); axB.set_ylim(0, 10)
axB.set_aspect('equal'); style(axB)
axB.set_xticks([]); axB.set_yticks([])
axB.set_xlabel(r"$s_1$:  component area  (small $\to$ large)")
axB.set_ylabel(r"$s_2$:  neighbor compatibility  $P_{\mathrm{cooc}}$")

# Region of "valid" structure: large area + compatible neighbors (top-right)
valid_region = mpatches.FancyBboxPatch(
    (4.2, 4.2), 5.4, 5.2,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    facecolor=C_OK, alpha=0.12, edgecolor=C_OK, lw=1.2, linestyle='-')
axB.add_patch(valid_region)
axB.text(6.9, 9.0, "feasible set\n(constraints satisfied)", color='#1A4F1A',
         fontsize=9, ha='center', fontweight='bold')

# True water pixels: large connected component, compatible neighbors -> top-right
true_water_s = rng.normal([7.0, 7.2], [0.7, 0.7], size=(11, 2))
axB.scatter(true_water_s[:, 0], true_water_s[:, 1], s=26, c=C_RARE, alpha=0.8,
            edgecolors='none', zorder=4)
# True head pixels: also large area + compatible -> top-right but spread
true_head_s = rng.normal([6.2, 6.6], [1.1, 1.0], size=(60, 2))
true_head_s = np.clip(true_head_s, 4.4, 9.4)
axB.scatter(true_head_s[:, 0], true_head_s[:, 1], s=18, c=C_HEAD, alpha=0.5,
            edgecolors='none', zorder=3)

# THE NOISY PIXEL in structure space: tiny isolated component (s1 small),
# incompatible neighbors (s2 small) -> bottom-left, far OUTSIDE feasible set
noise_struct = np.array([1.4, 1.6])
axB.scatter([noise_struct[0]], [noise_struct[1]], s=150, marker='X',
            c=C_NOISE, edgecolors='black', linewidths=1.2, zorder=7)
axB.annotate("same noisy 'water' pixel\n"
             "• area = 1 px  (tiny)\n"
             "• neighbors = vegetation\n  (incompatible)\n"
             "CLEAR OUTLIER → flag/correct",
             xy=(noise_struct[0], noise_struct[1]),
             xytext=(2.4, 5.6), fontsize=8.6, color=C_NOISE,
             bbox=dict(boxstyle='round,pad=0.30', fc='white',
                       ec=C_NOISE, lw=0.9),
             arrowprops=dict(arrowstyle='->', color=C_NOISE, lw=1.1,
                             connectionstyle='arc3,rad=-0.2'))

# Decision frontier (violation threshold)
xline = np.linspace(0, 10, 100)
axB.plot(xline, 8.6 - xline, color=C_OK, lw=1.6, ls='--', alpha=0.8)
axB.text(0.4, 8.6, "violation\nthreshold", color='#1A4F1A', fontsize=8,
         rotation=-45, ha='left', va='top')

# A correction arrow: pushing the noisy pixel toward the feasible set
arr = FancyArrowPatch(noise_struct + [0.3, 0.3], [5.5, 5.7],
                      arrowstyle='-|>', mutation_scale=18,
                      color=C_OK, lw=2.2,
                      connectionstyle='arc3,rad=0.25', zorder=6)
axB.add_patch(arr)
axB.text(3.7, 3.4, "constraint\nprojection", color=C_OK, fontsize=8.5,
         fontweight='bold', rotation=38)


# ==========================================================================
# PANEL C — THE MAPPING: each pixel has TWO descriptions
# ==========================================================================
panel_title(axC, 'c', "Each pixel carries two descriptions; P6 adds the structural one")
axC.set_xlim(0, 10); axC.set_ylim(0, 10)
axC.axis('off')

# Central pixel icon
axC.add_patch(FancyBboxPatch((4.2, 5.6), 1.6, 1.6,
              boxstyle="round,pad=0.05,rounding_size=0.2",
              facecolor=C_NOISE, edgecolor='black', lw=1.2))
axC.text(5.0, 6.4, "pixel\n$i$", ha='center', va='center', color='white',
         fontsize=9, fontweight='bold')

# Left branch: feature vector f
axC.add_patch(FancyBboxPatch((0.4, 7.6), 3.4, 1.8,
              boxstyle="round,pad=0.15,rounding_size=0.3",
              facecolor='#EFEFEF', edgecolor='#666', lw=1.0))
axC.text(2.1, 8.9, "appearance  $f_i$", ha='center', fontsize=9.5,
         fontweight='bold', color='#333')
axC.text(2.1, 8.15,
         "colour, texture,\ncontext embedding",
         ha='center', fontsize=8.2, color='#444')
axC.annotate('', xy=(4.2, 6.7), xytext=(3.8, 8.0),
             arrowprops=dict(arrowstyle='-|>', color='#666', lw=1.6,
                             connectionstyle='arc3,rad=0.2'))
axC.text(2.7, 7.1, "used by feature /\nprototype methods", fontsize=7.8,
         color='#666', style='italic')

# Right branch: structure vector s
axC.add_patch(FancyBboxPatch((6.2, 7.6), 3.4, 1.8,
              boxstyle="round,pad=0.15,rounding_size=0.3",
              facecolor='#EAF6E9', edgecolor=C_OK, lw=1.2))
axC.text(7.9, 8.9, "structure  $s_i$", ha='center', fontsize=9.5,
         fontweight='bold', color='#1A4F1A')
axC.text(7.9, 8.15,
         "area, connectivity,\nneighbors, containment",
         ha='center', fontsize=8.2, color='#1A4F1A')
axC.annotate('', xy=(5.8, 6.7), xytext=(6.2, 8.0),
             arrowprops=dict(arrowstyle='-|>', color=C_OK, lw=1.8,
                             connectionstyle='arc3,rad=-0.2'))
axC.text(6.0, 7.0, "added by P6\n(everyone ignores)", fontsize=7.8,
         color=C_OK, style='italic', fontweight='bold')

# Bottom: combined reliability
axC.add_patch(FancyBboxPatch((1.8, 2.3), 6.4, 2.2,
              boxstyle="round,pad=0.15,rounding_size=0.3",
              facecolor='#FFFCE5', edgecolor='#D6B656', lw=1.2))
axC.text(5.0, 3.95, "P6 reliability decision", ha='center', fontsize=10,
         fontweight='bold', color='#7A5C00')
axC.text(5.0, 3.05,
         r"reliable $\Leftrightarrow$ $f_i$ near a prototype  AND  $s_i$ in feasible set"
         "\nnoise that fools $f$ is caught by $s$",
         ha='center', fontsize=8.6, color='#5A4500')

axC.annotate('', xy=(3.9, 4.5), xytext=(4.7, 5.6),
             arrowprops=dict(arrowstyle='-|>', color='#999', lw=1.4))
axC.annotate('', xy=(6.1, 4.5), xytext=(5.3, 5.6),
             arrowprops=dict(arrowstyle='-|>', color='#999', lw=1.4))

# Key takeaway box
axC.add_patch(FancyBboxPatch((0.4, 0.2), 9.2, 1.5,
              boxstyle="round,pad=0.15,rounding_size=0.3",
              facecolor='#F3F3F3', edgecolor='#333', lw=1.0))
axC.text(5.0, 0.95,
         "Key idea: feature space and constraint space are COMPLEMENTARY.\n"
         "P6 separates noise that is geometrically inseparable in feature space.",
         ha='center', va='center', fontsize=9.0, color='#222',
         fontweight='bold')


# ==========================================================================
# Suptitle + legend
# ==========================================================================
fig.suptitle(
    "Why P6 is not a feature-space method — it adds a complementary 'constraint-satisfaction' axis\n"
    "The same noisy rare-class pixel: invisible in feature space (a), an obvious outlier in constraint space (b)",
    fontsize=12.5, fontweight='bold', y=0.99)

handles = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor=C_HEAD,
           markersize=9, label='head-class pixel'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=C_RARE,
           markersize=9, label='true rare-class pixel'),
    Line2D([0], [0], marker='*', color='none', markerfacecolor='#555',
           markeredgecolor='black', markersize=15, label='class prototype'),
    Line2D([0], [0], marker='X', color='none', markerfacecolor=C_NOISE,
           markeredgecolor='black', markersize=12,
           label='noisy / hallucinated pixel (teacher error)'),
    mpatches.Patch(facecolor=C_OK, alpha=0.18, edgecolor=C_OK,
                   label='feasible set / valid structure'),
]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
           frameon=True, framealpha=0.95, edgecolor='#888',
           bbox_to_anchor=(0.5, -0.005))

out = '/home/ubuntu/S4_SelfTraining/figures/p6_featurespace.png'
plt.savefig(out, dpi=175, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"saved {out}")
