"""P6 in UDA — the selling point:
   domain shift moves the FEATURE space (breaking prototypes) but leaves
   TOPOLOGICAL constraints invariant.

Layout (2 rows x 3 cols):

  Row 1 = SOURCE domain (labeled)        Row 2 = TARGET domain (unlabeled)
  Col 1 = feature space                  Col 2 = a scene + pseudo-label
  Col 3 = constraint space (shared / invariant)

We show:
  (a) source feature space: prototypes fit the source clusters well.
  (b) target feature space: clusters SHIFTED -> source prototypes misaligned
      -> feature-based reliability fails.
  (c) constraint space is the SAME for both domains -> the noisy pixel is an
      outlier in BOTH -> domain-invariant reliability signal.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'mathtext.fontset': 'dejavusans',
})

C_HEAD  = '#7F7F7F'
C_RARE  = '#1F77B4'
C_NOISE = '#D62728'
C_OK    = '#2CA02C'
C_PROTO = '#FF7F0E'
C_SHIFT = '#9467BD'

rng = np.random.default_rng(11)

fig = plt.figure(figsize=(16.5, 10.4), facecolor='white')
gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.24,
                      left=0.05, right=0.97, top=0.88, bottom=0.10)

ax_sf = fig.add_subplot(gs[0, 0])   # source feature space
ax_tf = fig.add_subplot(gs[1, 0])   # target feature space
ax_ss = fig.add_subplot(gs[0, 1])   # source scene mini
ax_ts = fig.add_subplot(gs[1, 1])   # target scene mini
ax_cs = fig.add_subplot(gs[:, 2])   # constraint space (spans both rows)


def style(ax):
    for s in ax.spines.values():
        s.set_color('#888'); s.set_linewidth(0.9)


# ---- Source / target prototypes (the model's fixed prototypes from source) ----
src_head_c = np.array([3.0, 7.0])
src_rare_c = np.array([7.2, 3.3])
# Target clusters SHIFT due to domain gap
shift = np.array([1.8, -1.4])
tgt_head_c = src_head_c + shift + np.array([0.3, 0.5])
tgt_rare_c = src_rare_c + shift + np.array([-0.4, 0.6])


def draw_feature_space(ax, head_c, rare_c, proto_head, proto_rare,
                       title, show_shift_from=None, misaligned=False):
    ax.set_xlim(0, 12); ax.set_ylim(0, 11)
    ax.set_aspect('equal'); style(ax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("feature dim 1"); ax.set_ylabel("feature dim 2")

    head = rng.multivariate_normal(head_c, [[1.2, 0.2], [0.2, 0.9]], size=55)
    rare = rng.multivariate_normal(rare_c, [[0.5, 0.05], [0.05, 0.4]], size=11)
    ax.add_patch(Ellipse(head_c, 4.4, 3.4, angle=12, facecolor=C_HEAD,
                         alpha=0.15, edgecolor='none'))
    ax.add_patch(Ellipse(rare_c, 2.3, 2.0, angle=10, facecolor=C_RARE,
                         alpha=0.20, edgecolor='none'))
    ax.scatter(head[:, 0], head[:, 1], s=16, c=C_HEAD, alpha=0.55, edgecolors='none')
    ax.scatter(rare[:, 0], rare[:, 1], s=22, c=C_RARE, alpha=0.8, edgecolors='none')

    # The MODEL's prototypes (fixed from source) — orange stars
    ax.scatter([proto_head[0]], [proto_head[1]], s=300, marker='*', c=C_PROTO,
               edgecolors='black', linewidths=1.0, zorder=6)
    ax.scatter([proto_rare[0]], [proto_rare[1]], s=320, marker='*', c=C_PROTO,
               edgecolors='black', linewidths=1.0, zorder=6)

    if misaligned:
        # Draw mismatch: prototype far from the actual (shifted) cluster centre
        ax.add_patch(FancyArrowPatch(proto_rare, rare_c, arrowstyle='->',
                     mutation_scale=16, color=C_NOISE, lw=2.0,
                     linestyle='--', zorder=7))
        midp = (proto_rare + rare_c) / 2
        ax.text(midp[0]+0.2, midp[1]+0.5, "misaligned!", color=C_NOISE,
                fontsize=8.5, fontweight='bold')

    if show_shift_from is not None:
        # Arrow showing the domain shift of cluster centres
        for src, tgt in show_shift_from:
            ax.add_patch(FancyArrowPatch(src, tgt, arrowstyle='-|>',
                         mutation_scale=14, color=C_SHIFT, lw=1.8,
                         alpha=0.7, zorder=4, linestyle=(0, (4, 2))))

    ax.set_title(title, loc='left', fontweight='bold', fontsize=11, pad=7)
    return head, rare


# Row 1 col 1: SOURCE feature space — prototypes fit well
draw_feature_space(ax_sf, src_head_c, src_rare_c, src_head_c, src_rare_c,
                   "(a) SOURCE feature space — prototypes fit",
                   misaligned=False)
ax_sf.text(0.3, 0.4, "prototypes (orange) sit on\nthe source clusters → reliable",
           fontsize=8.2, color='#1A4F1A',
           bbox=dict(boxstyle='round,pad=0.3', fc='#EAF6E9', ec=C_OK, lw=0.8))

# Row 2 col 1: TARGET feature space — clusters shifted, prototypes misaligned
draw_feature_space(ax_tf, tgt_head_c, tgt_rare_c, src_head_c, src_rare_c,
                   "(b) TARGET feature space — domain shift breaks prototypes",
                   show_shift_from=[(src_head_c, tgt_head_c),
                                    (src_rare_c, tgt_rare_c)],
                   misaligned=True)
ax_tf.text(0.3, 0.4,
           "clusters MOVED (purple arrows);\nsource prototypes now off-target\n→ feature reliability FAILS",
           fontsize=8.2, color='#7A1F1F',
           bbox=dict(boxstyle='round,pad=0.3', fc='#FCEAEA', ec=C_NOISE, lw=0.8))


# ---- Scene mini-maps (source + target) with the same topology ----
def draw_scene(ax, title, noise=False):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_aspect('equal'); style(ax)
    ax.set_xticks([]); ax.set_yticks([])
    # background vegetation
    ax.add_patch(Rectangle((0, 0), 10, 10, facecolor='#5FA052', edgecolor='none'))
    # a connected river (curved) — same topology in both domains
    xs = np.linspace(0.5, 9.5, 60)
    ys = 5 + 2.6 * np.sin(xs * 0.55)
    ax.plot(xs, ys, color='#3F77BB', lw=8, solid_capstyle='round', zorder=3)
    # a building block
    ax.add_patch(Rectangle((1.2, 7.2), 2.0, 1.8, facecolor='#444', edgecolor='none', zorder=4))
    ax.add_patch(Rectangle((6.6, 1.0), 2.2, 1.6, facecolor='#444', edgecolor='none', zorder=4))
    # a road
    ax.add_patch(Rectangle((0, 2.0), 10, 0.7, facecolor='#9C9C9C', edgecolor='none', zorder=2))

    if noise:
        # a hallucinated water pixel floating in vegetation
        ax.scatter([3.0], [8.6], s=70, marker='s', c='#3F77BB',
                   edgecolors=C_NOISE, linewidths=1.8, zorder=6)
        ax.annotate("noisy 'water'\n(floating, 1 px)", xy=(3.0, 8.6),
                    xytext=(3.6, 9.2), fontsize=7.6, color=C_NOISE,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec=C_NOISE, lw=0.7),
                    arrowprops=dict(arrowstyle='->', color=C_NOISE, lw=0.8))
    ax.set_title(title, loc='left', fontweight='bold', fontsize=10.5, pad=6)


draw_scene(ax_ss, "SOURCE scene (labeled)", noise=False)
draw_scene(ax_ts, "TARGET scene + teacher pseudo-label", noise=True)

# Add a note between the two scenes about invariant topology
ax_ss.text(5, -0.8, "same scene grammar: river connected, buildings solid, water hosted",
           ha='center', fontsize=8.0, color='#444', style='italic',
           transform=ax_ss.transData)


# ---- Constraint space (shared, invariant) ----
ax_cs.set_xlim(0, 10); ax_cs.set_ylim(0, 11)
ax_cs.set_aspect('auto'); style(ax_cs)
ax_cs.set_xticks([]); ax_cs.set_yticks([])
ax_cs.set_xlabel(r"$s_1$: component area")
ax_cs.set_ylabel(r"$s_2$: neighbor compatibility")
ax_cs.set_title("(c) Constraint space — INVARIANT across domains",
                loc='left', fontweight='bold', fontsize=11, pad=7)

# Feasible set
ax_cs.add_patch(FancyBboxPatch((4.0, 4.0), 5.6, 6.4,
                boxstyle="round,pad=0.1,rounding_size=0.3",
                facecolor=C_OK, alpha=0.12, edgecolor=C_OK, lw=1.3))
ax_cs.text(6.8, 10.0, "feasible set\n(constraints satisfied)", color='#1A4F1A',
           fontsize=9, ha='center', fontweight='bold')

# Valid pixels from BOTH domains land in the SAME feasible set
src_valid = rng.normal([6.6, 7.4], [1.1, 1.2], size=(30, 2)); src_valid = np.clip(src_valid, 4.2, 9.5)
tgt_valid = rng.normal([6.9, 6.9], [1.1, 1.2], size=(30, 2)); tgt_valid = np.clip(tgt_valid, 4.2, 9.5)
ax_cs.scatter(src_valid[:, 0], src_valid[:, 1], s=22, marker='o',
              facecolor=C_HEAD, edgecolors='none', alpha=0.55,
              label='valid pixels — SOURCE')
ax_cs.scatter(tgt_valid[:, 0], tgt_valid[:, 1], s=22, marker='^',
              facecolor=C_RARE, edgecolors='none', alpha=0.7,
              label='valid pixels — TARGET')

# The noisy pixel lands OUTSIDE feasible set in BOTH domains (same place!)
noise_struct = np.array([1.5, 1.7])
ax_cs.scatter([noise_struct[0]], [noise_struct[1]], s=170, marker='X',
              c=C_NOISE, edgecolors='black', linewidths=1.3, zorder=7)
ax_cs.annotate("noisy pixel — outlier in\nBOTH domains\n(tiny + incompatible)",
               xy=noise_struct, xytext=(2.0, 6.0), fontsize=8.4, color=C_NOISE,
               bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_NOISE, lw=0.9),
               arrowprops=dict(arrowstyle='->', color=C_NOISE, lw=1.1,
                               connectionstyle='arc3,rad=-0.2'))

# violation threshold
xl = np.linspace(0, 10, 80)
ax_cs.plot(xl, 8.4 - xl, color=C_OK, lw=1.5, ls='--', alpha=0.8)

ax_cs.text(0.3, 0.5,
           "Topology does NOT shift with domain:\n"
           "a river is connected in any city.\n"
           "→ domain-invariant reliability signal",
           fontsize=8.6, color='#1A4F1A',
           bbox=dict(boxstyle='round,pad=0.3', fc='#EAF6E9', ec=C_OK, lw=0.9))

ax_cs.legend(loc='upper left', fontsize=8.0, frameon=True, framealpha=0.95)


# ---- Big arrows linking feature-shift vs constraint-invariance ----
# Suptitle
fig.suptitle(
    "Why P6 shines in UDA:  domain shift breaks feature-space prototypes (left),\n"
    "but topological constraints are domain-invariant (right) — a reliability signal that survives the domain gap",
    fontsize=12.8, fontweight='bold', y=0.975)

# Footer takeaway
fig.text(0.5, 0.035,
         "Feature-based rivals (PDSSNet, ProSFDA, DGLE) rely on the LEFT column — they degrade under domain shift.  "
         "P6 adds the RIGHT column — invariant structural priors applied to pseudo-labels (no target labels needed).",
         ha='center', fontsize=9.2, color='#333', style='italic',
         bbox=dict(boxstyle='round,pad=0.4', fc='#FFFCE5', ec='#D6B656', lw=0.8))

out = '/home/ubuntu/S4_SelfTraining/figures/p6_uda.png'
plt.savefig(out, dpi=170, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"saved {out}")
