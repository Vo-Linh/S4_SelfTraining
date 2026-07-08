"""P6 — Neuro-Symbolic Constraint Correction for RS-SSL.

Four panels:
  (a)  Constraint library — five named, RS-specific rules with mini examples
  (b)  Teacher's noisy pseudo-label on a simulated aerial scene
  (c)  Violation map V_i — where the constraint library is broken
  (d)  Corrected pseudo-label after constraint projection

Designed in image space (not feature space), matching the user's intuition.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import (Rectangle, FancyBboxPatch, FancyArrowPatch,
                                Circle, Patch)
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# Class palette — aerial classes with distinct readable colors
# ============================================================
CLASSES = [
    ('background', '#E8E2D7'),  # 0 bare ground / background
    ('vegetation', '#5FA052'),  # 1 vegetation
    ('road',       '#9C9C9C'),  # 2 road
    ('building',   '#444444'),  # 3 building
    ('water',      '#3F77BB'),  # 4 water (rare)
    ('agri',       '#F2C247'),  # 5 agriculture (rare)
]
CLASS_NAMES = [c[0] for c in CLASSES]
CLASS_COLORS = [c[1] for c in CLASSES]
CMAP = ListedColormap(CLASS_COLORS)
NORM = BoundaryNorm(np.arange(-0.5, len(CLASSES) + 0.5, 1), CMAP.N)
NCLASS = len(CLASSES)

# ============================================================
# Build a simulated aerial GT scene  H x W
# ============================================================
H, W = 48, 60
gt = np.zeros((H, W), dtype=int)  # default = background

# Vegetation patches (large background)
gt[:, :] = 1
# A road grid
gt[18:21, :] = 2
gt[34:37, :] = 2
gt[:, 14:16] = 2
gt[:, 38:40] = 2
# A few buildings in the urban quadrant (top-left)
gt[5:12, 4:10]   = 3
gt[5:12, 22:30]  = 3
gt[24:30, 4:9]   = 3
gt[24:30, 22:29] = 3
# A river along the bottom-right curve
for r in range(36, H):
    c = 42 + int(8 * np.sin((r - 36) * 0.45))
    if 0 <= c < W:
        gt[r, max(0, c-2):min(W, c+2)] = 4
# Bridge: road crosses water at a single line
gt[36:38, 44:48] = 2
# Agricultural fields (bottom-left, large blocks)
gt[40:H, 0:12] = 5
gt[40:H, 18:30] = 5
# A few background patches (bare ground)
gt[12:14, 15:25] = 0
gt[28:30, 16:24] = 0

# ============================================================
# Simulate a noisy teacher pseudo-label
# (this is what the EMA teacher would output: realistic errors)
# ============================================================
rng = np.random.default_rng(42)
noisy = gt.copy()

# Error 1: floating building pixels in vegetation
for _ in range(5):
    r, c = rng.integers(0, H), rng.integers(30, W)
    noisy[r, c] = 3
# Error 2: a small "water" island floating away from the river (rare-class noise)
noisy[20, 50] = 4
noisy[10, 35] = 4
# Error 3: a piece of building is mislabeled vegetation (rare class lost case mirror)
noisy[7:9, 6:8] = 1
# Error 4: ragged building boundary
for _ in range(20):
    r = rng.integers(5, 12)
    c = rng.choice([3, 4, 10, 11, 22, 30])
    noisy[r, c] = rng.choice([3, 1])
# Error 5: a stray "agriculture" patch in urban area (rare-class hallucination on head)
noisy[16, 25:27] = 5
noisy[6, 12:14] = 5
# Error 6: a chunk of vegetation labeled as background
noisy[14:17, 32:36] = 0
# Error 7: river loses connectivity at a point
noisy[42, 46:50] = 1   # river break
# Error 8: building floats next to river without a road (no-floating-object violation)
noisy[44, 36:38] = 3

# ============================================================
# Build a violation map V_i by applying each constraint
# (simulated; the real implementation would be differentiable layers)
# ============================================================
V = np.zeros((H, W), dtype=float)


def soft_label_equal(a, b):
    return (a == b).astype(float)


def constraint_min_area(y, class_id, area_thresh):
    """Mark pixels that belong to components of class_id with area < area_thresh."""
    from scipy import ndimage
    mask = (y == class_id).astype(int)
    if mask.sum() == 0:
        return np.zeros_like(y, dtype=float)
    labeled, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(n + 1))
    too_small = sizes < area_thresh
    too_small[0] = False
    return too_small[labeled].astype(float)


def constraint_connectivity_break(y, class_id):
    """Pixels at gap positions that break otherwise-connected class regions."""
    from scipy import ndimage
    mask = (y == class_id).astype(int)
    if mask.sum() < 10:
        return np.zeros_like(y, dtype=float)
    labeled, n = ndimage.label(mask)
    # Mark gap pixels: not-of-class but with multiple class neighbors of different components
    out = np.zeros_like(y, dtype=float)
    if n <= 1:
        return out
    H, W = y.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if y[r, c] == class_id:
                continue
            nb = labeled[r-1:r+2, c-1:c+2]
            comps = set(int(v) for v in nb.flatten() if v != 0)
            if len(comps) >= 2:
                out[r, c] = 1.0
    return out


def constraint_no_floating(y, class_id, host_classes):
    """Flag class_id components whose boundary is dominated by non-host classes."""
    from scipy import ndimage
    mask = (y == class_id).astype(int)
    if mask.sum() == 0:
        return np.zeros_like(y, dtype=float)
    labeled, n = ndimage.label(mask)
    out = np.zeros_like(y, dtype=float)
    for ci in range(1, n + 1):
        comp = (labeled == ci)
        if comp.sum() == 0:
            continue
        dil = ndimage.binary_dilation(comp) & (~comp)
        nb_labels = y[dil]
        if len(nb_labels) == 0:
            continue
        frac_host = np.mean([1 if l in host_classes else 0 for l in nb_labels])
        if frac_host < 0.30:  # mostly non-host neighbors -> floating
            out[comp] = 1.0
    return out


def constraint_boundary_jagged(y, class_id):
    """Mark boundary pixels with high local curvature for class_id."""
    from scipy import ndimage
    mask = (y == class_id).astype(float)
    if mask.sum() == 0:
        return np.zeros_like(y, dtype=float)
    # Edges
    grad = ndimage.sobel(mask, axis=0)**2 + ndimage.sobel(mask, axis=1)**2
    edge = grad > 0.5
    # Curvature ~ laplacian of edge indicator
    curv = np.abs(ndimage.laplace(mask))
    out = edge.astype(float) * curv
    return out / (out.max() + 1e-6)


# Apply the constraints
V += 1.0 * constraint_min_area(noisy, class_id=4, area_thresh=8)        # water islands
V += 1.0 * constraint_min_area(noisy, class_id=3, area_thresh=4)        # tiny buildings
V += 1.0 * constraint_min_area(noisy, class_id=5, area_thresh=10)       # tiny agri
V += 0.8 * constraint_connectivity_break(noisy, class_id=4)             # river breaks
V += 1.0 * constraint_no_floating(noisy, class_id=3, host_classes={1, 2, 0})  # building floating
V += 0.6 * constraint_boundary_jagged(noisy, class_id=3)                # jagged building edges
V = V / (V.max() + 1e-9)


# ============================================================
# Build the corrected pseudo-label by applying constraints
# (greedy correction: remove small components, fill connectivity gaps)
# ============================================================
from scipy import ndimage

corrected = noisy.copy()

# Remove too-small components (replace with majority neighbor class)
def remove_small(y, class_id, area_thresh):
    mask = (y == class_id).astype(int)
    labeled, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(n + 1))
    for ci in range(1, n + 1):
        if sizes[ci] < area_thresh:
            comp = (labeled == ci)
            dil = ndimage.binary_dilation(comp) & (~comp)
            nb_labels = y[dil]
            if len(nb_labels) == 0:
                continue
            most = np.bincount(nb_labels.astype(int), minlength=NCLASS).argmax()
            y[comp] = most
    return y

for cid, thr in [(4, 8), (3, 4), (5, 10), (0, 8)]:
    corrected = remove_small(corrected, cid, thr)

# Fix river connectivity break: pixels at the break -> set to water if both sides are water
mask_water = (corrected == 4)
for r in range(40, H):
    if mask_water[r].any():
        cs = np.where(mask_water[r])[0]
        if len(cs) >= 2:
            for c in range(cs.min(), cs.max() + 1):
                # If a gap pixel is currently non-water and would otherwise be a small island
                if not mask_water[r, c]:
                    # only fill if neighbors above/below are water
                    above = mask_water[max(0, r - 1), c]
                    below = mask_water[min(H - 1, r + 1), c]
                    if above or below:
                        corrected[r, c] = 4

# Replace remaining stray building floating into river area
corrected[44, 36:38] = 1

# Remove the "agri in urban" hallucination by majority-neighbor
corrected = remove_small(corrected, 5, 10)


# ============================================================
# Figure
# ============================================================
fig = plt.figure(figsize=(15.0, 11.0), facecolor='white')
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.18,
                      left=0.04, right=0.97, top=0.90, bottom=0.10)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9.5,
    'mathtext.fontset':  'dejavusans',
})


def panel_label(ax, letter, title):
    ax.set_title(f"({letter})  {title}", loc='left',
                 fontweight='bold', pad=8, fontsize=11.5)


def show_seg(ax, y, title=None):
    ax.imshow(y, cmap=CMAP, norm=NORM, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#888'); s.set_linewidth(0.8)


# ============================================================
# Panel A — Constraint library (catalog of named rules)
# ============================================================
panel_label(axA, 'a', "Constraint library  $\\mathcal{C} = \\{c_1, \\dots, c_5\\}$ — RS-specific differentiable rules")
axA.set_xlim(0, 100); axA.set_ylim(0, 60); axA.axis('off')


def draw_mini(ax, x, y, size, pattern, title, rule, color_id, ok=True):
    """Draw a tiny class map illustrating a constraint."""
    cell = size / pattern.shape[0]
    for ri in range(pattern.shape[0]):
        for ci in range(pattern.shape[1]):
            v = pattern[ri, ci]
            if v < 0:
                continue
            ax.add_patch(Rectangle(
                (x + ci * cell, y + size - (ri + 1) * cell),
                cell, cell,
                facecolor=CLASS_COLORS[int(v)], edgecolor='none'))
    ax.add_patch(Rectangle((x, y), size, size,
                           facecolor='none', edgecolor='#333', linewidth=0.6))
    ax.text(x + size + 1, y + size * 0.72, title,
            fontsize=8.6, fontweight='bold', color='#222')
    ax.text(x + size + 1, y + size * 0.38, rule,
            fontsize=7.6, color='#444')


# c1: Connectivity (water)
p1 = -np.ones((6, 6), dtype=int)
p1[2:5, 0:3] = 4
p1[2:5, 4:6] = 4  # gap at column 3
p1[5, :] = 1
draw_mini(axA, 2, 35, 16, p1,
          "$c_1$  Connectivity",
          "water region must form a\nsingle connected component",
          4)

# c2: Minimum area
p2 = -np.ones((6, 6), dtype=int)
p2[:, :] = 1
p2[1, 4] = 3  # single floating building pixel
p2[3, 1] = 4  # single floating water pixel
draw_mini(axA, 2, 8, 16, p2,
          "$c_2$  Min. area",
          "no isolated single-pixel\nbuilding / water / agri",
          3)

# c3: Boundary regularity
p3 = -np.ones((6, 6), dtype=int)
p3[1:5, 1:5] = 3
p3[1, 1] = 1; p3[4, 4] = 1
p3[2, 0] = 3; p3[3, 5] = 3
draw_mini(axA, 33, 35, 16, p3,
          "$c_3$  Boundary regularity",
          "building edges should be\nsmooth / near axis-aligned",
          3)

# c4: Co-occurrence
p4 = -np.ones((6, 6), dtype=int)
p4[:, :] = 1
p4[2, 2] = 3      # building
p4[1:4, 4:6] = 4  # water adjacent — implausible without bridge
draw_mini(axA, 33, 8, 16, p4,
          "$c_4$  Co-occurrence",
          "neighbor class pairs follow\nlearned $P_{cooc}(y_i, y_j)$",
          1)

# c5: No-floating-object
p5 = -np.ones((6, 6), dtype=int)
p5[:, :] = 4
p5[2:4, 2:4] = 3  # building floating in water
draw_mini(axA, 64, 35, 16, p5,
          "$c_5$  No-floating-object",
          "no building inside water\n(unless 'bridge' present)",
          3)

# Math box for the projection
axA.text(64, 6,
         r"$\tilde{q} = \arg\min_{q}\;"
         r"D_{\mathrm{KL}}(q\,\|\,\hat{q}) \;+\; \lambda \sum_{k} w_k\, c_k(q)$"
         "\n\n"
         "MAP inference with the\nconstraint library as the prior",
         fontsize=8.5, color='#111',
         bbox=dict(boxstyle='round,pad=0.30', fc='#FFFCE5', ec='#D6B656', lw=0.8))

# ============================================================
# Panel B — Teacher's noisy pseudo-label
# ============================================================
panel_label(axB, 'b', "Teacher pseudo-label $\\hat{y}$ on an aerial scene — multiple constraint violations visible")
show_seg(axB, noisy)

# Annotate the visible errors
annots = [
    (10.5, 36, "stray\nwater"),
    (20.5, 50, "floating\nwater"),
    (44, 37, "building\non river"),
    (16, 26, "agri in\nurban"),
    (15, 34, "missing\nveg"),
    (42, 47, "river\nbreak"),
    (7, 7, "ragged\nbuilding"),
]
for (r, c, txt) in annots:
    axB.annotate(txt, xy=(c, r),
                 xytext=(c + 6, r - 6) if c < W - 12 else (c - 10, r - 6),
                 fontsize=7.0, color='black',
                 bbox=dict(boxstyle='round,pad=0.18', fc='white',
                           ec='#B40000', lw=0.7, alpha=0.92),
                 arrowprops=dict(arrowstyle='->', color='#B40000', lw=0.7))

# ============================================================
# Panel C — Violation map V_i
# ============================================================
panel_label(axC, 'c', "Per-pixel violation map $V_i = \\sum_k w_k\\,\\partial_{q_i} c_k(\\hat{q})$ — dense reliability signal")
# Overlay V on a faded version of the noisy map for context
axC.imshow(noisy, cmap=CMAP, norm=NORM, interpolation='nearest', alpha=0.30)
im = axC.imshow(V, cmap='inferno', interpolation='nearest', alpha=0.85, vmin=0, vmax=1)
axC.set_xticks([]); axC.set_yticks([])
for s in axC.spines.values():
    s.set_color('#888'); s.set_linewidth(0.8)

# Inline interpretation
axC.text(2, 4, "bright = violates\none or more constraints",
         fontsize=8.5, color='white',
         bbox=dict(boxstyle='round,pad=0.25', fc='#222',
                   ec='white', lw=0.6, alpha=0.92))

cbar = fig.colorbar(im, ax=axC, fraction=0.04, pad=0.02, shrink=0.85)
cbar.set_label("$V_i$  (normalized violation)", fontsize=8.5)
cbar.ax.tick_params(labelsize=7.5)

# ============================================================
# Panel D — Corrected pseudo-label after constraint projection
# ============================================================
panel_label(axD, 'd',
            "Corrected pseudo-label $\\tilde{y}$ after constraint projection — student trains on this")
show_seg(axD, corrected)

# Show where corrections happened
diff = (corrected != noisy)
ys, xs_ = np.where(diff)
axD.scatter(xs_, ys, s=8, marker='o', facecolor='none',
            edgecolor='#2CA02C', linewidths=0.8, alpha=0.9)

axD.text(2, 4, "circles = pixels\nrelabeled",
         fontsize=8.5, color='#1A4F1A',
         bbox=dict(boxstyle='round,pad=0.25', fc='white',
                   ec='#2CA02C', lw=0.8))

# ============================================================
# Pipeline strip at the very bottom
# ============================================================
ax_pipe = fig.add_axes([0.04, 0.025, 0.93, 0.06])
ax_pipe.axis('off')
ax_pipe.set_xlim(0, 100); ax_pipe.set_ylim(0, 10)

steps = [
    ("Teacher", "$\\hat{q}$", '#FFEFEF', '#D62728'),
    ("Constraint library", "$\\mathcal{C}$", '#FFFCE5', '#D6B656'),
    ("Violation map", "$V_i$", '#FFF3E0', '#E67E22'),
    ("MAP projection", "$\\tilde{q}$", '#E8F5E9', '#2CA02C'),
    ("Student", "$\\mathcal{L}_{\\mathrm{ssl}}$", 'white', '#444'),
]
for i, (name, sym, fc, ec) in enumerate(steps):
    x = 2 + i * 19.5
    ax_pipe.add_patch(FancyBboxPatch(
        (x, 1.5), 16, 6.5,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        facecolor=fc, edgecolor=ec, linewidth=1.4))
    ax_pipe.text(x + 8, 5.6, name, ha='center', va='center',
                 fontsize=9, fontweight='bold', color=ec)
    ax_pipe.text(x + 8, 3.0, sym, ha='center', va='center',
                 fontsize=10, color='#222')
    if i < len(steps) - 1:
        ax_pipe.annotate('', xy=(x + 19.5, 4.8), xytext=(x + 16, 4.8),
                         arrowprops=dict(arrowstyle='-|>', color='#444', lw=1.6))

# ============================================================
# Legend strip — class colors
# ============================================================
ax_leg = fig.add_axes([0.04, 0.005, 0.93, 0.025])
ax_leg.axis('off')
handles = [Patch(facecolor=c, edgecolor='#333', linewidth=0.5,
                 label=name) for name, c in CLASSES]
ax_leg.legend(handles=handles, loc='center', ncol=len(CLASSES),
              fontsize=8.6, frameon=False, handletextpad=0.5,
              columnspacing=1.4)

# ============================================================
# Title
# ============================================================
fig.suptitle(
    "P6 — Neuro-symbolic constraint correction for RS-SSL\n"
    "RS-specific logical rules used both as a dense reliability signal and as a label-correction operator",
    fontsize=13, fontweight='bold', y=0.98,
)

out = '/home/ubuntu/S4_SelfTraining/figures/p6_concept.png'
plt.savefig(out, dpi=170, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"saved {out}")
