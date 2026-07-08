"""P1 — publication-grade concept figure.

Four panels developing the full mechanism with real probability semantics:

  (a) THE PROBLEM
      Class-imbalanced training (P(head) >> P(rare)) pulls the teacher's
      Bayes-optimal decision boundary toward the rare cluster, so geometrically
      rare pixels end up on the head side with high posterior confidence.

  (b) TWO DECISION FUNCTIONS DISAGREE STRUCTURALLY
      Teacher boundary (uses train-set prior) vs. prototype-density Bayes
      boundary (uniform prior). The disagreement region is non-random: it
      coincides exactly with where rare-class pixels concentrate.

  (c) FLIP-ELIGIBLE REGION
      Likelihood ratio  Λ(f) = log p(f|rare) − log p(f|head) + η·log(π_head/π_rare)
      defines a *signed surface*. Contour at Λ = τ_flip is the flip frontier;
      a margin above the Bayes-rare-prior boundary is required, not a knife-edge.

  (d) ASYMMETRIC, BUDGET-CAPPED FLIPPING IN ACTION
      On a simulated batch: flips happen only inside the green region, only
      toward the rare class, only the top-K by margin. Head pixels never flip.

Designed for a CVPR/NeurIPS/AAAI front-tier figure 2.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import (Ellipse, Circle, FancyArrowPatch, Rectangle,
                                FancyBboxPatch)
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.patches as mpatches
import numpy as np

# ---------------- Palette ----------------
C_HEAD    = '#7F7F7F'   # head class
C_RARE    = '#1F77B4'   # rare class
C_TEACHER = '#D62728'   # teacher's decision
C_PROTO   = '#FF7F0E'   # prototype-density Bayes
C_FLIP    = '#2CA02C'   # flip-eligible / outcome
C_AMB     = '#FDB863'   # ambiguous band
C_TEXT    = '#1A1A1A'
C_BG_PANEL= '#FFFFFF'

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9.5,
    'axes.labelsize':    10,
    'axes.titlesize':    11,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'mathtext.fontset':  'dejavusans',
    'axes.linewidth':    0.9,
})

# ============================================================
# Data: two Gaussian mixtures (head = 3 components, rare = 1)
# ============================================================
rng = np.random.default_rng(2026)

head_means = np.array([[2.5, 7.3], [3.6, 6.4], [4.2, 7.7]])
head_cov   = np.array([[0.95, 0.18], [0.18, 0.78]])
rare_mean  = np.array([8.0, 3.0])
rare_cov   = np.array([[0.55, 0.10], [0.10, 0.42]])

# Class priors (training set is heavily imbalanced)
pi_head_train, pi_rare_train = 0.92, 0.08
# P1 rare-prior boost exponent  η  (used in Λ)
eta = 1.0

# Grid
nx, ny = 260, 260
xs = np.linspace(0.2, 10.8, nx)
ys = np.linspace(0.2, 9.8, ny)
X, Y = np.meshgrid(xs, ys)

def gauss2d(X, Y, mean, cov):
    z0 = X - mean[0]; z1 = Y - mean[1]
    inv = np.linalg.inv(cov); det = np.linalg.det(cov)
    q = (z0*z0*inv[0,0] + 2*z0*z1*inv[0,1] + z1*z1*inv[1,1])
    return np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))

p_head = sum(gauss2d(X, Y, m, head_cov) for m in head_means) / len(head_means)
p_rare = gauss2d(X, Y, rare_mean, rare_cov)

# Teacher posterior — uses training prior
denom_tch = pi_head_train * p_head + pi_rare_train * p_rare + 1e-30
post_teacher_rare = pi_rare_train * p_rare / denom_tch

# Bayes posterior under uniform prior  (what prototype-density alone gives)
denom_unif = 0.5 * p_head + 0.5 * p_rare + 1e-30
post_uniform_rare = 0.5 * p_rare / denom_unif

# Likelihood ratio with rare-prior boost
log_lr        = np.log(p_rare + 1e-30) - np.log(p_head + 1e-30)
log_boost     = eta * np.log(pi_head_train / pi_rare_train)
Lambda        = log_lr + log_boost
tau_flip      = 1.5            # margin above Bayes-with-boost
flip_eligible = Lambda > tau_flip

# ============================================================
# Figure
# ============================================================
fig = plt.figure(figsize=(13.6, 12.0), facecolor='white')
gs  = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.20,
                       left=0.05, right=0.97, top=0.92, bottom=0.07)
axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

def _common_axis(ax):
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('feature dim 1', labelpad=2)
    ax.set_ylabel('feature dim 2', labelpad=2)
    for s in ax.spines.values():
        s.set_color('#888'); s.set_linewidth(0.8)

for ax in (axA, axB, axC, axD):
    _common_axis(ax)

def panel_label(ax, letter, body):
    ax.set_title(f"({letter})  {body}", loc='left',
                 fontweight='bold', pad=8, fontsize=11.5)

# ============================================================
# PANEL A — The problem
# ============================================================
panel_label(axA, 'a',
            "Class imbalance shifts the Bayes-optimal boundary toward the rare cluster")

# Class densities as filled contours (alpha-blended)
levels_h = np.linspace(p_head.max()*0.05, p_head.max()*0.95, 4)
levels_r = np.linspace(p_rare.max()*0.05, p_rare.max()*0.95, 4)
axA.contourf(X, Y, p_head, levels=levels_h,
             colors=[(0.55,0.55,0.55,a) for a in (0.10,0.18,0.26,0.34)])
axA.contourf(X, Y, p_rare, levels=levels_r,
             colors=[(0.12,0.47,0.71,a) for a in (0.12,0.22,0.32,0.42)])

# Outlines
axA.contour(X, Y, p_head, levels=[p_head.max()*0.10],
            colors=[C_HEAD], linewidths=0.6, alpha=0.6)
axA.contour(X, Y, p_rare, levels=[p_rare.max()*0.10],
            colors=[C_RARE], linewidths=0.6, alpha=0.7)

# Class labels
axA.text(2.0, 8.7, 'Head class\n$p(f\\mid\\mathrm{head})$',
         color=C_HEAD, fontsize=10, fontweight='bold')
axA.text(7.0, 1.8, 'Rare class\n$p(f\\mid\\mathrm{rare})$',
         color=C_RARE, fontsize=10, fontweight='bold')

# Teacher's biased decision boundary  (P_teacher(rare|f) = 0.5)
ct = axA.contour(X, Y, post_teacher_rare, levels=[0.5],
                 colors=[C_TEACHER], linewidths=2.0, linestyles='--')
axA.clabel(ct, fmt={0.5: r"teacher boundary"}, inline=True, fontsize=8.5)

# Bayes-optimal under UNIFORM prior (the "fair" boundary)
cu = axA.contour(X, Y, post_uniform_rare, levels=[0.5],
                 colors=['#444'], linewidths=1.5, linestyles=':')
axA.clabel(cu, fmt={0.5: r"unbiased Bayes"}, inline=True, fontsize=8.0)

# Inset: class frequency bar
ax_inset = axA.inset_axes([0.03, 0.04, 0.30, 0.18])
ax_inset.bar([0, 1], [pi_head_train*100, pi_rare_train*100],
             color=[C_HEAD, C_RARE], edgecolor='black', linewidth=0.7)
ax_inset.set_xticks([0, 1]); ax_inset.set_xticklabels(['head', 'rare'], fontsize=8)
ax_inset.set_ylabel('%', fontsize=8); ax_inset.set_yticks([0, 50, 100])
ax_inset.tick_params(labelsize=7)
ax_inset.set_title('train freq.', fontsize=8.5, pad=2)
for s in ax_inset.spines.values(): s.set_linewidth(0.6)

# Annotation explaining the bias
axA.annotate(
    "Teacher learns\n"
    r"$P(c\mid f) \propto p(f\mid c)\,P(c)$"
    "\nimbalanced $P(c)$ pulls the\nboundary toward the rare class",
    xy=(6.5, 4.8), xytext=(0.6, 4.3),
    fontsize=8.5, color='#444',
    bbox=dict(boxstyle='round,pad=0.30', fc='white', ec='#999', lw=0.7),
    arrowprops=dict(arrowstyle='->', color='#555', lw=0.9,
                    connectionstyle='arc3,rad=-0.2'))

# ============================================================
# PANEL B — Two decision functions disagree structurally
# ============================================================
panel_label(axB, 'b',
            "Teacher vs. prototype-density Bayes: disagreement coincides with rare cluster")

# Class density faded as context
axB.contourf(X, Y, p_head, levels=levels_h,
             colors=[(0.55,0.55,0.55,a) for a in (0.06,0.12,0.18,0.24)])
axB.contourf(X, Y, p_rare, levels=levels_r,
             colors=[(0.12,0.47,0.71,a) for a in (0.08,0.14,0.20,0.28)])

# Disagreement region: teacher says head, but Bayes-uniform says rare
disagree = (post_teacher_rare < 0.5) & (post_uniform_rare > 0.5)
axB.contourf(X, Y, disagree.astype(float), levels=[0.5, 1.5],
             colors=[(0.46, 0.74, 0.42, 0.40)])

# The two boundaries on top
axB.contour(X, Y, post_teacher_rare, levels=[0.5],
            colors=[C_TEACHER], linewidths=2.0, linestyles='--')
axB.contour(X, Y, post_uniform_rare, levels=[0.5],
            colors=[C_PROTO], linewidths=2.0, linestyles='-')

# Sample rare-cluster pixels: draw 12 points from rare distribution
rare_samples = rng.multivariate_normal(rare_mean, rare_cov, size=18)
post_at_pts = []
for px, py in rare_samples:
    # interpolate posterior values at the point
    ix = np.clip(np.searchsorted(xs, px), 1, nx-1)
    iy = np.clip(np.searchsorted(ys, py), 1, ny-1)
    post_at_pts.append(post_teacher_rare[iy, ix])
post_at_pts = np.array(post_at_pts)
# Misclassified (teacher says head): these are recovery candidates
mis = post_at_pts < 0.5
axB.scatter(rare_samples[mis, 0], rare_samples[mis, 1],
            s=44, c=C_RARE, marker='o',
            edgecolors='#222', linewidths=1.2, zorder=6,
            label='rare pixels mislabeled\nas head by teacher')
axB.scatter(rare_samples[~mis, 0], rare_samples[~mis, 1],
            s=24, c=C_RARE, marker='o',
            edgecolors='none', alpha=0.55, zorder=5)

# Legend entries
leg_b = [
    Line2D([0],[0], color=C_TEACHER, lw=2.0, ls='--',
           label='teacher boundary (biased prior)'),
    Line2D([0],[0], color=C_PROTO, lw=2.0, ls='-',
           label='prototype-density Bayes (uniform prior)'),
    mpatches.Patch(facecolor=(0.46, 0.74, 0.42, 0.40),
                   edgecolor='none',
                   label='disagreement region'),
    Line2D([0],[0], marker='o', color='none',
           markerfacecolor=C_RARE, markeredgecolor='#222',
           markeredgewidth=1.0, markersize=8,
           label='rare pixel inside disagreement'),
]
axB.legend(handles=leg_b, loc='lower left', fontsize=7.8,
           frameon=True, framealpha=0.95)

axB.text(0.55, 9.4,
         "Key observation: the disagreement is not noise.\n"
         "It coincides with the rare-class density — these are the\n"
         "recovery candidates.",
         fontsize=8.6, color='#333',
         bbox=dict(boxstyle='round,pad=0.30', fc='#FFFCE5',
                   ec='#D6B656', lw=0.7))

# ============================================================
# PANEL C — Likelihood ratio surface with τ_flip margin
# ============================================================
panel_label(axC, 'c',
            "Flip frontier $\\Lambda(f)=\\tau_{\\mathrm{flip}}$ requires a margin above unbiased Bayes")

# Build a divergent colormap centered around tau_flip
norm = TwoSlopeNorm(vmin=Lambda.min(), vcenter=tau_flip, vmax=Lambda.max())
im = axC.imshow(Lambda, extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                origin='lower', cmap='RdBu_r', norm=norm,
                alpha=0.85, aspect='equal')

# Two contours: Bayes-with-boost frontier (Λ=0) and P1 flip frontier (Λ=τ_flip)
axC.contour(X, Y, Lambda, levels=[0],
            colors=['#444'], linewidths=1.2, linestyles=':')
axC.contour(X, Y, Lambda, levels=[tau_flip],
            colors=[C_FLIP], linewidths=2.2, linestyles='-')

# Labels for the contour lines
axC.text(3.8, 2.6, r"$\Lambda=0$" + "\n(Bayes-with-boost)",
         fontsize=8.0, color='#222',
         bbox=dict(boxstyle='round,pad=0.20', fc='white', ec='#888', lw=0.6))
axC.text(7.1, 4.5, r"$\Lambda=\tau_{\mathrm{flip}}$" + "\n(P1 flip frontier)",
         fontsize=8.4, color=C_FLIP, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.20', fc='white', ec=C_FLIP, lw=0.8))

# Region labels
axC.text(1.5, 8.5, "$\\Lambda \\ll 0$\nteacher's territory\n(no flip)",
         fontsize=8.0, color='#7A1F1F', ha='left')
axC.text(8.2, 7.5, "$0<\\Lambda<\\tau_{\\mathrm{flip}}$\nambiguous band\n(fall back to weighting)",
         fontsize=8.0, color='#7A4F00', ha='left')
axC.text(8.8, 2.0, "$\\Lambda>\\tau_{\\mathrm{flip}}$\nflip-eligible",
         fontsize=8.5, color=C_FLIP, ha='center', fontweight='bold')

# Formula box
axC.text(0.4, 0.6,
         r"$\Lambda(f) = \log\dfrac{p(f\mid\mathrm{rare})}{p(f\mid\mathrm{head})}"
         r"\;+\;\eta\,\log\dfrac{\pi^{\mathrm{train}}_{\mathrm{head}}}{\pi^{\mathrm{train}}_{\mathrm{rare}}}$",
         fontsize=9.5, color='#111',
         bbox=dict(boxstyle='round,pad=0.30', fc='white', ec='#666', lw=0.8))

# Colorbar
cbar = fig.colorbar(im, ax=axC, fraction=0.04, pad=0.02, shrink=0.85)
cbar.set_label(r"$\Lambda(f)$  (signed log-evidence for rare class)",
               fontsize=8.5)
cbar.ax.tick_params(labelsize=7.5)

# ============================================================
# PANEL D — Asymmetric, budget-capped flipping in action
# ============================================================
panel_label(axD, 'd',
            "Operation: flip only to rare, only above $\\tau_{\\mathrm{flip}}$, "
            "only the top-$K$ by margin")

# Render flip-eligible region softly
axD.contourf(X, Y, flip_eligible.astype(float), levels=[0.5, 1.5],
             colors=[(0.17, 0.63, 0.17, 0.20)])
axD.contour(X, Y, Lambda, levels=[tau_flip],
            colors=[C_FLIP], linewidths=1.6, linestyles='-')

# Faded teacher boundary for context
axD.contour(X, Y, post_teacher_rare, levels=[0.5],
            colors=[C_TEACHER], linewidths=1.2, linestyles='--', alpha=0.6)

# Class density outlines
axD.contour(X, Y, p_head, levels=[p_head.max()*0.10],
            colors=[C_HEAD], linewidths=0.6, alpha=0.5)
axD.contour(X, Y, p_rare, levels=[p_rare.max()*0.10],
            colors=[C_RARE], linewidths=0.6, alpha=0.6)

# Simulate a batch of pixels: mix from both classes
batch_head = rng.multivariate_normal(head_means[1], head_cov, size=70)
batch_rare = rng.multivariate_normal(rare_mean,    rare_cov, size=22)
batch = np.vstack([batch_head, batch_rare])

# Compute Λ at each batch point
def field_lookup(pts, F):
    out = np.empty(len(pts))
    for i, (px, py) in enumerate(pts):
        ix = int(np.clip(np.searchsorted(xs, px), 1, nx-1))
        iy = int(np.clip(np.searchsorted(ys, py), 1, ny-1))
        out[i] = F[iy, ix]
    return out

batch_lambda = field_lookup(batch, Lambda)
batch_teacher_rare = field_lookup(batch, post_teacher_rare) > 0.5
# What the teacher says: 'head' or 'rare'
teacher_label = np.where(batch_teacher_rare, 'rare', 'head')

# Flip eligibility (Λ > τ_flip) and asymmetric constraint (only flip head→rare)
elig = (batch_lambda > tau_flip) & (teacher_label == 'head')

# Budget cap: top-K by Λ
K_budget = 8
elig_idx = np.where(elig)[0]
if len(elig_idx) > K_budget:
    ranked = elig_idx[np.argsort(-batch_lambda[elig_idx])]
    elig_idx_flip   = ranked[:K_budget]
    elig_idx_unflip = ranked[K_budget:]
else:
    elig_idx_flip = elig_idx
    elig_idx_unflip = np.array([], dtype=int)

# Plot pixels:
# (1) teacher-says-head, kept
mask_kept_head = (teacher_label == 'head'); mask_kept_head[elig_idx_flip] = False
axD.scatter(batch[mask_kept_head, 0], batch[mask_kept_head, 1],
            s=22, c=C_HEAD, edgecolors='none', alpha=0.65, zorder=3,
            label='kept: teacher = head')
# (2) teacher-says-rare (these never flip — asymmetric constraint)
mask_kept_rare = (teacher_label == 'rare')
axD.scatter(batch[mask_kept_rare, 0], batch[mask_kept_rare, 1],
            s=28, c=C_RARE, edgecolors='none', alpha=0.75, zorder=3,
            label='kept: teacher = rare')
# (3) eligible but budget-capped (no flip this batch)
axD.scatter(batch[elig_idx_unflip, 0], batch[elig_idx_unflip, 1],
            s=42, facecolors='none', edgecolors='#999',
            linewidths=1.0, marker='o', zorder=4,
            label=f'eligible, over budget (no flip)')
# (4) actually flipped — draw arrow from head-label → rare-label position
for idx in elig_idx_flip:
    px, py = batch[idx]
    # arrow visualization: a short green arrow upward-right indicating the flip
    axD.annotate('', xy=(px+0.10, py+0.10),
                 xytext=(px, py),
                 arrowprops=dict(arrowstyle='-|>', color=C_FLIP, lw=1.8))
axD.scatter(batch[elig_idx_flip, 0], batch[elig_idx_flip, 1],
            s=70, marker='D', c=C_FLIP, edgecolors='black', linewidths=0.8,
            zorder=6, label=f'FLIPPED  (top-$K$, $K={K_budget}$)')

# Asymmetry callout: NO arrows in the other direction
axD.text(0.5, 1.0,
         "Asymmetry: flips only point $\\mathrm{head}\\!\\to\\!\\mathrm{rare}$.\n"
         "Pixels predicted rare by the teacher are never flipped.",
         fontsize=8.4, color='#333',
         bbox=dict(boxstyle='round,pad=0.30', fc='#EAF6E9',
                   ec=C_FLIP, lw=0.8))

axD.legend(loc='upper right', fontsize=7.6,
           frameon=True, framealpha=0.95)

# ============================================================
# Suptitle + caption-like figure footer
# ============================================================
fig.suptitle(
    "P1 — Pseudo-label inversion via prototype-density likelihood ratio:\n"
    "exploiting the structural disagreement between a biased teacher and an unbiased geometric witness",
    fontsize=13, fontweight='bold', y=0.985,
)

footer = (
    r"Setup: $p(f\mid c)$ Gaussian mixtures; teacher posterior uses train prior "
    r"$\pi^{\mathrm{train}}=(0.92,0.08)$; "
    r"$\Lambda$ uses rare-prior boost $\eta=1.0$; flip margin "
    r"$\tau_{\mathrm{flip}}=1.5$; budget $K=8$ flips per batch."
)
fig.text(0.5, 0.015, footer, ha='center', fontsize=8.2,
         color='#444', style='italic')

out = '/home/ubuntu/S4_SelfTraining/figures/p1_concept_v2.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"saved {out}")
