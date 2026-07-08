# DWPC — Dual-Witness Pseudo-label Correction for Rare-Class Recovery in Remote-Sensing SSL/UDA

**Status:** Full research proposal (combines P1 + TARS)
**Date:** 2026-05-29
**Author:** L. Vo
**Framework:** DAPCN-SSL (this repository)
**Provisional name:** DWPC (Dual-Witness Pseudo-label Correction)
**Target venues:** CVPR / ICCV / NeurIPS (method) · TGRS / ISPRS J. PRS (domain)
**Figure:** `figures/dwpc_concept.png` (concept) · supporting: `figures/p1_concept_v2.png`, `figures/p6_uda.png`

---

## 0. One-paragraph summary

A semi-supervised / domain-adaptive segmentation teacher (EMA) [Tarvainen & Valpola, 2017] is both **biased** (class imbalance pulls its decision boundary off rare classes) and **noisy** (domain shift corrupts its predictions on the target domain). Existing pseudo-label correctors use a *single* evidence source and inherit its blind spot: **prototype/feature methods** [ProDA, Zhang et al., 2021] hallucinate corrections when features are ambiguous, and **topological/structural methods** [clDice, Shit et al., 2021; Kervadec et al., 2018; TopoSemiSeg, Xu et al., 2024; Iqbal et al., 2023] *delete the very rare-class objects they should protect*. DWPC cross-examines every pseudo-label with **two independent witnesses** — an *appearance* witness (prototype-density likelihood ratio with a rare-class prior) and a *domain-invariant structure* witness (target-adaptive plausibility: area, connectivity, co-occurrence, containment). The pseudo-label is **flipped to a rare class only when both witnesses agree**, and **protected whenever either witness vouches for it**. Because the two witnesses have *complementary, non-overlapping failure modes*, their cross-examination repairs what each does wrong alone.

---

## 1. High-Level Concept

### 1.1 The two problems are one problem

Rare-class collapse and target-domain pseudo-label noise are the **same phenomenon** seen twice: the EMA teacher confidently mislabels rare-class pixels as head classes. Reweighting — the saturated 2022-era response [U²PL, Wang et al., 2022; FreeMatch, Wang et al., 2023] — cannot fix this: a wrong label at reduced weight is still wrong. The pixel must be **recovered** (its label changed), not merely down-weighted. This mirrors the long-tail finding that loss reshaping alone [Focal, Lin et al., 2017; Equalization, Tan et al., 2020] leaves systematically mislabeled minority instances uncorrected.

### 1.2 Why one witness is never enough

| Single-evidence corrector | Mechanism | Structural blind spot |
|---|---|---|
| **ProDA** (CVPR 2021) — appearance only | prototype-distance softmax → hard pseudo-label replacement | hallucinates: any pixel whose *feature* drifts near another class flips, even if spatially impossible |
| **clDice / Kervadec / TopoSemiSeg / topology-UDA** — structure only | topology / size / connectivity loss on the mask | **deletes small true rare objects** (min-area erases a true pond; erosion kills thin rivers) |

These two families are **disjoint** (verified, §3): nobody fuses appearance evidence with structural evidence for pseudo-label correction. That disjointness is the opening.

### 1.3 The dual-witness idea

Treat correction as **cross-examination by two witnesses with opposite failure modes**:

- **Witness A — Appearance** (domain-*sensitive*): prototype-density likelihood. Good at recognizing what a pixel *looks like*; fooled by domain shift and ambiguous appearance.
- **Witness B — Structure** (domain-*invariant*): topological/geometric plausibility of the label map. Good at catching spatially impossible labels; blind to appearance, and (in prior work) destructive to small rare objects.

The cross-examination (full table in §4.4, figure `dwpc_concept.png`):

| Witness A | Witness B | Action |
|---|---|---|
| says rare $c^*$ | plausible as $c^*$ | **FLIP** → recover rare pixel (budgeted) |
| says rare $c^*$ | implausible | **ABSTAIN** — likely appearance hallucination → down-weight |
| no rare evidence | current label implausible | **DOWN-WEIGHT** — structural noise |
| no rare evidence | plausible | **KEEP** at full weight |

Two principles fall out:
1. **Flip only on dual agreement** → kills ProDA's hallucinated flips (Witness B vetoes).
2. **Protect a pixel whenever either witness vouches** → kills the topology methods' rare-object deletion (Witness A gates the structural constraint off).

The witnesses *mutually repair* each other. That mutual repair — not either witness individually — is the contribution.

---

## 2. Problem Formulation

Single-domain SSL (`DAPCN_SSL`) or cross-domain UDA (`DAPCN`). Labeled set $\mathcal{L}$, unlabeled/target set $\mathcal{U}$. Student $f^S$, EMA teacher $f^T$. For an unlabeled pixel $i$ with backbone feature $f_i$:
- teacher softmax $\hat q_i \in \Delta^C$, hard pseudo-label $\hat y_i=\arg\max_c \hat q_{i,c}$, confidence $\hat c_i=\max_c \hat q_{i,c}$.
- rare class set $\mathcal{C}_{\text{rare}}$ = bottom-$r$ classes by labeled-set pixel frequency (**fixed before any experiment**).

Goal: produce a corrected pseudo-label $\tilde y_i$ and per-pixel loss weight $w_i$ that (i) recover rare-class pixels and (ii) suppress noise, **without deleting small true rare objects**.

---

## 3. Related Work Analysis (verified 2026-05-29)

### 3.1 Prototype / feature pseudo-label correction — *Witness A lineage*

- **ProDA** [Zhang et al., 2021]. Prototype-distance softmax $\omega_t^{(i,k)}=\mathrm{softmax}_k(-\lVert \tilde f_i-\eta^{(k)}\rVert/\tau)$ multiplies the frozen initial softmax; $\arg\max$ gives a **hard, online-rectified** pseudo-label. **This is the skeleton of P1's flip — we do NOT claim prototype-based flipping as novel.** ProDA is *symmetric* (any class can win), single-centroid-per-class, and **treats classes equally regardless of frequency** — i.e., no rare-class mechanism.
- **ProtoGMM** [Moradinasab et al., 2024]. GMM-over-multiple-prototypes for DA, used for **contrastive learning**, not pseudo-label flipping. Anticipates DWPC's *GMM-likelihood density model*; cited defensively. No flip, no rare prior.
- **ProSFDA** [Wang et al., 2025]. **Soft** prototype-similarity loss weighting for RS source-free DA; no flip, no rare prior. A core DAPCN competitor — appearance-only, and its reliability is circular under domain shift (prototypes drift with the corrupted encoder).
- **CBST / CRST** [Zou et al., 2018; 2019]. Class-balanced self-training: per-class confidence thresholds via percentiles lower the bar for rare classes. **Selection/rejection, not a likelihood-ratio flip**; no prototypes, no feature density, no budget.

**Delta for DWPC Witness A over this lineage:** explicit log-likelihood-**ratio** with a margin $\tau_{\text{flip}}$, a **rare-class prior boost** (ProDA disavows this), **asymmetric flip-to-rare-only**, and a **per-image flip budget**. This is a *narrow* delta and is NOT the paper's headline — it is one witness.

### 3.2 Topological / structural constraints — *Witness B lineage*

- **clDice** [Shit et al., 2021] — centerline-Dice topology loss; tubular/binary only; soft-erosion **deletes thin structures**.
- **Constrained-CNN losses** [Kervadec et al., 2018] — differentiable size/area inequality; authors state **"no guarantee constraints are satisfied"**; **min-area penalizes correct small rare objects**.
- **TopoSemiSeg** (Xu et al., ECCV 2024, arXiv 2311.16447) — persistence-diagram signal/noise split for teacher-student SSL; histopathology blobs; static persistence threshold **deletes low-persistence (small/rare) topology**; the paper itself notes the teacher's topology is not always correct.
- **Graph-Theoretic Consistency / TGC** (Pham et al., *AAAI 2026 Student Abstract*, arXiv 2509.22689) — Laplacian-spectrum/component-count constraints for SSL; **global statistics, not per-pixel localizable**; can merge/split rare objects. *(Student-abstract venue — cite as evidence of direction, not as a full competing method.)*
- **Leveraging Topology for DA Road Segmentation** (Iqbal et al., ISPRS J. Photogrammetry & Remote Sensing 2023, arXiv 2309.15625) — connectivity-based PL refinement for UDA aerial; **roads only**, single constraint, calls the skeleton "more domain-invariant." This pre-empts a naive "topology is domain-invariant" headline — DWPC must position against it (multi-class, rare-protective, fused-with-appearance).
- **Topology-preserving losses** more broadly [Hu et al., NeurIPS 2019] enforce Betti-number agreement; supervised, single-property, and not rare-class-aware.

**Delta for DWPC Witness B over this lineage:** (i) **multi-class land-cover** constraints (not tubular/blob/binary), (ii) **target-adaptive** parameters (EMA-estimated on the target stream, not static), (iii) **rare-class-gated** so the constraint never deletes a true rare object, (iv) used as a **per-pixel reliability + correction** signal, not only a loss.

### 3.3 Reliability-gated / class-balanced families (distinguish, do not conflate)

- Reliability/uncertainty-gated CE weighting: RAC-Net (arXiv 2303.05164), FARCLUSS (arXiv 2506.11142), U²PL [Wang et al., 2022] — weight the *loss*; do **not** gate a structural constraint.
- Class-balanced loss: DropLoss [Hsieh et al., AAAI 2021], equalization loss [Tan et al., CVPR 2020], size-balanced CE (WACV 2024, arXiv 2309.14117) — rebalance *loss magnitude*; DropLoss is the closest *switch-off-for-rare* mechanism but on classification CE, **not** a structural constraint.
- Multi-class topology: **Topologically Faithful Multi-class Segmentation** [Berger et al., MICCAI 2024] — per-class Betti matching, but topology is **enforced for every class** (per-class ≠ rare-gated-off).

### 3.4 Verified novelty position

| Claim | Verdict (2026-05-29) |
|---|---|
| Prototype-likelihood flip (Witness A core) | **Anticipated** by ProDA — credit openly |
| Rare-prior + asymmetric + budgeted flip | Narrow but **open** delta over ProDA/CBST |
| Rare-class-**gated** structural constraint | **Genuinely open** (no prior; closest DropLoss = wrong loss type, 2403.11001 = per-class not gated-off) |
| **Fusion** of appearance + structural per-pixel reliability for PL correction | **Open / partially anticipated** — defensible if framed precisely |
| **Dual-witness mutual-veto** decision rule | **Open** — the headline |

*Residual verification (do before submission):* one scoped pass on `"constraint relaxation"+"instance-aware"`, `"adaptive constraint"+"minority"`; confirm any 26xx arXiv IDs exist.

---

## 4. Method

### 4.1 Notation

| Symbol | Meaning |
|---|---|
| $\{\mu_k\}_{k=1}^K$, $\kappa(k)\in\{1..C\}$ | prototype bank (DynamicAnchor) + soft class binding |
| $q_k$ | MLP prototype quality gate (existing) |
| $\bar\pi_c$ | inverse-frequency prior, $\bar\pi_c\propto(1/N_c)^{\gamma}$ |
| $\hat A_c^{(t)}, \hat P^{(t)}_{\text{cooc}}$ | **target-adaptive** EMA estimates of per-class plausible area and co-occurrence |
| $\mathcal{C}_{\text{rare}}$ | rare class set, fixed a priori |
| $a_i, s_i(c)$ | Witness A rare-evidence, Witness B plausibility |

### 4.2 Witness A — Appearance (prototype-density likelihood ratio)

Per-class likelihood from a GMM over class-bound prototypes (shared isotropic $\sigma_c^2$ for stability):
$$p(f_i\mid c) = \sum_{k:\kappa(k)=c}\pi_k\,\mathcal{N}(f_i;\mu_k,\sigma_c^2 I).$$

Rare-prior-boosted log-likelihood ratio of the best alternative against the teacher's call:
$$\Lambda_i(c) = \log\frac{p(f_i\mid c)}{p(f_i\mid \hat y_i)} \;+\; \eta\,\log\frac{\bar\pi_{\hat y_i}}{\bar\pi_c},\qquad c^*_i=\arg\max_{c}\Lambda_i(c).$$

Appearance rare-evidence (asymmetric — only rare targets count):
$$\boxed{\,a_i = \sigma\!\big(\beta_a\,\Lambda_i(c^*_i)\big)\cdot \mathbb 1\!\left[c^*_i\in\mathcal C_{\text{rare}}\right]\,}$$

> **Lineage:** $p(f\mid c)$+argmax-replacement is ProDA [Zhang et al., 2021]; the multi-prototype GMM density follows ProtoGMM [Moradinasab et al., 2024]; the **ratio, $\eta$-rare-prior, and asymmetry** are the delta (§3.1), drawing the rare-favoring intuition from class-balanced self-training [Zou et al., 2018] but realized as a Bayesian flip rather than threshold selection.

### 4.3 Witness B — Domain-invariant structure (target-adaptive, rare-gated)

For a candidate class $c$ at pixel $i$, structural plausibility fuses differentiable terms (each $\in[0,1]$):
$$s_i(c) = \big[\underbrace{g_{\text{area}}(i,c)}_{\text{vs }\hat A_c^{(t)}}\big]^{\lambda_1}\big[\underbrace{g_{\text{conn}}(i,c)}_{\text{connectivity}}\big]^{\lambda_2}\big[\underbrace{g_{\text{cooc}}(i,c)}_{\text{vs }\hat P^{(t)}_{\text{cooc}}}\big]^{\lambda_3}\big[\underbrace{g_{\text{host}}(i,c)}_{\text{containment}}\big]^{\lambda_4}.$$

**Target-adaptive (domain-invariant transfer).** $\hat A_c^{(t)},\hat P^{(t)}_{\text{cooc}}$ are EMA-updated from the **target pseudo-label stream** (no target labels needed):
$$\hat A_c^{(t)} = m\,\hat A_c^{(t-1)} + (1-m)\,\bar A_c^{\text{(this batch, conf.)}}.$$
Topology of land cover does not shift across domains (a river is connected anywhere), so $s_i$ is a **shift-invariant** reliability source — the property feature/prototype methods lack (fig. `p6_uda.png`).

**Rare-class gate (the load-bearing novel primitive).** Structural penalties are suppressed exactly where appearance evidence supports a rare class — so Witness B can never delete a true rare object Witness A recognizes:
$$\boxed{\,\tilde c_k(i)=c_k(i)\cdot\big(1-a_i\big)\,}\qquad\text{(constraint }c_k\text{ gated by Witness-A rare-evidence }a_i\text{)}.$$

This is conceptually the *switch-off-for-rare* move of DropLoss [Hsieh et al., 2021] and equalization loss [Tan et al., 2020], but applied to a **structural constraint** rather than the classification CE term — a transfer that, to our knowledge, is unclaimed (§3.4). Class binding $\kappa(k)$ is obtained by Sinkhorn-Knopp balanced assignment [Caron et al., 2020], guaranteeing each class — including rare ones — receives prototypes.

### 4.4 Dual-witness decision

**Flip score** (agreement of both witnesses on a rare target):
$$F_i = a_i\cdot s_i(c^*_i).$$
**Flip rule** — asymmetric to rare, margin $\tau_{\text{flip}}$, per-image budget $\beta_{\text{flip}}|\mathcal U|$ (top-$K$ by $F_i$):
$$\tilde y_i = \begin{cases}c^*_i & F_i>\tau_{\text{flip}}\ \wedge\ i\in\text{top-}K(F)\ \wedge\ c^*_i\in\mathcal C_{\text{rare}}\\[2pt]\hat y_i & \text{otherwise.}\end{cases}$$

**Dual-witness reliability weight** for non-flipped pixels (fuse both; rare-protective via $1-(1-\cdot)(1-\cdot)$ "either vouches" form):
$$\boxed{\,w_i = \underbrace{\sigma(\beta_s\,s_i(\hat y_i))}_{\text{structure, shift-invariant}}\cdot\Big[1-(1-a_i)\big(1-\underbrace{r^{\text{conf}}_i}_{\text{teacher margin}}\big)\Big]\,}$$
so a structurally-small-but-appearance-rare pixel keeps a high weight (protected), while a pixel both structurally implausible and unsupported by appearance is down-weighted.

### 4.5 Final objective

$$\mathcal L_{\text{ssl}} = \frac{1}{|\mathcal U|}\sum_{i\in\mathcal U} w_i\cdot\mathrm{CE}\big(z^S_i,\tilde y_i\big),\qquad
\mathcal L = \mathcal L_{\text{sup}} + \lambda_u\mathcal L_{\text{ssl}} + \lambda_{\text{dapg}}\mathcal L_{\text{DAPG}} + \lambda_{\text{str}}\!\sum_k \tilde c_k.$$

Warmup: flipping enabled only after `proto_correction_start_iter` (prototypes + $\hat A_c,\hat P_{\text{cooc}}$ must stabilize); $\omega(t)$ ramps Witness B in (avoids cold-start, §7).

### 4.6 Integration with DAPCN-SSL six-step loop

Witness A reuses the **DynamicAnchor prototype bank + quality gate** (already present). Witness B is a new module on the teacher pseudo-label map. The flip + weight slot into step 3 (pseudo-label generation) and step 5 (mixed-image CE weight, where labeled patches are pasted via ClassMix [Olsson et al., 2021] as in DACS [Tranheden et al., 2021]). The backbone is the MiT-B5 / DAFormer decoder [Hoyer et al., 2022]; no backbone change. ~460 LoC total (§8).

---

## 5. Why DWPC repairs both failure modes (the argument the paper must win)

| Failure | Owner | DWPC repair |
|---|---|---|
| Hallucinated flip (feature drifts near wrong class) | ProDA (Witness A alone) | $F_i = a_i\cdot s_i(c^*)$ → **Witness B vetoes** ($s_i\!\to\!0$ for an isolated pixel) |
| Deleting a small true rare object | clDice/Kervadec/TopoSemiSeg (Witness B alone) | gated constraint $c_k(1-a_i)$ → **Witness A protects** ($a_i\!\to\!1$) |
| Reliability circular under domain shift | ProSFDA (feature reliability) | $w_i$ includes shift-invariant $s_i$ term |
| Constraint can't localize which pixel is wrong | TGC (global graph stats) | $s_i(c)$ is **per-pixel** |

The figure `figures/dwpc_concept.png` panels (c)/(d) visualize the two mutual rescues.

---

## 6. Hypotheses (preregistered falsifiers)

| # | Hypothesis | Pass / fail (committed before experiments) |
|---|---|---|
| **H1** | DWPC > DAPCN-SSL baseline, OEM + LoveDA mIoU | mean $\Delta\ge 1.5$, 3 seeds, $p<0.05$ |
| **H2 (rare)** | rare-class IoU (bottom-3, fixed a priori) improves | $\ge +3$ mIoU, head regression $\le 1$, $p<0.05$ |
| **H3 (dual-witness necessity)** | DWPC > Witness-A-only (≈ProDA+rare) **and** > Witness-B-only | each $\ge +1.0$ mIoU; if either fails, the *fusion* claim dies |
| **H4 (rare-gate necessity)** | removing the gate $c_k(1-a_i)\to c_k$ **hurts** rare IoU | rare-IoU drop $\ge 1.5$ when gate removed (proves the gate, not just the constraint, drives gains) |
| **H5 (veto necessity)** | removing structural veto ($F_i=a_i$) increases wrong flips | flip-precision drop $\ge 10$ pts on labeled val |
| **H6 (domain-invariance)** | Witness B alone degrades **less** than Witness A alone across the UDA gap | A-only mIoU drop > B-only mIoU drop, source→target |
| **H7 (super-additivity)** | full > sum of single-witness gains | $\Delta_{\text{full}} > 0.8(\Delta_A+\Delta_B)$ — else "engineering of known parts", reframe |
| **H8 (SOTA)** | beats MUCA [arXiv 2501.10736] and RS-MTDF [arXiv 2506.08772]; within 2 mIoU of S5 [Lv et al., 2025] (S5 has a foundation-model pretraining advantage) | OEM/LoveDA 1/5/10% |

**H3, H4, H7 are decisive** — they are exactly what a reviewer who knows ProDA + topology losses will demand.

---

## 7. Preregistered Ablation Matrix

| Row | Witness A | A: rare-prior+asym+budget | Witness B | B: target-adaptive | B: rare-gate | Purpose |
|---|:-:|:-:|:-:|:-:|:-:|---|
| A0 | ✗ | ✗ | ✗ | ✗ | ✗ | DAPCN-SSL baseline |
| A1 | ✓ (ProDA-style symmetric) | ✗ | ✗ | ✗ | ✗ | ProDA [Zhang et al., 2021] reproduced on MiT-B5 backbone |
| A2 | ✓ | ✓ | ✗ | ✗ | ✗ | Witness A full (P1) |
| A3 | ✗ | ✗ | ✓ | ✓ | ✗ | Witness B, no gate (≈topology methods) |
| A4 | ✗ | ✗ | ✓ | ✓ | ✓ | Witness B, rare-gated |
| **A5** | ✓ | ✓ | ✓ | ✓ | ✓ | **full DWPC** |
| A6 | ✓ | ✓ | ✓ | ✗ static | ✓ | tests target-adaptivity (UDA) |
| A7 | ✓ | ✓ | ✓ (no veto in $F$) | ✓ | ✓ | tests flip-veto (H5) |

A3 vs A4 isolates the **rare-gate** (H4). A2/A4 vs A5 isolates **fusion super-additivity** (H7). A1 vs A2 isolates the **rare-prior delta over ProDA**.

---

## 8. Implementation Cost

| Module | LoC | File |
|---|---:|---|
| Witness A: GMM likelihood + ratio + rare prior | ~90 | new `mmseg/models/utils/witness_appearance.py` |
| Witness B: area/conn/cooc/host plausibility (differentiable) | ~140 | new `mmseg/models/utils/witness_structure.py` |
| Target-adaptive EMA of $\hat A_c,\hat P_{\text{cooc}}$ (pre-mix) | ~40 | `dapcn_ssl.py` |
| Dual-witness flip + budget + weight | ~60 | `dapcn_ssl.py` |
| Sinkhorn class-binding $\kappa$ (from TARS) | ~80 | reused |
| Flip-precision + stratified eval (H2/H5) | ~50 | `tools/dwpc_eval.py` |
| **Total** | **~460** | |

---

## 9. Minimum Viable Experiment

Run rows **A2, A4, A5** only (Witness A full, Witness B rare-gated, full DWPC) on OEM 5%/10%, 3 seeds. Decisive checks:
- If **A5 ≯ max(A2, A4)** → no fusion benefit → H7 fails → reframe as a single-witness paper (weaker).
- If **A4 ≯ A3** (gate adds nothing) → the rare-gate primitive is inert → return to drawing board.

This MVE tests the headline before building the full multi-scale / MAP-projection machinery.

---

## 10. Risks & Mitigations

1. **"Engineering of known parts."** The strongest reviewer attack (DWPC composes ProDA [Zhang et al., 2021], topology losses [Shit et al., 2021; Kervadec et al., 2018], and rare-class switch-off [Hsieh et al., 2021]). Defense = H7 super-additivity + A1 ProDA-on-backbone baseline + A3 topology-on-backbone baseline + the conceptual mutual-repair argument (§5). If H7 fails, do not submit as a top-tier method paper.
2. **Cold-start of Witness B.** $\hat A_c,\hat P_{\text{cooc}}$ noisy early → warmup $\omega(t)$, enable flips only post-warmup.
3. **Differentiable structural terms unstable.** Soft connected-components / area are finicky. Mitigation: start with non-differentiable Witness B used *only* for the flip decision + weight (no gradient), add differentiability later if needed.
4. **Rare-set definition gaming.** Fix $\mathcal C_{\text{rare}}$ by labeled-set frequency before any run; report all classes.
5. **Residual prior-art.** Do the §3.4 scoped verification pass before camera-ready.

---

## 11. References

Citation style: author-year. All author/title/venue fields below were verified against the arXiv abstract pages on 2026-05-29 unless marked. Entries marked **[author-unverified]** have a verified title/venue/arXiv ID but the first-author name was taken from prior project notes, not re-verified this session — confirm before camera-ready. No reference is included that could not be located.

**Foundational SSL / UDA segmentation**

1. Tarvainen, A., & Valpola, H. (2017). Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. *NeurIPS 2017.* arXiv:1703.01780. *(canonical; standard cite)*
2. Sohn, K., Berthelot, D., Li, C.-L., et al. (2020). FixMatch: Simplifying semi-supervised learning with consistency and confidence. *NeurIPS 2020.* arXiv:2001.07685. *(canonical; standard cite)*
3. Olsson, V., Tranheden, W., Pinto, J., & Svensson, L. (2021). ClassMix: Segmentation-based data augmentation for semi-supervised learning. *WACV 2021.* arXiv:2007.07936.
4. Tranheden, W., Olsson, V., Pinto, J., & Svensson, L. (2021). DACS: Domain adaptation via cross-domain mixed sampling. *WACV 2021.* arXiv:2007.08702.
5. Hoyer, L., Dai, D., & Van Gool, L. (2022). DAFormer: Improving network architectures and training strategies for domain-adaptive semantic segmentation. *CVPR 2022.* arXiv:2111.14887. *(this repo's base)*
6. Wang, Y., Wang, H., Shen, Y., Fei, J., Li, W., Jin, G., Wu, L., Zhao, R., & Le, X. (2022). Semi-supervised semantic segmentation using unreliable pseudo-labels (U²PL). *CVPR 2022.* arXiv:2203.03884.
7. Wang, Y., Chen, H., Heng, Q., et al. (2023). FreeMatch: Self-adaptive thresholding for semi-supervised learning. *ICLR 2023.* arXiv:2205.07246.

**Witness A lineage — prototype / feature pseudo-label correction**

8. Zhang, P., Zhang, B., Zhang, T., Chen, D., Wang, Y., & Wen, F. (2021). Prototypical pseudo label denoising and target structure learning for domain adaptive semantic segmentation (**ProDA**). *CVPR 2021.* arXiv:2101.10979.
9. Moradinasab, N., Shankman, L. S., Deaton, R. A., Owens, G. K., & Brown, D. E. (2024). ProtoGMM: Multi-prototype Gaussian-mixture-based domain adaptation model for semantic segmentation. arXiv:2406.19225.
10. Zou, Y., Yu, Z., Vijaya Kumar, B. V. K., & Wang, J. (2018). Domain adaptation for semantic segmentation via class-balanced self-training (**CBST**). *ECCV 2018.* arXiv:1810.07911.
11. Zou, Y., et al. (2019). Confidence regularized self-training (**CRST**). *ICCV 2019.* arXiv:1908.09822. **[author-unverified]**
12. Wang, B., Deng, F., Chen, Z., Yu, Z., & Liu, Y. (2025). Prototype-based pseudo-label denoising for source-free domain adaptation in remote sensing semantic segmentation (**ProSFDA**). arXiv:2509.16942.

**Witness B lineage — topological / structural constraints**

13. Shit, S., Paetzold, J. C., Sekuboyina, A., Ezhov, I., Unger, A., Zhylka, A., Pluim, J. P. W., Bauer, U., & Menze, B. H. (2021). clDice — A novel topology-preserving loss function for tubular structure segmentation. *CVPR 2021.* arXiv:2003.07311.
14. Kervadec, H., Dolz, J., Tang, M., Granger, E., Boykov, Y., & Ben Ayed, I. (2018). Constrained-CNN losses for weakly supervised segmentation. *MIDL 2018 / Medical Image Analysis 2019.* arXiv:1805.04628. **[author-unverified]**
15. Hu, X., Fuxin, L., Samaras, D., & Chen, C. (2019). Topology-preserving deep image segmentation. *NeurIPS 2019.* arXiv:1906.05404.
16. Xu, M., Hu, X., Gupta, S., Abousamra, S., & Chen, C. (2024). Semi-supervised segmentation of histopathology images with noise-aware topological consistency (**TopoSemiSeg**). *ECCV 2024.* arXiv:2311.16447.
17. Pham, H.-H., Le, M., Huynh, H., Le, N. Q. K., & Pham, H.-H. (2026). Graph-theoretic consistency for robust and topology-aware semi-supervised histopathology segmentation (**TGC**). *AAAI 2026 Student Abstract & Poster Program.* arXiv:2509.22689. *(note: student-abstract venue — weaker prior-art weight.)*
18. Iqbal, J., Masood, A., Sultani, W., & Ali, M. (2023). Leveraging topology for domain adaptive road segmentation in satellite and aerial imagery. *ISPRS Journal of Photogrammetry and Remote Sensing.* arXiv:2309.15625.
19. Berger, A. H., Stucki, N., Lux, L., Buergin, V., Shit, S., Banaszak, A., Rueckert, D., Bauer, U., & Paetzold, J. C. (2024). Topologically faithful multi-class segmentation in medical images. *MICCAI 2024*, LNCS 15008, 721–731. arXiv:2403.11001.

**Long-tail / rare-class loss design**

20. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *ICCV 2017.* arXiv:1708.02002. *(canonical; standard cite)*
21. Tan, J., Wang, C., Li, B., Li, Q., Ouyang, W., Yin, C., & Yan, J. (2020). Equalization loss for long-tailed object recognition. *CVPR 2020.* arXiv:2003.05176.
22. Hsieh, T.-I., Robb, E., Chen, H.-T., & Huang, J.-B. (2021). DropLoss for long-tail instance segmentation. *AAAI 2021.* arXiv:2104.06402.

**Prototype / clustering machinery**

23. Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., & Joulin, A. (2020). Unsupervised learning of visual features by contrasting cluster assignments (**SwAV**; Sinkhorn-Knopp assignment). *NeurIPS 2020.* arXiv:2006.09882. *(canonical; standard cite)*

**Remote-sensing SSL/UDA competitors & benchmarks**

24. Lv, L., Wang, D., Zhang, J., & Zhang, L. (2025). S5: Scalable semi-supervised semantic segmentation in remote sensing. *AAAI 2026 (Oral).* arXiv:2508.12409.
25. MUCA: Semi-supervised semantic segmentation for remote sensing via multi-scale uncertainty consistency and cross-teacher-student attention (2025). *TGRS.* arXiv:2501.10736. **[author-unverified]**
26. RS-MTDF: Multi-teacher distillation for semi-supervised remote-sensing segmentation (2025). arXiv:2506.08772. **[author-unverified]**
27. DWL: Decouple and weight semi-supervised semantic segmentation of remote sensing images (2024). *ISPRS J. Photogrammetry & Remote Sensing.* doi:10.1016/j.isprsjprs.2024.04.020. **[author-unverified]**

*Excluded as unverifiable:* several 26xx-series arXiv IDs surfaced during search could not be confirmed and are deliberately omitted (see §10.5).

---

## 12. Provenance & honesty notes

- Witness A's core (prototype-likelihood hard flip) is **ProDA's**; credited as such throughout. The DWPC contribution is the **dual-witness fusion + rare-class-gated structural witness**, not the flip.
- Novelty verdicts in §3.4 are literature-search results (negatives are not proofs of absence); §10.5 verification pass required before publication.
- Some search-surfaced arXiv IDs in upstream notes were UNVERIFIED and excluded from §11.
