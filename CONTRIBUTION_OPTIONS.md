# DAPCN-SSL — Paper Contribution Options

**Status:** Decision document
**Date:** 2026-05-28
**Author:** L. Vo
**Context:** After devil's-advocate review of DRA-PLC v2 ([DRA_PLC_PROPOSAL.md](DRA_PLC_PROPOSAL.md)), user judged the proposal as recombination. This document lays out four ranked options for the paper's headline contribution, with verified prior-art audits and decision criteria.

---

## 0. Executive Summary

The "pseudo-label weighting" axis is **saturated**. Every component of DRA-PLC v2 has 1–3 direct prior-art papers (see §1). To produce a top-tier (AAAI / CVPR / NeurIPS) contribution rather than a tier-2 (TGRS / ISPRS) systems paper, the headline must be a **structurally new operation on pseudo-labels**, not a new reliability score.

Three structural options are presented (**P1, P2, P3**) plus the incremental fallback (**v2**). Recommendation: **P1 — pseudo-label inversion via prototype-density likelihood ratio**.

---

## 1. Saturation Map (Verified 2026-05-28)

| Component / direction | Direct prior art | Status |
|---|---|---|
| Dense per-pixel $w_i$ in RS-SSL | DWL (ISPRS J. PRS 2024), MUCA (TGRS 2025) | **done** |
| Class-adaptive $\tau_c$ | FreeMatch (CVPR 2023), FlexMatch | **done** |
| Adaptive $\alpha$ for prototype correction | partial — fixed $\alpha$ in ProSFDA; adaptive in UCGM | **weakened** |
| Multi-prototype Gaussian-mixture for RS-SSL | RSProtoSemiSeg (ISPRS J. PRS 2025, doi 10.1016/j.isprsjprs.2025.06.027); ProtoGMM (arXiv 2406.19225) | **done** |
| Sinkhorn-balanced prototype-to-class assignment | SwAV, SeLa, PiCO | **done** |
| Confidence-variance pseudo-label selection | When Confidence Fails (**ICCV 2025**, arXiv 2509.16704) | **done** |
| Energy-based pseudo-labeling for imbalance | EnergyMatch (arXiv 2206.06359), InPL (arXiv 2303.07269), arXiv 2501.01640 | **done** |
| Causal pseudo-label refinement (style confounder) | SDCL — Style Deconfounded Causal Learning (arXiv 2503.16852) | partial — style only |
| Mask-perturbation / counterfactual reliability | Trusted Mask Perturbation; Feedback-Driven Reliability (arXiv 2505.07691) | **done** |
| Open-world novel-class discovery in SSL | OpenLDN (arXiv 2207.02261); Open-world Semi-supervised Novel Class Discovery (2023) | **done for novel-class; open for known-rare** |
| Label propagation on prototype graph | CorrMatch (CVPR 2024); SPARK (arXiv 2602.00516) | **done** |

**Implication:** the only mechanism in v2 that is not directly killed is **adaptive $\alpha_i$ tied to teacher-uncertainty × prototype-reliability**, which is too small to carry a top-tier paper alone.

---

## 2. Option v2 — Incremental: DRA-PLC (Dense Reliability-Adaptive Pseudo-Label Correction)

See full proposal in [DRA_PLC_PROPOSAL.md](DRA_PLC_PROPOSAL.md). Summary:

- **Headline mechanism:** $\alpha_i = \alpha_{\max}\cdot(1-r^{\rm conf}_i)\cdot r^{\rm proto}_i$, capped per-pixel adaptive blend between EMA-teacher and prototype-projected pseudo-labels.
- **Supporting machinery:** class-adaptive $\tau_c$ (FreeMatch port), tri-source dense $w_i$ (weighted geometric mean), Sinkhorn-balanced multi-prototype class binding, pre-mix EMA fix.
- **Pros:** ~225 LoC, builds entirely on existing DAPCN-SSL machinery, ablation matrix already preregistered, fixes three of the four "weak spots" already in [project_weak_spots](memory).
- **Cons:** verdict from devil's-advocate review = **recombination**. Every supporting component has direct prior art; only $\alpha_i$ is materially new.
- **Venue fit:** TGRS / Remote Sensing / ISPRS J. PRS / IGARSS. **Not** a viable top-CV submission.
- **Verdict:** Worth implementing as the *system upgrade* anyway because all five components fix real DAPCN-SSL weak spots — but it is not the paper's headline.

---

## 3. Option P1 — Pseudo-Label INVERSION (rare-class rediscovery via prototype-density likelihood ratio)

### 3.1 Reframe

DAPCN-SSL's two pain points — **rare class** and **EMA-teacher pseudo-label noise on the target / unlabeled stream** — are the *same phenomenon*: pixels whose true class is rare get confidently mislabeled as a head class by the EMA teacher. Reweighting downgrades these pixels; what is actually needed is **recovery**.

No existing RS-SSL paper exploits this collapse. RSProtoSemiSeg uses GMM prototypes to *regularize* representations and *weight* pseudo-labels — it does not **flip** them.

### 3.2 New operation: pseudo-label flipping under a likelihood-ratio test

For each class $c$ build a multi-prototype mixture (re-using DAPCN's DynamicAnchor bank):
$$p(f \mid c) = \sum_{k:\kappa_k=c} \pi_k \cdot \mathcal{N}(f;\, \mu_k,\, \Sigma_c)$$

$\kappa_k$ is the soft class binding of prototype $k$ (Sinkhorn-Knopp, §4.5 of v2 retained).

For pixel $i$ with EMA-teacher pseudo-label $\hat y_i$, compute the **Bayesian log-likelihood ratio** to every other class, with a **rare-class prior boost**:
$$\Lambda_i(c) = \log \frac{p(f^S_i \mid c)}{p(f^S_i \mid \hat y_i)} \;+\; \log \frac{\pi^{\rm rare}_c}{\pi^{\rm rare}_{\hat y_i}}$$

where $\pi^{\rm rare}_c \propto (1 / N_c)^{\eta}$ is an inverse-frequency prior over classes.

**Flip rule:**
$$
\tilde y_i =
\begin{cases}
\arg\max_c \Lambda_i(c) & \text{if } \max_c \Lambda_i(c) > \tau_{\rm flip} \;\text{and}\; \arg\max_c \Lambda_i(c) \in \mathcal{C}_{\rm rare} \\[2pt]
\hat y_i & \text{otherwise}
\end{cases}
$$

The constraint $\arg\max_c \Lambda_i(c) \in \mathcal{C}_{\rm rare}$ restricts flipping to **rare-class destinations only** (asymmetric — flips can only *find* rare classes, never *create* head classes from rare). This guards against catastrophic flipping.

**Budget cap (safety):**
$$\#\{i: \tilde y_i \neq \hat y_i\} \;\leq\; \beta_{\rm flip} \cdot |U|$$

with $\beta_{\rm flip} \in [0.5\%, 5\%]$ — limits total flips per batch, preventing runaway label corruption.

### 3.3 Loss

Standard SSL CE on the **flipped** pseudo-label, with **per-pixel flip-confidence weight**:
$$w_i^{\rm flip} = \sigma\big(\beta_{\rm fc}(\Lambda_i(\tilde y_i) - \tau_{\rm flip})\big)$$

$$\mathcal L_{\rm ssl} = \frac{1}{|U|} \sum_i w_i^{\rm flip} \cdot \text{CE}(z^S_i,\, \tilde y_i)$$

Optionally combine with v2's adaptive $\alpha_i$ on the *non-flipped* pixels.

### 3.4 Distinction from prior art

| Method | Operation on pseudo-label | Rare-class mechanism |
|---|---|---|
| DWL, MUCA, U²PL | weight | none specific |
| ProSFDA | blend ($\alpha$) | none specific |
| FreeMatch | threshold | adaptive $\tau_c$ |
| RSProtoSemiSeg | weight + regularize | implicit via prototype distribution |
| **P1 (this)** | **FLIP (replace argmax)** | **Bayesian prior boost + asymmetric flip-to-rare-only** |

**Pseudo-label flipping** as an operation exists in *self-training* literature (e.g. CBST class-balanced self-training), but always as a hard re-prediction by the *same model*. P1 flips based on **prototype-density evidence**, which is a different evidential source from the model's own output. No exact precedent located in the 2024–2026 survey.

### 3.5 Hypotheses (preregistered falsifiers)

| # | Hypothesis | Pass criterion |
|---|---|---|
| H1 | Pseudo-label flipping improves rare-class IoU | Mean rare-IoU $\geq +3$ vs. DAPCN-SSL baseline (no flips); 3 seeds; $p<0.05$ |
| H2 | Flipping is not net-harmful to head classes | Head-class IoU regression $\leq 1.0$ |
| H3 | The flip-eligible pixels really are rare-class — verify on labeled validation | $\geq 60\%$ flip precision (flipped pixel's true label is the flip target) on labeled val |
| H4 | Asymmetric flip-to-rare-only constraint is necessary | Removing constraint (allow flips to any class) degrades mIoU $\geq 1$ |
| H5 | Beats RSProtoSemiSeg and S5 on rare-class IoU specifically | $\geq +1.5$ rare-class IoU vs. on-backbone reproductions |
| H6 | Super-additive vs. v2 weighting | $\Delta_{\rm flip+v2} > \Delta_{\rm flip} + \Delta_{\rm v2}$ |

### 3.6 Implementation cost

| Module | LoC | File |
|---|---:|---|
| Per-class GMM density $p(f \mid c)$ on DynamicAnchor bank | ~80 | new `mmseg/models/utils/prototype_gmm.py` |
| Likelihood-ratio test + flip decision | ~50 | `dapcn_ssl.py` (new method `_flip_pseudo_labels`) |
| Flip budget cap | ~20 | `dapcn_ssl.py` |
| Flip-precision diagnostic (validation only) | ~40 | `tools/flip_diagnostics.py` |
| Sinkhorn class binding (from v2) | ~80 | reused |
| **Total** | **~270 LoC** | |

### 3.7 Risks

1. **False flip propagation** — wrongly flipped pixels train the student toward the wrong class. Mitigation: low $\beta_{\rm flip}$ initial budget + flip-precision tracking on labeled val.
2. **Variance $\Sigma_c$ estimation** — per-class covariance is hard to estimate from finite prototypes. Mitigation: shared isotropic $\sigma^2_c I$, or learn $\sigma^2_c$ as a parameter.
3. **Cold-start** — early in training, prototypes are noisy → likelihood ratios unreliable. Mitigation: flipping only enabled after `proto_correction_start_iter` (existing warmup).
4. **Hidden prior art** — search may have missed a paper. Need a deeper second-pass survey before submission. Closest candidate to verify: CBST + prototypes (any 2024–2026 paper).

### 3.8 Verdict

**Strongly recommended.** Single structurally new operation, unifies the user's two problems, builds on existing DAPCN machinery, falsifiable in one ablation.

**Venue fit:** AAAI / WACV / ICCV / CVPR — viable if H1–H6 land.

---

## 4. Option P2 — Causal Pseudo-Label Adjustment (Class-Frequency Confounder)

### 4.1 Reframe

EMA-teacher confidence is **causally confounded** by training-set class frequency. The teacher gives high-confidence predictions to head classes not because of the pixel's content but because of the prior accumulated during training. This is a back-door path:

$$\text{class\_freq} \;\to\; \text{teacher\_weights}\;\theta^T \;\to\; \hat p_i$$

Existing causal-segmentation work (SDCL, arXiv 2503.16852) deconfounds **style**, not **class frequency**. The latter is unaddressed in segmentation literature.

### 4.2 New operation: front-door adjustment through prototypes

Structural causal model:
$$f \;\longrightarrow\; z \;\longrightarrow\; y$$

with $f$ = pixel feature, $z$ = soft prototype assignment, $y$ = pseudo-label. The front-door criterion gives:
$$P(y \mid \text{do}(f)) = \sum_z P(z \mid f)\, \sum_{f'} P(y \mid z,\, f')\, P(f')$$

Both terms are estimable from DAPCN-SSL state: $P(z \mid f)$ from prototype similarity softmax; $P(y \mid z, f')$ from the labeled-set classifier conditioned on $z$; $P(f')$ marginalized over the labeled-feature distribution (small finite sum).

Operationally, the pseudo-label becomes:
$$\tilde p_i \;=\; \mathbb{E}_{z \sim P(z \mid f_i)} \mathbb{E}_{f' \sim P(f')}\big[ P(y \mid z, f') \big]$$

— a doubly-marginalized estimator that, by front-door adjustment, removes the class-frequency back-door.

### 4.3 Distinction

| Method | Confounder addressed | Mechanism |
|---|---|---|
| SDCL (arXiv 2503.16852) | style | back-door stratification on style |
| Causal Inference Fundus Seg (MDPI 2025) | domain shift | self-supervised causal cross-domain |
| **P2 (this)** | **training-set class frequency** | **front-door through prototype mediator** |

### 4.4 Hypotheses

- H1: P2 reduces head-class confidence on rare-class pixels (calibration on labeled val).
- H2: Rare-class IoU improvement is concentrated on pixels where teacher prior is strongest (stratified by class-frequency-rank).
- H3: Beats DRA-PLC v2 by $\geq 1$ mIoU on OEM rare classes.

### 4.5 Risks

1. **Estimating $P(f')$** — labeled set is small; the marginal is high-dimensional. Mitigation: cluster labeled features into anchors, sum over anchors.
2. **Reviewer variance** — causal-inference papers receive bimodal reviews. Required: very clean SCM diagram + sensitivity analysis.
3. **Implementation complexity** — ~400 LoC, plus a careful derivation appendix. Heavier than P1.

### 4.6 Verdict

**Alternative.** Stronger theory, higher review variance, longer implementation. Venue fit: NeurIPS / ICLR (theory tracks) or CVPR with a strong appendix.

---

## 5. Option P3 — SAR-Optical Co-Validation of Pseudo-Labels

### 5.1 Reframe

OpenEarthMap / LoveDA are RGB-only. The same geographic regions have freely-available Sentinel-1 SAR. Use SAR as an **independent evidential source** for pseudo-label validation.

### 5.2 New operation: cross-modal prototype-consensus filtering

Two prototype banks, $\mu^{\rm opt}_k$ and $\mu^{\rm SAR}_k$, trained jointly. For pixel $i$:

$$\text{consensus}_i = \cos\!\big( P(z^{\rm opt} \mid f^{\rm opt}_i),\; P(z^{\rm SAR} \mid f^{\rm SAR}_i) \big)$$

Pseudo-label is admitted only if consensus exceeds a threshold *and* both modalities' top-class prototypes belong to the same class.

### 5.3 Distinction

Multimodal RS-SSL exists (MMAdapter, Cooperative-LoRA-SAM2 — ISPRS J. PRS 2026) but uses multimodal **fusion at the feature level**. P3 keeps modalities **separate** and uses cross-modal agreement as the reliability signal — closer to multi-view consistency in classical SSL but with prototype banks as the agreement basis.

### 5.4 Hypotheses

- H1: SAR-consensus admission produces higher pseudo-label precision than RGB-only.
- H2: Rare classes that are SAR-visible (water, agricultural land) benefit most.
- H3: Beats RGB-only baselines by $\geq 2$ mIoU.

### 5.5 Risks

1. **Dataset bandwidth** — Sentinel-1/-2 alignment with OEM tiles is non-trivial. ~2 weeks of dataset engineering before any method experiment.
2. **Modality gap** — SAR prototypes are hard to initialize; may need pretraining.
3. **Limited transfer to LoveDA / Potsdam** — Potsdam/Vaihingen don't have aligned SAR. P3 is OEM-specific.

### 5.6 Verdict

**Conditional.** Strong novelty *if* the dataset engineering is feasible. Venue fit: TGRS / ISPRS / WACV.

---

## 6. Decision Matrix

| Criterion | v2 | **P1** | P2 | P3 |
|---|:-:|:-:|:-:|:-:|
| Novelty (structural) | Low | **High** | High | High |
| Builds on existing code | Full | **Full** | Partial | Partial |
| Implementation cost (LoC) | ~225 | **~270** | ~400 | ~600 + dataset |
| Risk of hidden prior art | Low (verified) | **Medium** (needs deeper search) | Medium | Low |
| Falsifiability | Strong | **Strong** | Medium | Strong |
| Venue ceiling | TGRS/ISPRS | **AAAI/CVPR/ICCV** | NeurIPS/ICLR | TGRS/WACV |
| Time to MVE | 1 week | **1.5 weeks** | 3 weeks | 4–5 weeks |
| Unifies rare-class + noise problems | No | **Yes** | Yes | Partial |

---

## 7. Recommended Path

**Primary:** Commit to **P1 (pseudo-label inversion)** as the headline contribution.

**Secondary:** Implement v2's adaptive $\alpha_i$, Sinkhorn class-binding, and tri-source $w_i$ as the supporting machinery on non-flipped pixels. v2 becomes the *secondary contribution* — sound engineering, ablated cleanly.

**Compound paper structure:**
- **Headline:** "We do not weight noisy pseudo-labels; we flip them. Likelihood-ratio test on prototype mixtures rediscovers rare-class pixels mislabeled by the EMA teacher."
- **Supporting:** dense reliability + adaptive blend for the residual non-flipped pixels.

**Pre-commit risk-reduction (do before full implementation):**
1. Deeper second-pass survey: search for "class-balanced self-training prototypes 2024-2026", "pseudo-label flipping segmentation", and "CBST+prototype" combinations. Ensure no killer prior art was missed.
2. Verify RSProtoSemiSeg does **not** flip pseudo-labels (read full method section, not just abstract).
3. Sketch the minimal-viable experiment: prototype-density estimate + flip rule on 1 OEM seed, no other v2 machinery. If rare-class IoU moves on this minimal setup, the headline is real.

---

## 8. Next Actions

- [ ] Decision: confirm P1 (or alternative) as headline.
- [ ] Deeper prior-art search on "pseudo-label flipping" + "prototype-density CBST".
- [ ] Read RSProtoSemiSeg full paper to verify it does not flip.
- [ ] Implement P1 MVE (prototype GMM + flip rule only, no other v2 components) on OEM 1/5/10%.
- [ ] If MVE passes, layer v2 supporting machinery on non-flipped pixels.
- [ ] If MVE fails, return here and choose P2 or accept v2 at tier-2 venue.

---

## 9. References (verified 2026-05-28)

| Method | ID / Venue |
|---|---|
| DWL — Decouple and Weight Learning | ISPRS J. PRS 2024 |
| RSProtoSemiSeg | [doi 10.1016/j.isprsjprs.2025.06.027](https://www.sciencedirect.com/science/article/abs/pii/S0924271625003107), ISPRS J. PRS 2025 |
| ProtoGMM | [arXiv 2406.19225](https://arxiv.org/abs/2406.19225), 2024 |
| UCGM | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S095741742502737X), 2025 |
| ProSFDA | [arXiv 2509.16942](https://arxiv.org/abs/2509.16942), 2025 |
| PDSSNet | [arXiv 2508.04022](https://arxiv.org/abs/2508.04022), 2025 |
| S5 (AAAI 2026 Oral) | [arXiv 2508.12409](https://arxiv.org/abs/2508.12409) |
| MUCA | [arXiv 2501.10736](https://arxiv.org/abs/2501.10736), TGRS 2025 |
| When Confidence Fails | [arXiv 2509.16704](https://arxiv.org/abs/2509.16704), ICCV 2025 |
| Feedback-Driven Reliability | [arXiv 2505.07691](https://arxiv.org/abs/2505.07691), 2025 |
| FreeMatch | CVPR 2023 |
| FlexMatch | NeurIPS 2021 |
| U²PL | CVPR 2022 |
| EnergyMatch | [arXiv 2206.06359](https://arxiv.org/abs/2206.06359) |
| InPL | [arXiv 2303.07269](https://arxiv.org/abs/2303.07269) |
| Uncertainty + Energy SSL Seg | [arXiv 2501.01640](https://arxiv.org/abs/2501.01640) |
| SDCL — Style Deconfounded Causal Learning | [arXiv 2503.16852](https://arxiv.org/abs/2503.16852) |
| OpenLDN | [arXiv 2207.02261](https://arxiv.org/abs/2207.02261) |
| CorrMatch | CVPR 2024 |
| SwAV | NeurIPS 2020 |
| SeLa | ICLR 2020 |
| PiCO | ICLR 2022 |
| CBST — Class-Balanced Self-Training | ECCV 2018 |
| MILE (Mixture of Incremental LoRA Experts) | [arXiv 2605.03555](https://arxiv.org/abs/2605.03555), ICPR 2026 |
| DToP — Dynamic Token Pruning | [arXiv 2308.01045](https://arxiv.org/abs/2308.01045), ICCV 2023 |
| Multi-Faceted Adaptive Token Pruning (RS) | [MDPI Remote Sensing 17(14):2508, 2025](https://www.mdpi.com/2072-4292/17/14/2508) |
| Cooperative LoRA + SAM2 (RS) | ISPRS J. PRS 2026 |
