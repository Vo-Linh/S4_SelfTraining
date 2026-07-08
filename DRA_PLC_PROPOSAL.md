# DRA-PLC — Dense Reliability-Adaptive Pseudo-Label Correction for Rare-Class RS-SSL

**Status:** Proposal v2 (revised after devil's-advocate review)
**Date:** 2026-05-28
**Author:** L. Vo
**Framework:** DAPCN-SSL (this repository)
**Target venues:** AAAI / CVPR / TGRS / ISPRS J. PRS 2026-2027

---

## 1. Problem Statement

DAPCN-SSL targets semi-supervised semantic segmentation of aerial / satellite imagery. Two failure modes dominate evaluation:

1. **Rare class collapse** — the EMA-teacher branch produces high-confidence predictions for head classes (impervious, vegetation) and low-confidence predictions for rare classes (water, agricultural land in OpenEarthMap). The global `pseudo_threshold = 0.968` silently drops rare pixels before they ever enter the loss, partly undoing Rare Class Sampling.
2. **Pseudo-label noise in target / unlabeled domain** — the scalar `pseudo_weight` (`dapcn_ssl.py:831–833`) reduces a pixel-grained reliability question to one number per image. Pixels near class boundaries, in mixed regions, or at the prototype periphery are weighted identically to dead-center high-confidence pixels.

DAPCN-SSL already contains the machinery to attack both — DynamicAnchor prototype bank, MLP quality gate, affinity boundary head, EMA teacher — but they are fused through three coarse scalars (`pseudo_weight`, `pseudo_threshold`, `proto_correction_alpha = 0.5`). DRA-PLC replaces all three with a single dense field.

---

## 2. Literature Triage (2026-05 Survey)

| Axis | Verdict | Key prior art |
|---|---|---|
| Pixel-importance / dense pseudo-label weighting for RS-SSL | **CROWDED** | DWL (ISPRS 2024) — confidence-rank pixel weighting; MUCA (TGRS 2025) — multi-scale uncertainty; "When Confidence Fails" (arXiv 2509.16704) — convex pseudo-label selection beyond confidence |
| LoRA / PEFT for RS segmentation, spatial gating | CROWDED for FM adaptation; **GAP for spatial gating in SSL** | MMLoRA, Cooperative LoRA + SAM2, NAS-LoRA, MILE (per-task prototype-gated experts — natural images, continual learning) |
| Token sparsity in dense prediction for RS | **SATURATED** | DToP (ICCV 2023); Multi-Faceted Adaptive Token Pruning (RS 2025) |
| Direct DAPCN-SSL rivals | n/a | S5 (AAAI 2026 Oral), PDSSNet, ProSFDA (prototype-weighted pseudo-labels for SFDA), DGLE, MUCA, RS-MTDF |

**Conclusion:** Replacing scalar `pseudo_weight` with a dense map (option A from the original ideation) is **already done** for RS-SSL by DWL and for RS-SFDA by ProSFDA. Spatial LoRA gating in RS-SSL is an unclaimed gap but the motivation is fragile and a probable trap. The defensible thread is **integrating dense reliability with DAPCN's prototype quality gate, affinity boundary head, and class-conditioned prototypes**, with **per-pixel adaptive prototype-correction strength α_i** as the load-bearing novelty.

---

## 3. Notation

| Symbol | Meaning |
|---|---|
| $f^T, f^S$ | EMA teacher / student segmentor |
| $\hat{p}_i \in \Delta^C$ | teacher softmax at pixel $i$, $C$ classes |
| $\hat{y}_i = \arg\max_c \hat{p}_{i,c}$ | hard pseudo-label |
| $\hat{c}_i = \max_c \hat{p}_{i,c}$ | teacher confidence |
| $\{\mu_k\}_{k=1}^K$, $q_k$ | DynamicAnchor prototypes + MLP quality scores |
| $s_{i,k} = \cos(f^S_i, \mu_k)$ | per-pixel prototype similarity |
| $b_i \in [0,1]$ | affinity boundary score (1 = on boundary) |
| $\bar{p}_c^{(t)}, A_c^{(t)}$ | EMA-tracked per-class teacher confidence / admitted mass |
| $\bar\mu_c^{\rm mem}$ | EMA class centroid from `PrototypeMemory` |
| $g(\cdot)$ | existing `proto_to_decoder` projection |
| $\mathbb{P}[\kappa(\mu_k)=c]$ | soft class binding of prototype $k$ to class $c$ |

---

## 4. Method: DRA-PLC v2

### 4.1 Component 1 — Class-adaptive threshold (rare-class admission)

FreeMatch-style EMA tracking, computed on the **pre-ClassMix** teacher prediction (fixes the EMA-contamination bug found in review):

$$
\bar{p}_c^{(t)} = m\,\bar{p}_c^{(t-1)} + (1-m)\cdot \mathbb{E}_{i:\hat{y}_i=c}\!\left[\hat{c}_i\right]
$$

$$
\tau_c^{(t)} = \tau_{\rm base}\cdot\left(\frac{\bar{p}_c^{(t)}}{\max_{c'} \bar{p}_{c'}^{(t)}}\right)^{\!\gamma_\tau}
$$

Replaces the global `pseudo_threshold = 0.968` (`dapcn_ssl.py:830`). Rare classes drop their threshold automatically.

### 4.2 Component 2 — Tri-source dense reliability $w_i$

**Confidence reliability:**

$$
r^{\rm conf}_i = \sigma\!\big(\beta\,(\hat{c}_i - \tau_{\hat{y}_i}^{(t)})\big)
$$

**Soft prototype agreement** — replaces hard $\arg\max + \mathbb{1}[\cdot]$ from v1 (which the reviewer correctly identified as collapsing $w_i$ to a prototype-only gate):

$$
\pi_{i,k} = \frac{\exp(s_{i,k}/T_s)}{\sum_{k'}\exp(s_{i,k'}/T_s)},\qquad
\rho_i = \sum_k \pi_{i,k}\,\mathbb{P}[\kappa(\mu_k)=\hat{y}_i]
$$

$$
r^{\rm proto}_i = \Big(\!\sum_k \pi_{i,k}\,q_k\Big) \cdot \rho_i
$$

The soft class-binding distribution $\mathbb{P}[\kappa(\mu_k)=c]$ comes from Sinkhorn-Knopp balanced assignment (§4.5).

**Boundary deflation:**

$$
r^{\rm bound}_i = 1 - \sigma(b_i / T_b)
$$

**Combined dense reliability** — weighted geometric mean (replaces hard multiplicative AND):

$$
\boxed{\;w_i = \big(r^{\rm conf}_i\big)^{a_1}\big(r^{\rm proto}_i\big)^{a_2}\big(r^{\rm bound}_i\big)^{a_3},\quad \sum_j a_j = 1\;}
$$

Default $a_1 = a_2 = a_3 = 1/3$; optionally learn via $a_j = \text{softmax}(\theta_j)$ with three scalar parameters. Ablation rows in §7 hold $a_j = 0$ to isolate each source.

**Warmup for prototype branch** (fixes the chicken-and-egg bug):

$$
r^{\rm proto}_i \leftarrow (1 - \omega(t)) \cdot 1 + \omega(t) \cdot r^{\rm proto}_i,\qquad
\omega(t) = \min\!\Big(\tfrac{t - t_{\rm warm}}{\Delta_{\rm warm}},\,1\Big)_{+}
$$

For $t < t_{\rm warm}$ the prototype source is neutralized ($r^{\rm proto} \equiv 1$) so only confidence + boundary gate the loss. $t_{\rm warm}$ aligns with the existing `proto_correction_start_iter`.

### 4.3 Component 3 — Adaptive $\alpha_i$ correction strength

**The load-bearing novelty.** Replaces fixed `proto_correction_alpha = 0.5` (`dapcn_ssl.py:491–492`):

$$
\boxed{\;\alpha_i = \alpha_{\max}\cdot(1 - r^{\rm conf}_i)\cdot r^{\rm proto}_i,\qquad \alpha_{\max} \in [0.3,\,0.5]\;}
$$

**Reading:** trust the prototype-projected distribution **only when** (i) the teacher is uncertain (low $r^{\rm conf}$) **and** (ii) the prototype is itself reliable (high $r^{\rm proto}$). Hard cap $\alpha_{\max} < 1$ prevents the ProSFDA-known failure mode of fully replacing the teacher's distribution.

Corrected pseudo-label distribution:

$$
\tilde{p}_i = (1 - \alpha_i)\,\hat{p}_i + \alpha_i\cdot f_\theta(\text{PT})\cdot a_i
$$

where $a_i = \text{softmax}(s_{i,\cdot}/T_s)$ is the prototype-assignment vector and $f_\theta(\text{PT})$ projects prototypes through `conv_seg`.

### 4.4 Component 4 — Class-rebalanced loss weight (optional, on-demand)

Two preregistered options — commit to one before running:

**Option L1 (default):** drop $\lambda_c$ entirely. $\tau_c$ admits rare pixels; let the gradient speak. Removes a hyperparameter and avoids triple-counting.

**Option L2 (if L1 underperforms):** post-admission coupled reweighting:

$$
A_c^{(t)} = m\,A_c^{(t-1)} + (1-m)\!\sum_{i:\hat{y}_i=c}\!w_i\cdot\mathbb{1}[\hat{c}_i > \tau_c^{(t)}]
$$

$$
\lambda_c^{(t)} = \big(\bar{A}^{(t)} / A_c^{(t)}\big)^{\gamma_\lambda},\qquad \bar{A}^{(t)} = \tfrac{1}{C}\sum_c A_c^{(t)}
$$

$\lambda_c$ kicks in **only** if $\tau_c$ + $w_i$ failed to admit enough rare mass — no cancellation.

### 4.5 Component 5 — Class-conditioned multi-prototype binding (Sinkhorn-Knopp)

Auxiliary loss enforcing **balanced** class binding (each class gets at least $\lceil K/C \rceil$ prototypes — addresses multi-modal RS classes such as "building" = urban + rural sheds):

$$
\min_{Q}\;\sum_{k,c} Q_{k,c}\,\|g(\mu_k) - \bar\mu_c^{\rm mem}\|_2^2 - \epsilon H(Q)
$$

subject to $Q\mathbf{1} = \tfrac{1}{K}\mathbf{1}$, $Q^\top\mathbf{1} = \tfrac{1}{C}\mathbf{1}$, with $Q_{k,c} = \mathbb{P}[\kappa(\mu_k)=c]$.

Solved by 3-iter Sinkhorn on log-domain costs (~20 LoC, SwAV/SeLa machinery). The soft assignment $Q$ feeds back into $\rho_i$ in §4.2.

Auxiliary anchor loss:

$$
\mathcal{L}_{\rm anchor} = \sum_{k,c} Q_{k,c}\,\big\|\text{sg}[\bar\mu_c^{\rm mem}] - g(\mu_k)\big\|_2^2
$$

`sg[·]` = stop-gradient on the class memory (memory updates by EMA outside backprop).

### 4.6 Final SSL loss

$$
\boxed{\;\mathcal{L}_{\rm ssl} = \frac{1}{|U|}\sum_{i \in U} w_i\cdot\lambda_{\hat{y}_i}\cdot\text{CE}(z^S_i,\,\tilde{y}_i)\;}
$$

with $\tilde{y}_i = \arg\max_c \tilde{p}_{i,c}$ (hard) or KL-to-soft target $\text{KL}(\text{softmax}(z^S_i)\,\|\,\tilde{p}_i)$ (variant).

**Full objective:**

$$
\mathcal{L} = \mathcal{L}_{\rm sup} + \lambda_{\rm ssl}\,\mathcal{L}_{\rm ssl} + \lambda_{\rm dapg}\,\mathcal{L}_{\rm DAPG} + \lambda_{\rm anchor}\,\mathcal{L}_{\rm anchor} + \lambda_{\rm bnd}\,\mathcal{L}_{\rm affinity}
$$

---

## 5. Mapping to DAPCN-SSL Weak Spots

| Weak spot (project memory) | Fixed by | File:line touched |
|---|---|---|
| Scalar `pseudo_weight` | §4.2 ($w_i$) | `dapcn_ssl.py:831–833` |
| Fixed `proto_correction_alpha = 0.5` | §4.3 ($\alpha_i$) | `dapcn_ssl.py:491–492` |
| Global `pseudo_threshold = 0.968` | §4.1 ($\tau_c$) | `dapcn_ssl.py:830` |
| Class-agnostic prototypes | §4.5 (Sinkhorn) | `dynamic_anchor.py`, `prototype_memory.py` |
| Multi-modal class appearance | §4.5 (multi-proto per class) | same |
| ClassMix / EMA contamination | §4.1 (pre-mix EMA) | `dapcn_ssl.py:_get_pseudo_weight_scale` + step 3 |

---

## 6. Hypotheses (Preregistered Falsifiers)

| # | Hypothesis | Pass / fail criterion (committed before any experiment) |
|---|---|---|
| **H1** | Dense $w_i$ > scalar `pseudo_weight` | $\Delta$mIoU > 0.3 on OEM val, 3-seed mean |
| **H2** | Adaptive $\alpha_i$ helps high-entropy near-boundary pixels specifically | Stratified mIoU on (entropy quartile $Q_4$ ∩ boundary mask) $\Delta > 0.5$ |
| **H3** | Class binding (Sinkhorn) is **necessary** for $r^{\rm proto}$ | Replace $\rho_i \to 1$; if mIoU drops < 0.5, H3 is dead |
| **H4** | Rare-class IoU improves | Rare set = bottom-3 classes by training pixel frequency, **fixed before experiment**. Pass = mean rare IoU $\geq +2$ AND head IoU regression $\leq 1$ AND $p < 0.05$, 3 seeds |
| **H5** | $\tau_c$ + class-bound prototypes > $\tau_c$ alone | Beat FreeMatch-on-backbone by $\geq 1.0$ mIoU |
| **H6** | **Beats published RS-SSL SOTA** | OEM or LoveDA at 1 / 5 / 10% labels: beat MUCA (TGRS 2025), RS-MTDF; within 2 mIoU of S5 (foundation-model advantage acknowledged) |
| **H7** | **Super-additive interaction** | $\Delta_{\rm full} > 0.8 \cdot (\Delta_w + \Delta_\alpha + \Delta_\tau + \Delta_{\rm proto})$. If linear-additive, paper is glue → reframe |

H3, H6, H7 are the **decisive** hypotheses. If any of them fail, the paper needs restructuring before submission.

---

## 7. Ablation Matrix (Preregistered)

| Row | $\tau_c$ | $w_i$ | $\alpha_i$ | Class-bound proto | $\lambda_c$ | Purpose |
|---|:-:|:-:|:-:|:-:|:-:|---|
| A0 | global | scalar | 0.5 | ✗ | ✗ | DAPCN-SSL baseline |
| A1 | adaptive | scalar | 0.5 | ✗ | ✗ | FreeMatch contribution |
| A2 | global | tri-source | 0.5 | ✓ | ✗ | DWL + ProSFDA-style |
| **A3** | **global** | **scalar** | **adaptive** | **✓** | **✗** | **Isolates $\alpha_i$ novelty (decisive)** |
| A4 | adaptive | tri-source | adaptive | ✓ | ✗ | **Full DRA-PLC (L1)** |
| A5 | adaptive | tri-source | adaptive | ✓ | coupled L2 | DRA-PLC + class reweight |
| A6 | adaptive | tri-source | adaptive | hard binding | ✗ | Tests Sinkhorn vs. hard $\arg\min$ |
| A7 | adaptive | tri-source ($a_3 = 0$) | adaptive | ✓ | ✗ | Tests boundary contribution |
| A8 | adaptive | tri-source ($a_1 = 0$) | adaptive | ✓ | ✗ | Tests confidence contribution |
| A9 | adaptive | tri-source ($a_2 = 0$) | adaptive | ✓ | ✗ | Tests prototype contribution |

A3 is the load-bearing experiment — if it doesn't clearly beat A0, the headline contribution doesn't hold.

---

## 8. Required Baselines (On Same Backbone)

Reproduce these on **identical MiT-B5 / OpenEarthMap / LoveDA setup** as DAPCN-SSL:

1. **FreeMatch + ClassMix** — no prototypes; isolates contribution of DAPCN's prototype machinery.
2. **FreeMatch + ProSFDA-style prototype reweighting (fixed $\alpha$)** — isolates contribution of *adaptive* $\alpha_i$ specifically.
3. **U²PL + ClassMix** — pixel-wise reliability without prototypes or class-adaptive $\tau$.
4. **DWL** (if code available) — pixel-wise confidence-rank weighting on the same backbone.

Without these on-backbone reproductions the contribution cannot be defended in review.

---

## 9. Differentiation Table

| Method | Dense $w_i$ | Class-adaptive $\tau_c$ | Adaptive $\alpha_i$ | Class-bound prototypes | Boundary signal in $w$ | RS-SSL |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| DWL (ISPRS 2024) | ✓ (rank) | ✗ | n/a | ✗ | ✗ | ✓ |
| ProSFDA (arXiv 2509.16942) | ✓ (proto) | ✗ | ✗ | partial | ✗ | SFDA |
| FreeMatch (CVPR 2023) | ✗ | ✓ | n/a | ✗ | ✗ | ✗ |
| U²PL (CVPR 2022) | ✓ (uncertainty) | ✗ | n/a | ✗ | ✗ | ✗ |
| MUCA (TGRS 2025) | ✓ (uncertainty) | ✗ | n/a | ✗ | ✗ | ✓ |
| S5 (AAAI 2026 Oral) | ✗ (entropy filter at data curation) | ✗ | n/a | ✗ | ✗ | ✓ (foundation models) |
| **DRA-PLC (this work)** | **✓** | **✓** | **✓** | **✓ (Sinkhorn)** | **✓** | **✓** |

**Novelty claim (must be empirically defended via H7):** the *adaptive* $\alpha_i$ tied to teacher uncertainty × prototype reliability is, per survey, without prior art in RS-SSL or general segmentation. Everything else is integration.

---

## 10. Implementation Cost

| Component | LoC | File |
|---|---:|---|
| Per-class $\tau_c$ EMA (pre-mix) | ~40 | `dapcn_ssl.py` |
| Sinkhorn-Knopp class binding | ~80 | new `mmseg/models/utils/sinkhorn.py` |
| Soft $\rho_i$ + geometric-mean $w_i$ | ~60 | new `mmseg/models/utils/reliability.py` |
| Adaptive $\alpha_i$ in `_correct_pseudo_labels` | ~10 | `dapcn_ssl.py` |
| Warmup schedule $\omega(t)$ | ~5 | `dapcn_ssl.py` |
| Per-class anchor / Sinkhorn loss | ~30 | `dapcn_ssl.py`, `dynamic_anchor.py` |
| Stratified mIoU evaluation (H2/H4) | ~50 | new `tools/stratified_eval.py` |
| **Total** | **~275 LoC** | |

Estimated **one week** of focused implementation + **one week** for the ablation matrix (10 rows × 3 seeds = 30 runs).

---

## 11. Minimum Viable Experiment (MVE)

Before committing to v2 in full, run only row **A3** of the ablation:

> scalar $w_i$ (baseline `pseudo_weight`), global $\tau_c$, but **adaptive $\alpha_i$** + class-bound prototypes (Sinkhorn).

Implementation cost: ~120 LoC. If A3 does **not** beat A0 (DAPCN-SSL baseline) on OEM val by a clear margin in 3-seed mean, the load-bearing claim of the paper is unsupported and v2 needs to be restructured before further work.

---

## 12. Open Risks (Acknowledged)

- **Foundation-model gap** — S5 (AAAI 2026 Oral) uses RS4P-1M pretraining; a from-scratch MiT-B5 cannot beat it head-to-head. Position DRA-PLC as a *method* contribution, orthogonal to pretraining scale.
- **Cold-start of $\kappa$** — even with the warmup schedule, class binding for very rare classes may stay noisy through the full schedule. Mitigation: $\lceil K/C \rceil \geq 2$ in Sinkhorn, hard-floor on class assignment mass.
- **OEM rare-class definition** — OEM's 8-class taxonomy means "bottom-3" is a small set; results may not transfer to higher-cardinality benchmarks (e.g. LoveDA's 7 classes are already balanced-ish). Cross-validate on both.

---

## 13. References (Verified)

| Method | ID / Venue |
|---|---|
| DWL — Decouple and Weight | ISPRS J. PRS 2024, [doi 10.1016/j.isprsjprs.2024.04.020](https://www.sciencedirect.com/science/article/pii/S0924271624001710) |
| S5 — Scalable SSL for RS | AAAI 2026 Oral, [arXiv 2508.12409](https://arxiv.org/abs/2508.12409) |
| ProSFDA | [arXiv 2509.16942](https://arxiv.org/abs/2509.16942), 2025 |
| PDSSNet | [arXiv 2508.04022](https://arxiv.org/abs/2508.04022), 2025 |
| DGLE | [arXiv 2509.18502](https://arxiv.org/abs/2509.18502), 2025 |
| MUCA | [arXiv 2501.10736](https://arxiv.org/abs/2501.10736), TGRS 2025 |
| FreeMatch | CVPR 2023 |
| U²PL | CVPR 2022 |
| ClassMix | WACV 2021 |
| SwAV (Sinkhorn-Knopp prototypes) | NeurIPS 2020 |
| SeLa (Sinkhorn for self-labeling) | ICLR 2020 |
| MILE — Mixture of Incremental LoRA Experts | ICPR 2026, [arXiv 2605.03555](https://arxiv.org/abs/2605.03555) |
| DToP — Dynamic Token Pruning | ICCV 2023, [arXiv 2308.01045](https://arxiv.org/abs/2308.01045) |
| Multi-Faceted Adaptive Token Pruning for RS | Remote Sensing 17(14):2508, 2025, [MDPI](https://www.mdpi.com/2072-4292/17/14/2508) |

---

## 14. Devil's-Advocate Review Summary

Reviewer verdict on DRA-PLC v1: **MAJOR REVISION**. Accepted the core novelty ($\alpha_i$) and demolished surrounding glue.

Bugs found in v1 and fixed in v2 above:

1. Hard multiplicative AND collapsed $w_i$ to a prototype-only gate via the indicator function → soft Sinkhorn class binding + weighted geometric mean.
2. $\kappa(\mu_k)$ chicken-and-egg with class memory → explicit warmup $\omega(t)$.
3. $\alpha_i \to 1$ pathological regime → hard cap $\alpha_{\max} < 1$.
4. Triple-counting of rare-class signal ($\tau_c + \lambda_c +$ prototype quality) → drop $\lambda_c$ (L1) or post-admission coupling (L2).
5. ClassMix contamination of $\bar p_c$ EMA → update on pre-mix image only.
6. Single prototype per class fails on multi-modal RS classes → Sinkhorn-Knopp balanced multi-prototype binding.
7. H4 cherry-picked; H3 not falsifiable; H6/H7 missing → preregistered falsifiers committed.

The reviewer's strongest demand: the **super-additive hypothesis H7**. If full DRA-PLC's gain is merely the sum of single-component gains, the paper is glue and reviewer rejection is correct.
