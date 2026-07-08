# ---------------------------------------------------------------
# DWPC Witness A — Appearance (prototype-density likelihood ratio)
#
# The appearance witness models each class as an isotropic Gaussian
# mixture over the class-conditioned prototypes held in a
# ``PrototypeMemory`` bank.  For an unlabeled pixel it computes a
# rare-prior-boosted log-likelihood ratio of the best alternative
# (rare) class against the teacher's predicted class, and turns it
# into a rare-evidence score a_i in [0, 1] plus the best candidate
# class c*_i.
#
# Lineage (see DWPC_PROPOSAL.md §4.2 / §3.1):
#   - p(f|c) + argmax replacement is ProDA [Zhang et al., 2021].
#   - multi-prototype GMM density follows ProtoGMM [Moradinasab 2024].
#   - the *ratio*, the rare-class prior, the asymmetric flip-to-rare
#     restriction and the per-image budget (budget lives in the mixin)
#     are the DWPC delta.
#
# This is intentionally non-differentiable: Witness A only informs the
# flip decision and the per-pixel loss weight, neither of which carries
# a gradient through the witness itself.
# ---------------------------------------------------------------

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def witness_appearance(feat,
                       teacher_label,
                       proto_memory,
                       rare_class_ids,
                       beta=1.0,
                       sigma2=0.5,
                       rare_prior_logits=None,
                       symmetric=False,
                       standardize=True):
    """Appearance rare-evidence from a class-bound prototype GMM.

    Args:
        feat (Tensor): (B, D, Hf, Wf) features in ``proto_memory``'s
            space (NOT pre-normalised).
        teacher_label (Tensor): (B, Hf, Wf) long — teacher argmax at
            ``feat`` resolution.
        proto_memory (PrototypeMemory): class-conditioned prototype bank.
        rare_class_ids (list[int]): rare class indices C_rare.
        beta (float): logistic temperature on the (standardised) ratio.
        sigma2 (float): shared isotropic variance of the GMM components.
        rare_prior_logits (Tensor | None): (C,) precomputed log-prior
            ``log pi_c`` added to ``log p(f|c)``. ``None`` disables it
            (used by the A1 ProDA row).
        symmetric (bool): if True, candidates are *all* classes (ProDA
            reproduction, row A1). If False (default), candidates are
            restricted to ``rare_class_ids`` (asymmetric flip-to-rare).
        standardize (bool): robustly standardise the per-image ratio
            (subtract median, divide by MAD) before the sigmoid. The
            single most calibration-sensitive switch.

    Returns:
        tuple:
            a_map (Tensor): (B, Hf, Wf) in [0, 1] rare-evidence.
            cstar_map (Tensor): (B, Hf, Wf) long — best candidate class
                (-1 where no candidate exists).
    """
    B, D, Hf, Wf = feat.shape
    device = feat.device
    C = proto_memory.num_classes
    K = proto_memory.K

    # --- Cold-start guard: an empty bank cannot judge appearance. ---
    if not proto_memory.is_initialised():
        a_map = torch.zeros(B, Hf, Wf, device=device)
        cstar_map = torch.full((B, Hf, Wf), -1, dtype=torch.long,
                               device=device)
        return a_map, cstar_map

    N = B * Hf * Wf
    f = feat.permute(0, 2, 3, 1).reshape(N, D)
    f = torch.nan_to_num(F.normalize(f, dim=1), nan=0.0)

    protos = proto_memory.get_all_normalised()              # (C*K, D)
    sim = f @ protos.t()                                    # (N, C*K)
    # squared-Euclidean distance on the unit sphere
    d = 2.0 * (1.0 - sim)
    logp_comp = (-d / (2.0 * sigma2)).view(N, C, K)
    logp_c = torch.logsumexp(logp_comp, dim=2) - math.log(K)  # (N, C)

    if rare_prior_logits is not None:
        logp_c = logp_c + rare_prior_logits.view(1, C).to(device)

    tl = teacher_label.reshape(N).clamp(0, C - 1)
    logp_teacher = logp_c.gather(1, tl.view(N, 1)).squeeze(1)  # (N,)

    # --- Candidate restriction ---
    cand = logp_c.clone()
    if not symmetric:
        rare_mask = torch.zeros(C, dtype=torch.bool, device=device)
        if len(rare_class_ids) > 0:
            rare_mask[torch.as_tensor(rare_class_ids, device=device)] = True
        cand[:, ~rare_mask] = float('-inf')
    # the teacher's own class is never an "alternative" candidate
    cand.scatter_(1, tl.view(N, 1), float('-inf'))

    Lstar, cstar = cand.max(dim=1)                          # (N,)
    no_cand = torch.isinf(Lstar)                            # no alternative
    Lambda = torch.where(no_cand, torch.zeros_like(Lstar),
                         Lstar - logp_teacher)

    if standardize:
        Lam_img = Lambda.view(B, -1)
        med = Lam_img.median(dim=1, keepdim=True).values
        mad = (Lam_img - med).abs().median(dim=1, keepdim=True).values
        Lam_img = (Lam_img - med) / (mad + 1e-6)
        Lambda = Lam_img.reshape(N)

    a = torch.sigmoid(beta * Lambda)
    a = torch.where(no_cand, torch.zeros_like(a), a)
    cstar = torch.where(no_cand, torch.full_like(cstar, -1), cstar)

    return a.view(B, Hf, Wf), cstar.view(B, Hf, Wf)


def build_rare_prior_logits(num_classes, rare_class_ids, class_freq=None,
                            gamma=0.0):
    """Build ``log pi_c`` with an inverse-frequency boost for rare classes.

    ``pi_c ∝ (1 / N_c)^gamma`` for ``c in rare_class_ids`` and a flat
    prior elsewhere. ``gamma == 0`` returns ``None`` (prior disabled),
    which the A1 row relies on.

    Args:
        num_classes (int): C.
        rare_class_ids (list[int]): rare class indices.
        class_freq (list[float] | None): per-class pixel frequency. If
            ``None``, every rare class gets an equal boost of magnitude
            ``gamma`` (in log-space).
        gamma (float): boost strength. 0 disables the prior.

    Returns:
        Tensor | None: (C,) log-prior, or ``None`` when ``gamma == 0``.
    """
    if gamma == 0.0:
        return None
    logits = torch.zeros(num_classes)
    for c in rare_class_ids:
        if class_freq is not None and class_freq[c] > 0:
            logits[c] = gamma * math.log(1.0 / class_freq[c])
        else:
            logits[c] = gamma
    return logits
