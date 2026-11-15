# ---------------------------------------------------------------
# DynamicAnchorModule: Dataset-Level Learnable Prototype Discovery
# Reference: docs/DAPCN.md
#
# Revision history:
#   v1 — Per-batch transient prototypes (FPS init each forward pass)
#   v2 — Persistent learnable prototypes (nn.Parameter) with
#         per-batch EM refinement.  Gradients from DAPGLoss flow
#         back through the differentiable EM chain to update the
#         persistent prototypes via the optimiser.
# ---------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F_func

from mmcv.runner import BaseModule
from mmseg.models.builder import MODELS


@MODELS.register_module(force=True)
class DynamicAnchorModule(BaseModule):
    """Dataset-level learnable prototype discovery with EM refinement.

    Unlike the original per-batch design where prototypes were
    re-initialised from features every forward pass, this version
    maintains **persistent learnable prototypes** as ``nn.Parameter``.
    They evolve across the entire training trajectory, capturing
    dataset-wide structure rather than batch-specific geometry.

    Each forward pass performs differentiable EM refinement starting
    from the persistent prototypes.  Because both the E-step (softmax
    assignment) and M-step (weighted mean) are differentiable, loss
    gradients propagate back through all EM iterations to the
    ``nn.Parameter``, allowing the optimiser (e.g. AdamW) to update
    them jointly with the rest of the network.

    Computation graph (single EM iteration):

        proto_t  ──► sim = feats @ proto_t.T / tau
                         │
                         ▼
                     assign = softmax(sim)
                         │
                         ▼
                     proto_{t+1} = assign.T @ feats / sizes
                         │
                         ▼
                     normalise(proto_{t+1})

    After ``num_iters`` iterations, the refined prototypes are passed
    to DAPGLoss.  Gradients flow:
        L_dapg ─► proto_refined ─► ... ─► assign_0 ─► self.prototypes

    Key differences from v1 (per-batch):
        - Prototypes are ``nn.Parameter``, not transient tensors.
        - No FPS/random/importance re-init each batch.
        - ``init_method`` controls the one-time parameter initialisation
          (Xavier, Kaiming, or normal).
        - Prototypes accumulate dataset knowledge across iterations.
        - Quality-net and mask-predictor remain for adaptive K and
          efficiency, but operate on the refined (not initial) protos.

    Args:
        feature_dim (int): Dimension of input features.
        max_groups (int): Number of persistent prototypes (K).
            Default: 64.
        min_quality (float): Quality threshold for filtering.
            Default: 0.1.
        num_iters (int): Number of EM refinement iterations per
            forward pass. Default: 3.
        temperature (float): Temperature for soft assignment.
            Default: 0.1.
        init_method (str): One-time prototype initialisation strategy.
            'xavier' — Xavier uniform (default, good for normalised
                       features).
            'kaiming' — Kaiming normal (good for ReLU-style features).
            'normal'  — N(0, 1/sqrt(D)).
            Default: 'xavier'.
        use_quality_gate (bool): Enable quality-net filtering.
            Default: True.
        use_mask_predictor (bool): Enable per-pixel attention mask.
            Default: False (removes the learnable predictor concern
            from the original TODO).
        ema_decay (float): If > 0, apply additional EMA smoothing on
            prototype parameters after EM refinement (provides extra
            stability).  0 disables EMA. Default: 0.0.
        use_learned_temperature (bool): Learn the E-step temperature
            instead of fixing it, as a log-scale parameter in the style
            of CLIP's logit scale.  Initialised to exactly
            ``1 / temperature``, so enabling it does not change the
            model at iteration 0 — it only lets training sharpen or
            soften the assignment.  One extra scalar parameter.
            Default: False.
        use_learned_assignment (bool): Replace the fixed cosine
            similarity of the E-step with a learned bilinear
            (Query-Key) similarity.  See ``_similarity``.  Initialised
            to the identity when square, so again the model starts
            exactly where plain cosine leaves off.  Much heavier than
            ``use_learned_temperature`` (two projection matrices), so
            prefer trying the temperature first. Default: False.
        assignment_dim (int, optional): Output dimension of the
            Query/Key projections when ``use_learned_assignment`` is
            set.  None means ``feature_dim`` (square, identity-init).
            A smaller value gives a low-rank projection: fewer
            parameters and less room to overfit a small labeled set,
            at the cost of losing the identity initialisation.
            Default: None.
        max_logit_scale (float): Upper clamp on the learned scale, to
            stop the softmax collapsing to a hard argmax.
            Default: 100.0.
        init_cfg (dict, optional): MMEngine init config. Default: None.

    Note:
        Both learned variants receive gradients from the same place the
        prototypes do — DAPGLoss on *labeled* images.  Nothing in the
        pseudo-label correction path back-propagates into them (its
        output is consumed by ``torch.max`` as a target, not as a
        loss).  So they learn to cluster the labeled feature
        distribution better, and the hope that this transfers to the
        unlabeled pool is an assumption to be checked empirically with
        ``ProtoCorrectionDebugHook`` — not something the objective
        enforces.
    """

    EPS = 1e-6
    MAX_PROTO_NORM = 10.0
    MIN_PROTO_NORM = 0.1

    def __init__(self,
                 feature_dim,
                 max_groups=64,
                 min_quality=0.1,
                 num_iters=3,
                 temperature=0.1,
                 init_method='xavier',
                 use_quality_gate=True,
                 use_mask_predictor=False,
                 ema_decay=0.0,
                 use_learned_temperature=False,
                 use_learned_assignment=False,
                 proto_source='memory',
                 assignment_dim=None,
                 max_logit_scale=100.0,
                 aux_banks=(),
                 init_cfg=None):
        super(DynamicAnchorModule, self).__init__(init_cfg)
        self.feature_dim = feature_dim
        self.max_groups = max_groups
        self.min_quality = min_quality
        self.num_iters = num_iters
        self.temperature = temperature
        self.init_method = init_method
        self.use_quality_gate = use_quality_gate
        self.use_mask_predictor = use_mask_predictor
        self.ema_decay = ema_decay
        self.use_learned_temperature = use_learned_temperature
        self.use_learned_assignment = use_learned_assignment
        assert proto_source in ('memory', 'gradient'), (
            "proto_source must be 'memory' or 'gradient', got %r" % proto_source)
        self.proto_source = proto_source
        self.max_logit_scale = max_logit_scale

        # ---- Persistent learnable prototypes ----
        self.prototypes = nn.Parameter(
            torch.empty(max_groups, feature_dim))
        self._init_prototypes()

        # ---- Learned E-step temperature (optional) ----
        # Stored in log space so the effective scale stays positive.
        # Initialised to 1 / temperature => identical to the fixed case
        # at iteration 0.
        if use_learned_temperature:
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(1.0 / temperature)))

        # ---- Learned Query/Key projections (optional) ----
        if use_learned_assignment:
            self.assignment_dim = (
                feature_dim if assignment_dim is None else assignment_dim)
            self.query_proj = nn.Linear(
                feature_dim, self.assignment_dim, bias=False)
            self.key_proj = nn.Linear(
                feature_dim, self.assignment_dim, bias=False)
            self._init_assignment_projections()
        else:
            self.assignment_dim = None

        # ---- Quality estimator (optional) ----
        if use_quality_gate:
            self.quality_net = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, 1),
                nn.Sigmoid()
            )

        # ---- Mask predictor (optional) ----
        if use_mask_predictor:
            self.mask_predictor = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, 1),
                nn.Sigmoid()
            )

        # ---- EMA shadow (non-learnable copy for smoothing) ----
        if ema_decay > 0:
            self.register_buffer(
                '_ema_prototypes',
                torch.zeros(max_groups, feature_dim))
            self.register_buffer(
                '_ema_initialised', torch.tensor(False))

        # ---- Auxiliary diagnostic banks (additive only) ----
        # One extra EMA buffer per name, e.g. ('labeled', 'unlabeled').
        # Written via forward(..., aux_bank=<name>), never used as EM
        # initialisation, never carrying gradient. With the default
        # aux_banks=() no buffer is registered and training is
        # bit-identical to before this feature existed.
        self.aux_banks = tuple(aux_banks)
        for _name in self.aux_banks:
            self.register_buffer(
                f'_ema_prototypes_{_name}',
                torch.zeros(max_groups, feature_dim))
            self.register_buffer(
                f'_ema_initialised_{_name}', torch.tensor(False))

    def _init_prototypes(self):
        """One-time initialisation of the persistent prototype tensor."""
        if self.init_method == 'xavier':
            nn.init.xavier_uniform_(self.prototypes)
        elif self.init_method == 'kaiming':
            # fan_out = feature_dim (each prototype is a row)
            nn.init.kaiming_normal_(
                self.prototypes, mode='fan_out', nonlinearity='relu')
        elif self.init_method == 'normal':
            std = 1.0 / math.sqrt(self.feature_dim)
            nn.init.normal_(self.prototypes, mean=0.0, std=std)
        else:
            raise ValueError(
                f"Unknown init_method: '{self.init_method}'. "
                f"Choose from 'xavier', 'kaiming', 'normal'.")

    def _init_assignment_projections(self):
        """Initialise the Query/Key projections.

        Square projections start at the identity, which makes the
        learned similarity reduce *exactly* to the plain cosine at
        iteration 0.  Enabling the flag then costs nothing up front and
        any divergence from cosine is something training chose.  A
        low-rank projection cannot be the identity, so it falls back to
        Xavier.
        """
        if self.assignment_dim == self.feature_dim:
            nn.init.eye_(self.query_proj.weight)
            nn.init.eye_(self.key_proj.weight)
        else:
            nn.init.xavier_uniform_(self.query_proj.weight)
            nn.init.xavier_uniform_(self.key_proj.weight)

    # ------------------------------------------------------------------
    # E-step similarity
    # ------------------------------------------------------------------
    def _get_initial_prototypes(self):
        """Starting point for this batch's EM refinement.

        The EMA memory bank when it exists, the learnable seed otherwise.
        The bank is what makes the prototypes global: it accumulates
        refined prototypes across the whole training trajectory, so each
        forward starts from dataset-level structure instead of
        rediscovering it from scratch.  Measured influence with
        num_iters=3: prototypes returned under ema_decay=0.999 sit at
        cos 0.40 from those returned under ema_decay=0 on the same
        batches, and retain cos 0.40 to the bank they started from -- EM
        does not wash the initialisation out.
        """
        if self.ema_decay > 0 and self._ema_initialised:
            source = self._ema_prototypes
        else:
            source = self.prototypes

        proto = F_func.normalize(source, dim=1)
        # A prototype that collected no mass can normalise to NaN and
        # would then poison every assignment through the softmax.
        return torch.where(
            torch.isnan(proto).any(dim=1, keepdim=True),
            torch.randn_like(proto) * 0.01, proto)

    def _similarity(self, feats_norm, proto):
        """Scaled similarity between pixels and prototypes.

        Plain cosine (the default) is standard practice for
        prototype-based methods, but it treats every feature dimension
        as equally informative.  The optional Query-Key form relaxes
        that:

            sim = <normalize(W_q f), normalize(W_k p)> * scale

        Re-normalising after projection keeps the result in [-1, 1] so
        that ``temperature`` retains its meaning; with identity
        projections the expression collapses back to plain cosine.

        Args:
            feats_norm (Tensor): L2-normalised features (N, C).
            proto (Tensor): L2-normalised prototypes (K, C).

        Returns:
            Tensor: Scaled similarity logits (N, K).
        """
        if self.use_learned_assignment:
            query = F_func.normalize(self.query_proj(feats_norm), dim=1)
            key = F_func.normalize(self.key_proj(proto), dim=1)
        else:
            query, key = feats_norm, proto

        cos = torch.mm(query, key.t())

        if self.use_learned_temperature:
            scale = self.logit_scale.exp().clamp(max=self.max_logit_scale)
        else:
            scale = 1.0 / self.temperature

        return cos * scale

    def _m_assign(self, sim, assign):
        """Responsibilities the M-step averages with (N, K).

        The plain EM uses the E-step softmax itself.  Subclasses override
        this to re-weight pixels or split the anchors into groups; the
        assignment RETURNED to the losses is always ``assign``.
        """
        return assign

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, features, update_bank=False, aux_bank=None):
        """Forward pass with EM refinement of persistent prototypes.

        Args:
            features (torch.Tensor): Encoder/decoder features of shape
                (B, C, H, W).
            aux_bank (str | None): Also fold this batch's refined
                prototypes into the named auxiliary bank (requires it
                to be listed in ``aux_banks``). Diagnostic only — the
                aux bank never seeds EM and never carries gradient.

        Returns:
            tuple:
                - assign_valid (Tensor): Soft assignments (N, K') where
                  K' <= max_groups after quality filtering.
                - proto_valid (Tensor): Refined prototype vectors (K', C).
                - quality_valid (Tensor): Quality scores (K',).
        """
        B, C, H, W = features.shape
        N = B * H * W
        feats = features.permute(0, 2, 3, 1).reshape(N, C)

        feats_norm = F_func.normalize(feats, dim=1)
        feats_norm = torch.where(
            torch.isnan(feats_norm).any(dim=1, keepdim=True),
            torch.zeros_like(feats_norm), feats_norm)

        if self.proto_source == 'gradient':
            # ---- Prototypes ARE the parameter -------------------------
            # No in-batch M-step and no EMA: pixels are assigned to the
            # persistent bank, and the bank is what the losses receive,
            # so it carries gradient.  DAPGLoss's intra term
            #     mean_j (1 - sum_k a_jk cos(f_j, q_k))
            # differentiates w.r.t. q_k and pulls each prototype toward
            # the features assigned to it, while the inter term pushes
            # them apart -- an M-step run by SGD over the whole dataset
            # instead of one discarded per batch.
            #
            # Cost, measured before: the bank starts at xavier, which
            # sits near-orthogonal to real features (cos to the batch
            # mean was -0.0003), so it walks in slowly -- proto_intra
            # fell only 0.9558 -> 0.8635 over 200 iters.  What it buys is
            # the absence of the cold-start collapse the memory path has:
            # proto_cos_mean stayed at 0.0010 where 'memory' hits 0.9999
            # by iter 25.
            proto = F_func.normalize(self.prototypes, dim=1)
            sim = torch.clamp(self._similarity(feats_norm, proto),
                              min=-50.0, max=50.0)
            assign = torch.softmax(sim, dim=1)
        else:
            # ---- EM initialisation: global memory bank, or the seed ----
            proto = self._get_initial_prototypes()

            # ---- Differentiable EM refinement ----
            assign = None
            for _ in range(self.num_iters):
                # E-step: soft assignment via scaled similarity.  Clamped
                # because a learned logit_scale can drive the exponent past
                # the float range before softmax ever sees it.
                sim = torch.clamp(self._similarity(feats_norm, proto),
                                  min=-50.0, max=50.0)
                assign = torch.softmax(sim, dim=1)       # (N, K)

                # M-step: recompute prototypes from the features assigned to
                # them.  ``feats`` is un-normalised so the mean keeps
                # magnitude information; the norm is clamped before the final
                # normalisation so a near-empty group cannot blow up or
                # collapse to zero.
                assign_m = self._m_assign(sim, assign)
                group_sizes = assign_m.sum(dim=0).clamp(min=self.EPS * 100)
                proto_new = torch.mm(assign_m.t(), feats) / group_sizes.unsqueeze(1)

                proto_norms = torch.norm(proto_new, dim=1, keepdim=True).clamp(
                    min=self.MIN_PROTO_NORM, max=self.MAX_PROTO_NORM)
                proto = F_func.normalize(proto_new / proto_norms, dim=1)

            # Edge case: num_iters == 0 (skip EM, use the initial protos)
            if assign is None:
                sim = torch.clamp(self._similarity(feats_norm, proto),
                                  min=-50.0, max=50.0)
                assign = torch.softmax(sim, dim=1)

            # ---- Save refined prototypes to the global memory bank ----
            # Restricted to ``update_bank`` callers, which is how the bank
            # stays a representation of the LABELED distribution: the
            # correction and alignment paths read it but must not write the
            # unlabeled statistics they are being measured against back into
            # it.
            if self.ema_decay > 0 and self.training and update_bank:
                self._update_memory(proto.detach())

            # ---- Auxiliary per-source bank (diagnostic only) ----
            # Fold the same refined prototypes into the named aux bank.
            # The aux bank never seeds a later EM run (see
            # _get_initial_prototypes) and _update_memory is no_grad,
            # so this changes no gradient and no training dynamic.
            if aux_bank is not None:
                if aux_bank not in self.aux_banks:
                    raise ValueError(
                        f"Unknown aux_bank: {aux_bank!r}. "
                        f"Configured aux_banks={list(self.aux_banks)}.")
                if self.ema_decay > 0 and self.training:
                    self._update_memory(proto.detach(), bank=aux_bank)


        # ---- Quality-guided filtering (adaptive K) ----
        if self.use_quality_gate:
            group_quality = self.quality_net(proto).squeeze(-1)
            valid_mask = group_quality > self.min_quality

            # Safety: ensure at least 1 prototype survives
            if valid_mask.sum() == 0:
                valid_mask[group_quality.argmax()] = True

            assign_valid = assign[:, valid_mask]
            proto_valid = proto[valid_mask]
            quality_valid = group_quality[valid_mask]
        else:
            # No filtering — all prototypes are valid
            assign_valid = assign
            proto_valid = proto
            quality_valid = torch.ones(
                proto.shape[0], device=proto.device)

        # ---- Optional per-pixel attention mask ----
        if self.use_mask_predictor:
            mask = self.mask_predictor(feats).squeeze(-1)  # (N,)
            assign_valid = assign_valid * mask.unsqueeze(1)
            assign_valid = assign_valid / (
                assign_valid.sum(dim=1, keepdim=True) + self.EPS)

        return assign_valid, proto_valid, quality_valid

    # ------------------------------------------------------------------
    # EMA helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    @torch.no_grad()
    def _update_memory(self, proto_refined, bank='main'):
        """Fold this batch's refined prototypes into the memory bank.

        Writes to the buffer ONLY.  An earlier revision also blended the
        result into ``self.prototypes.data``; that mutates a live
        parameter behind AdamW's back, leaving its momentum describing a
        value the parameter no longer holds.  The seed is left alone --
        after cold start it has done its job.

        Args:
            proto_refined (Tensor): Refined prototypes (K, C), detached.
            bank (str): 'main' or one of ``aux_banks``. Aux banks share
                the same EMA rule; they are read via ``bank_cos_mean``
                or the ``_ema_prototypes_<name>`` buffer directly.
        """
        if bank == 'main':
            ema = self._ema_prototypes
            initialised = self._ema_initialised
        else:
            ema = getattr(self, f'_ema_prototypes_{bank}')
            initialised = getattr(self, f'_ema_initialised_{bank}')
        if not initialised:
            ema.copy_(proto_refined)
            initialised.fill_(True)
        else:
            ema.mul_(self.ema_decay).add_(
                proto_refined, alpha=1.0 - self.ema_decay)

    def bank_cos_mean(self, bank='main'):
        """Mean pairwise cosine of one bank (diagnostic scalar).

        Off-diagonal mean of ``normalize(P) @ normalize(P).T`` over the
        full K prototypes (no quality-gate filtering). Returns NaN when
        the bank has not been written yet.
        """
        if bank == 'main':
            proto = self._ema_prototypes
            initialised = self._ema_initialised
        else:
            proto = getattr(self, f'_ema_prototypes_{bank}')
            initialised = getattr(self, f'_ema_initialised_{bank}')
        if not initialised.item():
            return torch.tensor(float('nan'), device=proto.device)
        pn = F_func.normalize(proto, dim=1)
        sim = pn @ pn.t()
        k = sim.shape[0]
        mask = 1.0 - torch.eye(k, device=sim.device)
        return (sim * mask).sum() / (k * (k - 1))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def extra_repr(self):
        return (
            f'feature_dim={self.feature_dim}, '
            f'max_groups={self.max_groups}, '
            f'num_iters={self.num_iters}, '
            f'temperature={self.temperature}, '
            f'init_method={self.init_method}, '
            f'quality_gate={self.use_quality_gate}, '
            f'mask_predictor={self.use_mask_predictor}, '
            f'ema_decay={self.ema_decay}, '
            f'learned_temperature={self.use_learned_temperature}, '
            f'learned_assignment={self.use_learned_assignment}, '
            f'assignment_dim={self.assignment_dim}, '
            f'aux_banks={list(self.aux_banks)}'
        )
