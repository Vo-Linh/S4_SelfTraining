# ---------------------------------------------------------------
# DynamicAnchorModule: Dataset-Level Learnable Prototype Discovery
# Reference: docs/DAPCN.md
#
# Revision history:
#   v1 — Per-batch transient prototypes (FPS init each forward pass)
#   v2 — Persistent learnable prototypes (nn.Parameter) with
#         per-batch EM refinement.
#   v3 — EMA memory bank as EM initialisation.  Refined prototypes
#         are saved to the bank after each forward pass; the next
#         forward starts from the bank instead of from scratch.
#         Cold start falls back to the nn.Parameter seed.
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

    Maintains persistent prototypes that accumulate dataset-wide
    knowledge across the entire training trajectory.  When
    ``ema_decay > 0``, an EMA memory bank stores cross-batch
    prototype history and serves as the starting point for each
    forward pass's EM refinement, giving the model a warm start
    rather than re-discovering structure from scratch.

    Forward flow::

        memory bank (or seed)
              │
              ▼
        EM refinement (num_iters steps)
              │
              ▼
        refined prototypes ──► DAPGLoss
              │
              ▼
        EMA update ──► memory bank  (saved for next forward)

    Args:
        feature_dim (int): Dimension of input features.
        max_groups (int): Number of prototypes (K). Default: 64.
        min_quality (float): Quality threshold for filtering.
            Default: 0.1.
        num_iters (int): EM refinement iterations per forward pass.
            Default: 3.
        temperature (float): Temperature for soft assignment.
            Default: 0.1.
        init_method (str): One-time seed initialisation
            ('xavier', 'kaiming', 'normal'). Default: 'xavier'.
        use_quality_gate (bool): Enable quality-net filtering.
            Default: True.
        use_mask_predictor (bool): Enable per-pixel attention mask.
            Default: False.
        ema_decay (float): EMA momentum for the memory bank.
            0 disables the bank (EM starts from nn.Parameter seed).
            Default: 0.0.
        init_cfg (dict, optional): MMEngine init config. Default: None.
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
                 center_features=True,
                 init_cfg=None):
        super(DynamicAnchorModule, self).__init__(init_cfg)
        self.feature_dim = feature_dim
        self.max_groups = max_groups
        self.min_quality = min_quality
        self.num_iters = num_iters
        self.temperature = temperature
        self.init_method = init_method
        self.use_quality_gate = use_quality_gate
        # See center(): without this the prototypes provably collapse on
        # anisotropic (transformer) features. Set False only to ablate.
        self.center_features = center_features
        self.use_mask_predictor = use_mask_predictor
        self.ema_decay = ema_decay

        # ---- Seed prototypes (cold-start only) ----
        self.prototypes = nn.Parameter(
            torch.empty(max_groups, feature_dim))
        self._init_prototypes()

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

        # ---- EMA memory bank ----
        if ema_decay > 0:
            self.register_buffer(
                '_ema_prototypes',
                torch.zeros(max_groups, feature_dim))
            self.register_buffer(
                '_ema_initialised', torch.tensor(False))

    def center(self, feat):
        """Remove the shared mean direction from an anchor feature map.

        Deep encoder features are strongly anisotropic: a large common mean
        vector dominates the per-pixel residual, so cos(f_i, f_j) ~= 1 for
        *every* pixel pair (measured: 0.99999 on MiT-B5 stage-4). The E-step's
        softmax(f . q / tau) then sees near-constant similarity and assigns
        uniformly; the M-step recomputes each prototype as a weighted mean of
        near-identical vectors, so all K prototypes collapse to a single point
        and stay there. Signature: DAPG ``loss_inter`` pinned at exactly
        ``relu(1 - margin)``, and the pseudo-label correction degenerating into
        one constant class prior applied to every pixel.

        Call this where the anchor feature is *obtained*, so the flattened
        ``feats_flat`` given to DAPGLoss and the EM input live in the same
        space -- centering only inside the EM would leave DAPG comparing raw
        features against centered prototypes. The op is idempotent (recentering
        a centered tensor is a no-op), so ``forward`` also applies it as a
        safety net for callers that don't.

        Only the DAPCN branch is affected; the segmentation path is untouched.
        """
        if not self.center_features:
            return feat
        dims = (0, 2, 3) if feat.dim() == 4 else (0,)
        return feat - feat.mean(dim=dims, keepdim=True)

    def _init_prototypes(self):
        """One-time initialisation of the seed prototype tensor."""
        if self.init_method == 'xavier':
            nn.init.xavier_uniform_(self.prototypes)
        elif self.init_method == 'kaiming':
            nn.init.kaiming_normal_(
                self.prototypes, mode='fan_out', nonlinearity='relu')
        elif self.init_method == 'normal':
            std = 1.0 / math.sqrt(self.feature_dim)
            nn.init.normal_(self.prototypes, mean=0.0, std=std)
        else:
            raise ValueError(
                f"Unknown init_method: '{self.init_method}'. "
                f"Choose from 'xavier', 'kaiming', 'normal'.")

    # ------------------------------------------------------------------
    # EM initialisation
    # ------------------------------------------------------------------
    def _get_initial_prototypes(self):
        """Return normalised starting prototypes for EM refinement.

        Uses the EMA memory bank when available (warm start),
        otherwise falls back to the learnable seed (cold start).
        """
        if self.ema_decay > 0 and self._ema_initialised:
            source = self._ema_prototypes
        else:
            source = self.prototypes

        proto = F_func.normalize(source, dim=1)
        proto = torch.where(
            torch.isnan(proto).any(dim=1, keepdim=True),
            torch.randn_like(proto) * 0.01,
            proto
        )
        return proto

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, features):
        """Forward pass with EM refinement.

        Args:
            features (torch.Tensor): (B, C, H, W) encoder/decoder features.

        Returns:
            tuple:
                - assign_valid (Tensor): Soft assignments (N, K').
                - proto_valid (Tensor): Refined prototypes (K', C).
                - quality_valid (Tensor): Quality scores (K',).
        """
        # Idempotent safety net: callers should have centered at the feature
        # source (so DAPGLoss sees the same space), but if one did not, at
        # least protect the EM from prototype collapse.
        features = self.center(features)

        B, C, H, W = features.shape
        N = B * H * W
        feats = features.permute(0, 2, 3, 1).reshape(N, C)

        feats_norm = F_func.normalize(feats, dim=1)
        feats_norm = torch.where(
            torch.isnan(feats_norm).any(dim=1, keepdim=True),
            torch.zeros_like(feats_norm),
            feats_norm
        )

        # ---- EM initialisation: memory bank or seed ----
        proto = self._get_initial_prototypes()

        # ---- Differentiable EM refinement ----
        assign = None
        for _ in range(self.num_iters):
            # E-step: soft assignment
            sim = torch.mm(feats_norm, proto.t()) / self.temperature
            sim = torch.clamp(sim, min=-50, max=50)
            assign = torch.softmax(sim, dim=1)

            # M-step: recompute prototypes
            group_sizes = assign.sum(dim=0).clamp(min=self.EPS * 100)
            proto_new = torch.mm(assign.t(), feats) / group_sizes.unsqueeze(1)

            proto_norms = torch.norm(proto_new, dim=1, keepdim=True)
            proto_norms = torch.clamp(
                proto_norms, min=self.MIN_PROTO_NORM, max=self.MAX_PROTO_NORM)
            proto = F_func.normalize(proto_new / proto_norms, dim=1)

        # Edge case: num_iters == 0
        if assign is None:
            sim = torch.mm(feats_norm, proto.t()) / self.temperature
            assign = torch.softmax(sim, dim=1)

        # ---- Save refined prototypes to memory bank ----
        if self.ema_decay > 0 and self.training:
            self._update_memory(proto.detach())

        # ---- Quality-guided filtering ----
        if self.use_quality_gate:
            group_quality = self.quality_net(proto).squeeze(-1)
            valid_mask = group_quality > self.min_quality

            if valid_mask.sum() == 0:
                valid_mask[group_quality.argmax()] = True

            assign_valid = assign[:, valid_mask]
            proto_valid = proto[valid_mask]
            quality_valid = group_quality[valid_mask]
        else:
            assign_valid = assign
            proto_valid = proto
            quality_valid = torch.ones(
                proto.shape[0], device=proto.device)

        # ---- Optional per-pixel attention mask ----
        if self.use_mask_predictor:
            mask = self.mask_predictor(feats).squeeze(-1)
            assign_valid = assign_valid * mask.unsqueeze(1)
            assign_valid = assign_valid / (
                assign_valid.sum(dim=1, keepdim=True) + self.EPS)

        return assign_valid, proto_valid, quality_valid

    # ------------------------------------------------------------------
    # Memory bank
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_memory(self, proto_refined):
        """Save refined prototypes to the EMA memory bank."""
        if not self._ema_initialised:
            self._ema_prototypes.copy_(proto_refined)
            self._ema_initialised.fill_(True)
        else:
            self._ema_prototypes.mul_(self.ema_decay).add_(
                proto_refined, alpha=1.0 - self.ema_decay)

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
            f'ema_decay={self.ema_decay}'
        )
