# ---------------------------------------------------------------
# DAPCNHeadMixin: Mixin class for DAPCN auxiliary losses
# Integrates boundary-aware loss, dynamic anchor prototypes,
# DAPG loss, and persistent prototype memory bank with
# contrastive regularisation into any decode head.
#
# Dimension contract:
#   - DynamicAnchorModule receives a 4-D feature (B, C_da, H, W).
#     C_da depends on da_position:
#       * 'before_fusion': C_da = in_channels[-1]  (raw backbone)
#       * 'after_fusion' : C_da = channels          (fused decoder)
#   - PrototypeMemory always uses feature_dim = self.channels,
#     matching the fused_feature tensor from the decode head.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import MODELS, build_loss
# NOTE: DynamicAnchorModule is instantiated via the registry (MODELS.build)
# below, so no direct import is needed here. In this fork the module lives
# under mmseg.models.uda.dynamic_anchor; a direct top-level import would also
# risk a circular import through mmseg.models.uda.__init__.
from mmseg.models.utils.dapcn_utils import (
    compute_boundary_gt,
    extract_boundary_map,
)
from mmseg.models.utils.prototype_memory import (
    PrototypeMemory,
    prototype_contrastive_loss,
)


class DAPCNHeadMixin:
    """Mixin class that adds DAPCN auxiliary losses to any decode head.

    This mixin should be inherited BEFORE the base decode head class in MRO.
    It provides three auxiliary loss components:
        1. Boundary loss — Sobel/Laplacian/diff binary BCE, or
           affinity-based relational loss, or hybrid.
        2. DynamicAnchorModule + DAPGLoss — dataset-level learnable
           prototypes with per-batch EM refinement.
        3. PrototypeMemory + InfoNCE — persistent EMA class-conditioned
           prototypes accumulated across iterations.

    Loss budget (total = CE + auxiliary):
        L = L_ce
            + boundary_lambda   * L_boundary
            + proto_lambda      * L_dapg
            + contrastive_lambda * L_contrastive

    Dimension routing (critical):
        da_position='before_fusion':
            DA operates on inputs[-1] with C = in_channels[-1].
            This is the raw backbone feature before any decoder
            fusion, preserving the original feature geometry.
        da_position='after_fusion':
            DA operates on fused_feature with C = self.channels.
            This is the decoder's fused representation after
            multi-scale aggregation.

    Additional Args (for subclass __init__):
        da_position (str): Where the Dynamic Anchor Module operates.
            'before_fusion' or 'after_fusion'. Default: 'before_fusion'.
        da_feature_dim (int | None): Explicit override for the DA
            module's feature_dim. If None, inferred from da_position.
        boundary_lambda (float): Weight for boundary loss. Default: 0.3.
        proto_lambda (float): Weight for DAPG prototype loss. Default: 0.1.
        contrastive_lambda (float): Weight for memory-bank contrastive
            loss. Default: 0.1.
        boundary_mode (str): Gradient operator for boundary extraction
            ('sobel', 'laplacian', 'diff'). Default: 'sobel'.
        boundary_loss_mode (str): 'binary', 'affinity', or 'hybrid'.
            Default: 'binary'.
        hybrid_binary_weight (float): Binary weight in hybrid mode.
            Default: 0.5.
        contrastive_temperature (float): InfoNCE temperature. Default: 0.07.
        contrastive_sample_ratio (float): Fraction of valid pixels to
            sample for contrastive loss. Default: 0.1.
        warmup_iters (int): Iterations before enabling contrastive loss.
            Default: 500.
        num_prototypes_per_class (int): Number of prototypes per class
            in memory bank. Default: 1.
        prototype_ema (float): EMA momentum for prototype updates.
            Default: 0.999.
        dynamic_anchor (dict | None): Config for DynamicAnchorModule.
        dapg_loss (dict | None): Config for DAPGLoss.
        affinity_loss (dict | None): Config for AffinityBoundaryLoss.
    """

    def init_dapcn(self,
                   da_position='before_fusion',
                   da_feature_dim=None,
                   boundary_lambda=0.3,
                   proto_lambda=0.1,
                   contrastive_lambda=0.1,
                   boundary_mode='sobel',
                   boundary_loss_mode='binary',
                   hybrid_binary_weight=0.5,
                   contrastive_temperature=0.07,
                   contrastive_sample_ratio=0.1,
                   warmup_iters=500,
                   num_prototypes_per_class=1,
                   prototype_ema=0.999,
                   prototype_init_strategy='zeros',
                   lovasz_lambda=0.0,
                   lovasz_loss=None,
                   dynamic_anchor=None,
                   dapg_loss=None,
                   affinity_loss=None):
        """Initialize DAPCN components.

        Args:
            da_position (str): Where the Dynamic Anchor Module operates.
                'before_fusion': DA receives inputs[-1] (raw backbone
                    feature) with C = in_channels[-1].
                'after_fusion': DA receives fused_feature from the
                    decoder with C = self.channels.
                Default: 'before_fusion'.
            da_feature_dim (int | None): Explicit override for the DA
                module's feature_dim. When None, the dimension is
                automatically inferred:
                    before_fusion → in_channels[-1]
                    after_fusion  → self.channels
                Default: None.
            boundary_lambda (float): Weight for boundary loss. Default: 0.3.
            proto_lambda (float): Weight for DAPG prototype loss. Default: 0.1.
            contrastive_lambda (float): Weight for memory-bank contrastive
                loss. Default: 0.1.
            boundary_mode (str): Gradient operator for boundary extraction
                ('sobel', 'laplacian', 'diff'). Default: 'sobel'.
            boundary_loss_mode (str): 'binary', 'affinity', or 'hybrid'.
                Default: 'binary'.
            hybrid_binary_weight (float): Binary weight in hybrid mode.
                Default: 0.5.
            contrastive_temperature (float): InfoNCE temperature.
                Default: 0.07.
            contrastive_sample_ratio (float): Fraction of valid pixels to
                sample for contrastive loss. Default: 0.1.
            warmup_iters (int): Iterations before enabling contrastive loss.
                Default: 500.
            num_prototypes_per_class (int): Number of prototypes per class.
                Default: 1.
            prototype_ema (float): EMA momentum for prototype updates.
                Default: 0.999.
            prototype_init_strategy (str): Initialization strategy for
                prototypes ('zeros', 'random', etc.). Default: 'zeros'.
            dynamic_anchor (dict | None): Config for DynamicAnchorModule.
            dapg_loss (dict | None): Config for DAPGLoss.
            affinity_loss (dict | None): Config for AffinityBoundaryLoss.
        """
        # ---- DA position and feature dimension routing -----------------------
        assert da_position in ('before_fusion', 'after_fusion'), \
            f"da_position must be 'before_fusion' or 'after_fusion', " \
            f"got '{da_position}'"
        self.da_position = da_position

        # Infer DA feature_dim from position if not explicitly provided
        if da_feature_dim is not None:
            self._da_feature_dim = da_feature_dim
        elif da_position == 'before_fusion':
            # Raw backbone feature: use last backbone output channel
            if isinstance(self.in_channels, (list, tuple)):
                self._da_feature_dim = self.in_channels[-1]
            else:
                self._da_feature_dim = self.in_channels
        else:  # after_fusion
            # Fused decoder feature: use decoder hidden channels
            self._da_feature_dim = self.channels

        # ---- Hyper-parameters ------------------------------------------------
        self.boundary_lambda = boundary_lambda
        self.proto_lambda = proto_lambda
        self.contrastive_lambda = contrastive_lambda
        self.boundary_mode = boundary_mode
        self.boundary_loss_mode = boundary_loss_mode
        self.hybrid_binary_weight = hybrid_binary_weight
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_sample_ratio = contrastive_sample_ratio
        self.warmup_iters = warmup_iters
        self.num_prototypes_per_class = num_prototypes_per_class
        self.lovasz_lambda = lovasz_lambda

        # Iteration counter (updated in forward_train)
        self.register_buffer('_iter', torch.tensor(0, dtype=torch.long))

        # ---- Lovasz-Softmax auxiliary loss (direct mIoU surrogate) ----------
        # Injected here (not via loss_decode, which is single in this fork).
        # Gated by lovasz_lambda > 0; inert otherwise.
        if self.lovasz_lambda > 0:
            lv_cfg = lovasz_loss.copy() if lovasz_loss else {}
            lv_cfg.setdefault('type', 'LovaszLoss')
            lv_cfg.setdefault('loss_type', 'multi_class')
            lv_cfg.setdefault('per_image', False)
            self.lovasz_loss_fn = build_loss(lv_cfg)

        # ---- Build sub-modules -----------------------------------------------

        # 1. DynamicAnchorModule (persistent learnable prototypes)
        #    feature_dim is set to self._da_feature_dim based on da_position.
        if self.proto_lambda > 0:
            da_cfg = dynamic_anchor.copy() if dynamic_anchor else {}
            da_cfg.setdefault('type', 'DynamicAnchorModule')
            da_cfg.setdefault('feature_dim', self._da_feature_dim)
            da_cfg.setdefault('max_groups', 64)
            da_cfg.setdefault('temperature', 0.1)
            da_cfg.setdefault('num_iters', 3)
            da_cfg.setdefault('init_method', 'xavier')
            da_cfg.setdefault('min_quality', 0.1)
            da_cfg.setdefault('use_quality_gate', True)
            da_cfg.setdefault('use_mask_predictor', False)
            da_cfg.setdefault('ema_decay', 0.0)

            # Validate: feature_dim in the config must match our expectation
            actual_dim = da_cfg['feature_dim']
            assert actual_dim == self._da_feature_dim, (
                f"DynamicAnchorModule feature_dim={actual_dim} does not match "
                f"expected dimension={self._da_feature_dim} for "
                f"da_position='{da_position}'. "
                f"in_channels={self.in_channels}, channels={self.channels}. "
                f"Please check the config."
            )

            self.dynamic_anchor = MODELS.build(da_cfg)

            # DAPGLoss
            dapg_cfg = dapg_loss.copy() if dapg_loss else {}
            dapg_cfg.setdefault('type', 'DAPGLoss')
            dapg_cfg.setdefault('margin', 0.3)
            dapg_cfg.setdefault('lambda_inter', 0.5)
            dapg_cfg.setdefault('lambda_quality', 0.1)
            self.dapg_loss_fn = build_loss(dapg_cfg)

        # 2. Boundary loss (affinity branch)
        if self.boundary_lambda > 0 and boundary_loss_mode in (
                'affinity', 'hybrid'):
            aff_cfg = affinity_loss.copy() if affinity_loss else {}
            aff_cfg.setdefault('type', 'AffinityBoundaryLoss')
            aff_cfg.setdefault('temperature', 0.5)
            aff_cfg.setdefault('scale', 2)
            aff_cfg.setdefault('num_neighbors', 4)
            aff_cfg.setdefault('ignore_index', self.ignore_index)
            self.affinity_loss_fn = build_loss(aff_cfg)

        # 3. Prototype Memory Bank (class-aware, currently unused —
        #    unsupervised contrastive uses DA EMA prototypes instead)
        # if self.contrastive_lambda > 0:
        #     contrastive_feature_dim = getattr(self, '_da_feature_dim', None) or self.channels
        #     self.proto_memory = PrototypeMemory(
        #         num_classes=self.num_classes,
        #         feature_dim=contrastive_feature_dim,
        #         num_prototypes_per_class=num_prototypes_per_class,
        #         ema=prototype_ema,
        #         init_strategy=prototype_init_strategy,
        #     )

    def dapcn_forward_train(self, inputs, seg_logits, gt_semantic_seg,
                            fused_feature):
        """Compute DAPCN auxiliary losses.

        Args:
            inputs (list[Tensor]): Multi-scale backbone features.
                inputs[-1] has shape (B, in_channels[-1], H_b, W_b).
            seg_logits (Tensor): Classification logits (B, num_cls, H, W).
            gt_semantic_seg (Tensor): Ground truth semantic map.
            fused_feature (Tensor): Fused decoder feature before
                classification, shape (B, self.channels, H_f, W_f).

        Returns:
            dict: Dictionary of auxiliary losses.
        """
        losses = {}

        # Resize GT to match logit spatial dimensions
        _, _, H, W = seg_logits.shape
        gt_resized = F.interpolate(
            gt_semantic_seg.float(), size=(H, W),
            mode='nearest').long().squeeze(1)

        if self.boundary_lambda > 0:
            boundary_losses = self._boundary_loss(seg_logits, gt_resized, inputs)
            for k, v in boundary_losses.items():
                if torch.isnan(v):
                    raise ValueError(f"dapcn_forward_train: {k} is NaN")
            losses.update(boundary_losses)

        # ---- 2. DAPG prototype grouping loss --------------------------------
        if self.proto_lambda > 0:
            if self.da_position == 'before_fusion':
                if hasattr(self, 'in_index') and isinstance(self.in_index, (list, tuple)):
                    da_feat = inputs[self.in_index[-1]]
                else:
                    da_feat = inputs[-1]
            else:
                da_feat = fused_feature
            dapg_losses = self._dapg_loss(da_feat)
            for k, v in dapg_losses.items():
                if torch.isnan(v):
                    raise ValueError(f"dapcn_forward_train: {k} is NaN")
            losses.update(dapg_losses)

        # ---- 3. Memory-bank contrastive loss --------------------------------
        # if self.contrastive_lambda > 0:
        #     contrastive_losses = self._contrastive_loss(fused_feature, gt_resized)
        #     for k, v in contrastive_losses.items():
        #         if torch.isnan(v):
        #             raise ValueError(f"dapcn_forward_train: {k} is NaN")
        #     losses.update(contrastive_losses)

        # ---- 4. Lovasz-Softmax loss (direct mIoU surrogate) -----------------
        # Computed at FULL label resolution (logits upsampled bilinearly to the
        # GT size, matching how BaseDecodeHead.losses() computes CE) so the IoU
        # surrogate is optimised at the same resolution the metric is measured.
        if getattr(self, 'lovasz_lambda', 0.0) > 0:
            seg_logits_full = F.interpolate(
                seg_logits,
                size=gt_semantic_seg.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            gt_full = gt_semantic_seg.squeeze(1).long()
            loss_lovasz = self.lovasz_lambda * self.lovasz_loss_fn(
                seg_logits_full, gt_full, ignore_index=self.ignore_index)
            if torch.isnan(loss_lovasz):
                raise ValueError("dapcn_forward_train: loss_lovasz is NaN")
            losses['loss_lovasz'] = loss_lovasz

        # ---- Advance iteration counter --------------------------------------
        self._iter += 1

        return losses

    def _boundary_loss(self, seg_logits, gt_resized, inputs):
        """Compute boundary-aware loss.

        Args:
            seg_logits (Tensor): Classification logits.
            gt_resized (Tensor): Resized ground truth labels.
            inputs (list[Tensor]): Multi-scale backbone features.

        Returns:
            dict: Loss dictionary with 'loss_boundary' key.
        """
        losses = {}
        mode = self.boundary_loss_mode

        if mode == 'binary':
            b_pred = extract_boundary_map(seg_logits, mode=self.boundary_mode)
            b_gt = compute_boundary_gt(
                gt_resized, ignore_index=self.ignore_index)
            losses['loss_boundary'] = self.boundary_lambda * \
                F.binary_cross_entropy(b_pred, b_gt.float())

        elif mode == 'affinity':
            feat = inputs[-1].detach()
            losses['loss_boundary'] = self.boundary_lambda * \
                self.affinity_loss_fn(feat, gt_resized)

        elif mode == 'hybrid':
            b_pred = extract_boundary_map(seg_logits, mode=self.boundary_mode)
            b_gt = compute_boundary_gt(
                gt_resized, ignore_index=self.ignore_index)
            binary_l = F.binary_cross_entropy(b_pred, b_gt.float())
            affinity_l = self.affinity_loss_fn(inputs[-1].detach(), gt_resized)
            w = self.hybrid_binary_weight
            losses['loss_boundary'] = self.boundary_lambda * \
                (w * binary_l + (1 - w) * affinity_l)

        return losses

    def _dapg_loss(self, da_feat):
        """Compute dynamic-anchor prototype grouping loss.

        Args:
            da_feat (Tensor): Feature tensor for DA module, shape
                (B, C_da, H, W). C_da matches self._da_feature_dim.

        Returns:
            dict: Loss dictionary with 'loss_dapg' and component keys.
        """
        losses = {}
        _, C, _, _ = da_feat.shape

        if torch.isnan(da_feat).any() or torch.isinf(da_feat).any():
            # Defensive: return zero loss instead of crashing so training
            # can recover from transient NaN spikes in backbone features.
            nan_ratio = torch.isnan(da_feat).float().mean()
            inf_ratio = torch.isinf(da_feat).float().mean()
            import logging
            logging.getLogger('mmseg').warning(
                f"_dapg_loss: skipping DAPG loss due to NaN/Inf in da_feat "
                f"(NaN ratio={nan_ratio:.4f}, Inf ratio={inf_ratio:.4f}, "
                f"shape={da_feat.shape})")
            losses['loss_dapg'] = da_feat.sum() * 0.0
            losses['dapg_loss_intra'] = torch.tensor(0.0, device=da_feat.device)
            losses['dapg_loss_inter'] = torch.tensor(0.0, device=da_feat.device)
            losses['dapg_loss_quality'] = torch.tensor(0.0, device=da_feat.device)
            return losses

        # Sanity check: C must match DA module's feature_dim
        assert C == self.dynamic_anchor.feature_dim, (
            f"DA input feature dim={C} != "
            f"DynamicAnchorModule.feature_dim="
            f"{self.dynamic_anchor.feature_dim}. "
            f"da_position='{self.da_position}', "
            f"in_channels={self.in_channels}, channels={self.channels}."
        )

        # Centre BEFORE flattening, so feats_flat (DAPGLoss) and the EM input
        # share one space. See DynamicAnchorModule.center(): without this the
        # prototypes collapse to a single point on anisotropic features.
        da_feat = self.dynamic_anchor.center(da_feat)
        feats_flat = da_feat.permute(0, 2, 3, 1).reshape(-1, C)

        assign, proto, quality = self.dynamic_anchor(da_feat)

        # NaN detection: DA outputs
        if torch.isnan(assign).any():
            raise ValueError(f"_dapg_loss: assign contains NaN")
        if torch.isnan(proto).any():
            raise ValueError(f"_dapg_loss: proto contains NaN")
        if torch.isnan(quality).any():
            raise ValueError(f"_dapg_loss: quality contains NaN")

        loss_proto, proto_dict = self.dapg_loss_fn(
            feats_flat, assign, proto, quality)

        losses['loss_dapg'] = self.proto_lambda * loss_proto
        # Log sub-components (no gradient, informational)
        for k, v in proto_dict.items():
            losses[f'dapg_{k}'] = v.detach()

        return losses

    def _contrastive_loss(self, fused_feature, gt_resized):
        """Compute memory-bank contrastive loss and update the bank.

        Args:
            fused_feature (Tensor): Fused decoder feature (B, D, H, W)
                where D = self.channels.
            gt_resized (Tensor): Resized ground truth labels.

        Returns:
            dict: Loss dictionary with 'loss_contrastive' key.
        """
        losses = {}

        B, D, H, W = fused_feature.shape

        if torch.isnan(fused_feature).any():
            nan_ratio = torch.isnan(fused_feature).float().mean()
            raise ValueError(
                f"_contrastive_loss: fused_feature contains NaN "
                f"(ratio={nan_ratio:.4f})"
            )

        # Sanity check: D must match PrototypeMemory.feature_dim
        assert D == self.proto_memory.feature_dim, (
            f"Fused feature dim={D} != "
            f"PrototypeMemory.feature_dim={self.proto_memory.feature_dim}. "
            f"self.channels={self.channels}."
        )

        feats_flat = fused_feature.permute(0, 2, 3, 1).reshape(-1, D)
        labels_flat = gt_resized.reshape(-1)

        # Update memory bank (no gradient)
        valid_mask = labels_flat != self.ignore_index
        self.proto_memory.update(feats_flat.detach(), labels_flat,
                                 mask=valid_mask)

        # Contrastive loss (only after warmup & once memory is populated)
        if (self._iter >= self.warmup_iters
                and self.proto_memory.is_initialised()):

            # Sub-sample for memory efficiency
            valid_idx = torch.where(valid_mask)[0]
            if valid_idx.numel() == 0:
                losses['loss_contrastive'] = torch.tensor(
                    0.0, device=fused_feature.device, requires_grad=True)
                return losses

            n_sample = max(1, int(valid_idx.numel()
                                  * self.contrastive_sample_ratio))
            perm = torch.randperm(valid_idx.numel(),
                                  device=fused_feature.device)[:n_sample]
            sample_idx = valid_idx[perm]

            sample_feats = feats_flat[sample_idx]
            sample_labels = labels_flat[sample_idx]

            loss_c = prototype_contrastive_loss(
                features=sample_feats,
                prototypes=self.proto_memory(),
                labels=sample_labels,
                num_classes=self.num_classes,
                num_prototypes_per_class=self.num_prototypes_per_class,
                temperature=self.contrastive_temperature,
                ignore_index=self.ignore_index,
            )
            losses['loss_contrastive'] = self.contrastive_lambda * loss_c
        else:
            # Before warmup: zero loss (still differentiable)
            losses['loss_contrastive'] = feats_flat.sum() * 0.0

        return losses
