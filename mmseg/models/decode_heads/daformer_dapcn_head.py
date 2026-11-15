# ---------------------------------------------------------------
# DAFormerDAPCNHead: Supervised segmentation with DAPCN losses
# Integrates boundary-aware loss, dynamic anchor prototypes,
# DAPG loss, and a persistent prototype memory bank with
# contrastive regularisation — all within the standard
# MMSegmentation decode-head interface (no UDA decorator).
# ---------------------------------------------------------------

import torch
import torch.nn.functional as F

from mmseg.models.builder import HEADS, MODELS, build_loss
from mmseg.models.decode_heads.daformer_head import DAFormerHead
from mmseg.models.uda.dynamic_anchor import DynamicAnchorModule
from mmseg.models.uda.utils.dapcn_utils import (
    compute_boundary_gt,
    extract_boundary_map,
)
from mmseg.models.utils.prototype_memory import (
    PrototypeMemory,
    prototype_contrastive_loss,
)


@HEADS.register_module()
class DAFormerDAPCNHead(DAFormerHead):
    """DAFormer decode head augmented with DAPCN auxiliary losses
    and a dynamic prototype memory bank for supervised training.

    Loss budget (total = CE + auxiliary):
        L = L_ce
            + boundary_lambda   * L_boundary
            + proto_lambda      * L_dapg
            + contrastive_lambda * L_contrastive

    Auxiliary components:
        1. Boundary loss — Sobel/Laplacian/diff binary BCE, or
           affinity-based relational loss, or hybrid.
        2. DynamicAnchorModule + DAPGLoss — dataset-level learnable
           prototypes (nn.Parameter) with per-batch EM refinement.
           Prototypes persist across iterations and are updated by
           the optimiser, capturing dataset-wide structure.
        3. PrototypeMemory + InfoNCE — persistent EMA class-conditioned
           prototypes accumulated across iterations; contrastive loss
           encourages pixel features to cluster near their class
           prototype and away from other classes.

    Additional Args (on top of DAFormerHead):
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
            sample for contrastive loss (memory efficiency). Default: 0.1.
        warmup_iters (int): Iterations before enabling contrastive loss
            (the memory bank needs a few updates first). Default: 500.
        num_prototypes_per_class (int): Number of prototypes per class
            in the memory bank.  >1 enables multi-modal representation.
            Default: 1.
        prototype_ema (float): EMA momentum for prototype updates.
            Default: 0.999.
        dynamic_anchor (dict | None): Config for DynamicAnchorModule.
        dapg_loss (dict | None): Config for DAPGLoss.
        affinity_loss (dict | None): Config for AffinityBoundaryLoss.
        log_dapg_diagnostics (bool): Log the detached DAPGLoss
            sub-components (dapg_cos_mean, ...) each iteration.
            Default: True.
    """

    def __init__(self,
                 # --- DAPCN loss weights ---
                 boundary_lambda=0.3,
                 proto_lambda=0.1,
                 contrastive_lambda=0.1,
                 # --- Boundary config ---
                 boundary_mode='sobel',
                 boundary_loss_mode='binary',
                 hybrid_binary_weight=0.5,
                 # --- Contrastive / memory config ---
                 contrastive_temperature=0.07,
                 contrastive_sample_ratio=0.1,
                 warmup_iters=500,
                 num_prototypes_per_class=1,
                 prototype_ema=0.999,
                 prototype_init_strategy='zeros',
                 # --- Sub-module configs ---
                 dynamic_anchor=None,
                 dapg_loss=None,
                 affinity_loss=None,
                 # --- Logging ---
                 log_dapg_diagnostics=True,
                 **kwargs):
        super().__init__(**kwargs)

        # Hyper-parameters
        self.boundary_lambda = boundary_lambda
        self.proto_lambda = proto_lambda
        self.contrastive_lambda = contrastive_lambda
        self.log_dapg_diagnostics = log_dapg_diagnostics
        self.boundary_mode = boundary_mode
        self.boundary_loss_mode = boundary_loss_mode
        self.hybrid_binary_weight = hybrid_binary_weight
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_sample_ratio = contrastive_sample_ratio
        self.warmup_iters = warmup_iters
        self.num_prototypes_per_class = num_prototypes_per_class

        # Iteration counter (updated in forward_train)
        self.register_buffer('_iter', torch.tensor(0, dtype=torch.long))

        # ---- Build sub-modules ------------------------------------------------

        # 1. DynamicAnchorModule (persistent learnable prototypes)
        if self.proto_lambda > 0:
            da_cfg = dynamic_anchor or {}
            da_cfg.setdefault('type', 'DynamicAnchorModule')
            da_cfg.setdefault('feature_dim', self.in_channels[-1])
            da_cfg.setdefault('max_groups', 64)
            da_cfg.setdefault('temperature', 0.1)
            da_cfg.setdefault('num_iters', 3)
            da_cfg.setdefault('init_method', 'xavier')
            da_cfg.setdefault('min_quality', 0.1)
            da_cfg.setdefault('use_quality_gate', True)
            da_cfg.setdefault('use_mask_predictor', False)
            da_cfg.setdefault('ema_decay', 0.0)
            self.dynamic_anchor = MODELS.build(da_cfg)

            # DAPGLoss
            dapg_cfg = dapg_loss or {}
            dapg_cfg.setdefault('type', 'DAPGLoss')
            dapg_cfg.setdefault('margin', 0.3)
            dapg_cfg.setdefault('lambda_inter', 0.5)
            dapg_cfg.setdefault('lambda_quality', 0.1)
            self.dapg_loss_fn = build_loss(dapg_cfg)

        # 2. Boundary loss (affinity branch)
        if self.boundary_lambda > 0 and boundary_loss_mode in (
                'affinity', 'hybrid'):
            aff_cfg = affinity_loss or {}
            aff_cfg.setdefault('type', 'AffinityBoundaryLoss')
            aff_cfg.setdefault('temperature', 0.5)
            aff_cfg.setdefault('scale', 2)
            aff_cfg.setdefault('num_neighbors', 4)
            aff_cfg.setdefault('ignore_index', self.ignore_index)
            self.affinity_loss_fn = build_loss(aff_cfg)

        # 3. Prototype Memory Bank
        if self.contrastive_lambda > 0:
            self.proto_memory = PrototypeMemory(
                num_classes=self.num_classes,
                feature_dim=self.channels,  # DAFormer fused dim (256)
                num_prototypes_per_class=num_prototypes_per_class,
                ema=prototype_ema,
                init_strategy=prototype_init_strategy,
            )

    # ------------------------------------------------------------------
    # Override forward_train to inject DAPCN losses
    # ------------------------------------------------------------------
    def forward_train(self, inputs, img_metas, gt_semantic_seg,
                      train_cfg, seg_weight=None):
        """Forward + loss computation with DAPCN auxiliary objectives.

        The call chain from EncoderDecoder is:
            EncoderDecoder.forward_train
              -> _decode_head_forward_train
                   -> self.forward_train(x, img_metas, gt, train_cfg)

        where ``inputs`` is a list of multi-scale backbone features.
        """
        # ---- Standard segmentation forward --------------------------------
        seg_logits = self.forward(inputs)                  # (B, C, H', W')
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)

        # ---- Prepare resized GT -------------------------------------------
        _, _, H, W = seg_logits.shape
        gt_resized = F.interpolate(
            gt_semantic_seg.float(), size=(H, W),
            mode='nearest').long().squeeze(1)              # (B, H, W)

        # ---- 1. Boundary loss ---------------------------------------------
        if self.boundary_lambda > 0:
            losses.update(self._boundary_loss(
                seg_logits, gt_resized, inputs))

        # ---- 2. DAPG prototype grouping loss ------------------------------
        if self.proto_lambda > 0:
            losses.update(self._dapg_loss(inputs))

        # ---- 3. Memory-bank contrastive loss ------------------------------
        if self.contrastive_lambda > 0:
            losses.update(self._contrastive_loss(
                seg_logits, gt_resized, inputs))

        # ---- Advance iteration counter ------------------------------------
        self._iter += 1

        return losses

    # ------------------------------------------------------------------
    # Individual loss branches
    # ------------------------------------------------------------------
    def _boundary_loss(self, seg_logits, gt_resized, inputs):
        """Compute boundary-aware loss."""
        losses = {}
        mode = self.boundary_loss_mode

        if mode == 'binary':
            b_pred = extract_boundary_map(seg_logits, mode=self.boundary_mode)
            b_gt = compute_boundary_gt(
                gt_resized, ignore_index=self.ignore_index)
            losses['loss_boundary'] = self.boundary_lambda * \
                F.binary_cross_entropy(b_pred, b_gt.float())

        elif mode == 'affinity':
            feat = inputs[-1]
            losses['loss_boundary'] = self.boundary_lambda * \
                self.affinity_loss_fn(feat, gt_resized)

        elif mode == 'hybrid':
            b_pred = extract_boundary_map(seg_logits, mode=self.boundary_mode)
            b_gt = compute_boundary_gt(
                gt_resized, ignore_index=self.ignore_index)
            binary_l = F.binary_cross_entropy(b_pred, b_gt.float())
            affinity_l = self.affinity_loss_fn(inputs[-1], gt_resized)
            w = self.hybrid_binary_weight
            losses['loss_boundary'] = self.boundary_lambda * \
                (w * binary_l + (1 - w) * affinity_l)

        return losses

    def _dapg_loss(self, inputs, update_bank=True):
        """Compute dynamic-anchor prototype grouping loss.

        Args:
            update_bank (bool): Forwarded to DynamicAnchorModule under
                proto_source='memory'.  Defaults to True -- unlike the
                DAPCN_SSL wrapper (which gates this on `is_labeled` so the
                bank only absorbs labeled-domain statistics), this head
                has no unlabeled-vs-labeled distinction of its own: every
                call to this method is meant to shape the bank.  A caller
                feeding it unlabeled features (see the two-dataloader
                supervised+unlabeled-DAPG experiment) passes update_bank
                explicitly rather than relying on this default.
        """
        losses = {}
        feat = inputs[-1]                                  # (B, C_enc, H, W)
        B, C, Hf, Wf = feat.shape
        feats_flat = feat.permute(0, 2, 3, 1).reshape(-1, C)

        assign, proto, quality = self.dynamic_anchor(feat, update_bank=update_bank)
        loss_proto, proto_dict = self.dapg_loss_fn(
            feats_flat, assign, proto, quality)

        losses['loss_dapg'] = self.proto_lambda * loss_proto
        # Log sub-components (no gradient, informational). Disabled for
        # runs that do not read them (e.g. Step 1: X_L only, the Step 2
        # separation metrics are not meaningful yet).
        if self.log_dapg_diagnostics:
            for k, v in proto_dict.items():
                losses[f'dapg_{k}'] = v.detach()
        return losses

    def _contrastive_loss(self, seg_logits, gt_resized, inputs):
        """Compute memory-bank contrastive loss and update the bank."""
        losses = {}

        # Use the fused decoder feature (after embed + fuse, before cls_seg).
        # We recompute the fused feature to get the pre-classification
        # representation.  This is the feature that the memory bank stores.
        feat = self._get_fused_feature(inputs)             # (B, D, H', W')

        B, D, H, W = feat.shape
        feats_flat = feat.permute(0, 2, 3, 1).reshape(-1, D)  # (BHW, D)
        labels_flat = gt_resized.reshape(-1)                    # (BHW,)

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
                    0.0, device=feat.device, requires_grad=True)
                return losses

            n_sample = max(1, int(valid_idx.numel()
                                  * self.contrastive_sample_ratio))
            perm = torch.randperm(valid_idx.numel(),
                                  device=feat.device)[:n_sample]
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_fused_feature(self, inputs):
        """Re-derive the fused decoder feature (before cls_seg).

        This mirrors ``DAFormerHead.forward`` but stops before the
        classification convolution, so we get the (B, channels, H, W)
        representation that the memory bank should store.
        """
        from mmseg.ops import resize as mmseg_resize

        x = inputs
        n, _, h, w = x[-1].shape
        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous() \
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            if _c[i].size()[2:] != os_size:
                _c[i] = mmseg_resize(
                    _c[i], size=os_size, mode='bilinear',
                    align_corners=self.align_corners)
        fused = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        return fused
