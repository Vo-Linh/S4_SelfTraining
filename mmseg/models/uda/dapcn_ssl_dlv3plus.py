# ---------------------------------------------------------------
# DAPCN-SSL-DLV3Plus: DeepLabV3+ variant of DAPCN-SSL
#
# Cloned from dapcn_ssl.py (the DAFormer+MiT-B5 version).
# This file is the designated development copy for experiments
# with DeepLabV3+ + ResNet-101 backbone.  The original
# dapcn_ssl.py (DAPCN_SSL) remains untouched for DAFormer.
#
# Decoder compatibility:
#   - DepthwiseSeparableASPPHead._fuse_features() for Solution 2
#   - DepthwiseSeparableASPPHead._transform_inputs() for Solution 1
#   - conv_seg input: 512-d (DeepLabV3+ channels)
#
# Dimension trace (ResNet-101 + DeepLabV3+):
#   Encoder: [256, 512, 1024, 2048]
#   S1: DynAnchor on 2048-d → Linear(2048, 512) → conv_seg
#   S2: DynAnchor on 512-d (fused) → conv_seg directly
#
# The training loop is identical to DAPCN_SSL:
#   1. Supervised loss on labeled images
#   2. DAPCN losses (boundary + prototype) on labeled
#   3. EMA teacher generates pseudo-labels on unlabeled
#   3b. Prototype correction refines pseudo-labels (when enabled)
#   4. ClassMix: labeled patches pasted onto unlabeled
#   5. Mixed supervised loss (confidence- and warmup-weighted)
#   6. DAPCN losses on mixed — DISABLED BY DEFAULT
#
# Loss scope (default): the boundary and prototype (DAPG) losses run
# on labeled images ONLY.  The mixed branch keeps exactly two
# mechanisms: prototype-based pseudo-label correction (step 3b) and
# pseudo-label confidence weighting (steps 3-5).  See dapcn_ssl.py
# for the full rationale.
# ---------------------------------------------------------------

import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor, build_loss
from mmseg.models.builder import MODELS
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.uda.utils.dapcn_utils import (
    compute_boundary_gt,
    extract_boundary_map,
)
from mmseg.models.utils.dacs_transforms import (
    get_class_masks,
    get_mean_std,
    strong_transform,
)


@UDA.register_module()
class DAPCN_SSL_DLV3Plus(UDADecorator):
    """Semi-Supervised DAPCN for DeepLabV3+ + ResNet-101.

    Cloned from DAPCN_SSL (DAFormer variant). This is the designated
    development copy for DeepLabV3+ experiments — the original
    dapcn_ssl.py remains untouched.

    Uses the same self-training framework as UDA-DAPCN but operates
    on labeled/unlabeled splits from the same domain instead of
    cross-domain source/target pairs.

    Prototype-Based Pseudo-Label Correction:
        The DynamicAnchorModule learns K dataset-level prototypes that
        capture the distribution of the feature space *without* relying
        on class labels. These prototypes can be projected through the
        decoder's classifier f_theta (conv_seg) to obtain per-prototype
        class probability distributions.

        Given the soft assignment (adjacency) a_ij between pixel j and
        prototype i, the corrected pseudo-label probability is:

            p^c_j = sum_{i=1}^{K} f_theta(PT_i) * a_ij

        This is then blended with the teacher's prediction:

            p_final = (1 - beta) * p_teacher + beta * p^c

        where beta is ``proto_correction_alpha``. The correction is
        activated after ``proto_correction_start_iter`` to allow the
        prototypes to stabilise before influencing pseudo-labels.

    Args:
        boundary_lambda (float): Weight for boundary loss. Default: 0.3.
        proto_lambda (float): Weight for prototype grouping loss. Default: 0.1.
        boundary_mode (str): Boundary extraction mode. Default: 'sobel'.
        apply_boundary_on_target (bool): Apply boundary loss on unlabeled
            (mixed) images. Default: False — the boundary loss is
            restricted to labeled images, where the supervision is clean.
        apply_proto_on_target (bool): Apply prototype loss on unlabeled
            (mixed) images. Default: False — DAPGLoss has no
            confidence-weighting mechanism, so on mixed images it would
            shape the prototypes with unweighted noisy pseudo-labels.
        boundary_loss_mode (str): 'binary', 'affinity', or 'hybrid'.
            Default: 'affinity'.
        hybrid_binary_weight (float): Binary weight in hybrid mode.
            Default: 0.5.
        ignore_index (int): Ignore index. Default: 255.
        pseudo_label_warmup_iters (int): Iterations before using pseudo
            labels at full weight. During warmup, pseudo_weight is scaled
            linearly from 0 to 1. Default: 0.
        proto_correction (bool): Enable prototype-based pseudo-label
            correction. Requires proto_lambda > 0. Default: True.
        proto_correction_alpha (float): Blending weight between teacher
            prediction and prototype-corrected prediction. 0 = teacher
            only, 1 = prototype only. Default: 0.5.
        proto_correction_start_iter (int): Start prototype correction
            after this many iterations. Prototypes need time to learn
            meaningful representations before they can correct pseudo-
            labels. Default: 1000.
        anchor_after_fusion (bool): Where to place DynamicAnchorModule.
            False (Solution 1): DynAnchor operates in encoder space
                (in_channels[-1], e.g. 512-d). A learned linear layer
                projects prototypes to decoder space for conv_seg.
            True (Solution 2): DynAnchor operates in the decoder's
                fused feature space (channels, e.g. 256-d). Prototypes
                are natively compatible with conv_seg — no projection
                needed. This is semantically richer as the fused features
                carry multi-scale contextual information.
            Default: False.
        dynamic_anchor (dict, optional): Config for DynamicAnchorModule.
        dapg_loss (dict, optional): Config for DAPGLoss.
        affinity_loss (dict, optional): Config for AffinityBoundaryLoss.
    """

    EPS = 1e-6

    def __init__(self,
                 boundary_lambda=0.3,
                 proto_lambda=0.1,
                 boundary_mode='sobel',
                 apply_boundary_on_target=False,
                 apply_proto_on_target=False,
                 boundary_loss_mode='affinity',
                 hybrid_binary_weight=0.5,
                 ignore_index=255,
                 pseudo_label_warmup_iters=0,
                 # Prototype pseudo-label correction
                 proto_correction=True,
                 proto_correction_alpha=0.5,
                 proto_correction_start_iter=1000,
                 anchor_after_fusion=False,
                 # Sub-module configs
                 dynamic_anchor=None,
                 dapg_loss=None,
                 affinity_loss=None,
                 **cfg):
        super(DAPCN_SSL_DLV3Plus, self).__init__(**cfg)

        # Self-training parameters
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg.get('pseudo_weight_ignore_top', 0)
        self.psweight_ignore_bottom = cfg.get('pseudo_weight_ignore_bottom', 0)
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        assert self.mix == 'class'

        # DAPCN-specific parameters
        self.boundary_lambda = boundary_lambda
        self.proto_lambda = proto_lambda
        self.boundary_mode = boundary_mode
        self.apply_boundary_on_target = apply_boundary_on_target
        self.apply_proto_on_target = apply_proto_on_target
        self.boundary_loss_mode = boundary_loss_mode
        self.hybrid_binary_weight = hybrid_binary_weight
        self.ignore_index = ignore_index
        self.pseudo_label_warmup_iters = pseudo_label_warmup_iters

        # Prototype-based pseudo-label correction
        self.proto_correction = proto_correction
        self.proto_correction_alpha = proto_correction_alpha
        self.proto_correction_start_iter = proto_correction_start_iter
        self.anchor_after_fusion = anchor_after_fusion

        # EMA teacher model
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        # Store sub-module configs
        self._dynamic_anchor_cfg = dynamic_anchor
        self._dapg_loss_cfg = dapg_loss

        # DynamicAnchorModule + DAPGLoss
        self.dynamic_anchor = None
        self.dapg_loss_fn = None
        if self.proto_lambda > 0:
            self._init_dynamic_anchor()
            self._init_dapg_loss()

        # AffinityBoundaryLoss
        if affinity_loss is not None:
            self.affinity_loss_fn = build_loss(affinity_loss)
        else:
            self.affinity_loss_fn = build_loss(dict(
                type='AffinityBoundaryLoss',
                temperature=0.5,
                scale=2,
                num_neighbors=4,
                ignore_index=ignore_index,
                loss_weight=1.0,
            ))

        self.class_probs = {}

    def _init_dynamic_anchor(self):
        """Build DynamicAnchorModule, auto-detecting feature_dim.

        Two placement strategies controlled by ``anchor_after_fusion``:

        Solution 1 (anchor_after_fusion=False):
            DynAnchor operates in encoder space (in_channels[-1], e.g. 512).
            A learned Linear(512→256) projects prototypes to decoder space
            for pseudo-label correction through conv_seg.

        Solution 2 (anchor_after_fusion=True):
            DynAnchor operates in the decoder's fused space (channels, e.g.
            256). Prototypes are natively compatible with conv_seg — no
            projection needed. Features are obtained via
            DAFormerHead._fuse_features() instead of _transform_inputs().
        """
        decode_head = self.get_model().decode_head
        in_channels = decode_head.in_channels
        if isinstance(in_channels, (list, tuple)):
            in_channels = in_channels[-1]
        decoder_channels = decode_head.channels  # conv_seg input dim

        # Determine feature_dim based on placement strategy
        if self.anchor_after_fusion:
            feature_dim = decoder_channels  # 256-d (fused decoder space)
        else:
            feature_dim = in_channels       # 512-d (encoder space)

        if self._dynamic_anchor_cfg is not None:
            da_cfg = deepcopy(self._dynamic_anchor_cfg)
            da_cfg.setdefault('feature_dim', feature_dim)
            self.dynamic_anchor = MODELS.build(da_cfg)
        else:
            from mmseg.models.uda.dynamic_anchor import DynamicAnchorModule
            self.dynamic_anchor = DynamicAnchorModule(
                feature_dim=feature_dim,
                max_groups=64,
                temperature=0.1,
                num_iters=3,
                init_method='xavier',
                min_quality=0.1,
            )

        # Solution 1 only: projection from encoder to decoder space
        # Solution 2: proto_to_decoder is None (no projection needed)
        if not self.anchor_after_fusion and feature_dim != decoder_channels:
            self.proto_to_decoder = nn.Linear(feature_dim, decoder_channels)
        else:
            self.proto_to_decoder = None

    def _init_dapg_loss(self):
        """Build DAPGLoss for prototype grouping."""
        if self._dapg_loss_cfg is not None:
            self.dapg_loss_fn = build_loss(self._dapg_loss_cfg)
        else:
            from mmseg.models.losses import DAPGLoss
            self.dapg_loss_fn = DAPGLoss(
                margin=0.3, lambda_inter=0.5, lambda_quality=0.1)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:
                ema_param.data = (alpha_teacher * ema_param.data +
                                  (1 - alpha_teacher) * param.data)
            else:
                ema_param.data[:] = (alpha_teacher * ema_param.data[:] +
                                     (1 - alpha_teacher) * param.data[:])

    def train_step(self, data_batch, optimizer, **kwargs):
        """Single training iteration.

        At ``proto_correction_start_iter``, Adam moment estimates for
        DynAnchor parameters are reset.  Before this point the module
        receives zero gradients (proto correction is inactive), so the
        running first/second moments are biased toward zero.  Resetting
        them prevents a destabilising spike when gradients suddenly
        appear.
        """
        # Reset Adam state for DynAnchor params at activation boundary
        if (self.dynamic_anchor is not None
                and self.local_iter == self.proto_correction_start_iter):
            for p in self.dynamic_anchor.parameters():
                if p in optimizer.state:
                    del optimizer.state[p]
            if self.proto_to_decoder is not None:
                for p in self.proto_to_decoder.parameters():
                    if p in optimizer.state:
                        del optimizer.state[p]

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def _get_anchor_features(self, encoder_features):
        """Get the feature map for DynamicAnchorModule.

        Args:
            encoder_features: Multi-scale encoder features (list of tensors).

        Returns:
            Tensor: Feature map in the appropriate space.
                Solution 1: encoder features[-1] (in_channels[-1]-d, e.g. 512)
                Solution 2: fused decoder features (channels-d, e.g. 256)
        """
        if self.anchor_after_fusion:
            # Solution 2: run features through decoder fusion
            decode_head = self.get_model().decode_head
            fused = decode_head._fuse_features(encoder_features)
            return fused
        else:
            # Solution 1: use raw encoder features (last scale)
            decode_head = self.get_model().decode_head
            feat = decode_head._transform_inputs(encoder_features)
            if isinstance(feat, list):
                feat = feat[-1]
            return feat

    def _get_pseudo_weight_scale(self):
        """Linear warmup scale for pseudo-label weight.

        Returns 1.0 after warmup completes. During warmup, returns
        a linearly increasing value from 0 to 1. This prevents the
        model from overfitting to noisy pseudo-labels in the early
        training phase when the teacher is still unreliable.
        """
        if self.pseudo_label_warmup_iters <= 0:
            return 1.0
        return min(1.0, self.local_iter / self.pseudo_label_warmup_iters)

    def _compute_prototype_probs(self, img, out_hw=None):
        """Compute prototype-derived class probabilities p^c.

        The DynamicAnchorModule learns K prototypes {PT_i} that capture
        dataset-level feature structure without class supervision. By
        passing these prototypes through the decoder's classifier f_theta
        (conv_seg), we obtain per-prototype class distributions.

        The prototype-derived probability for each pixel j is:

            p^c_j = sum_{i=1}^{K} f_theta(PT_i) * a_ij

        where a_ij is the soft assignment (adjacency) between pixel j
        and prototype i, measuring their cosine similarity.

        Shared by ``_correct_pseudo_labels`` (training) and
        ``compute_correction_diagnostics`` (offline debugging) so both
        paths use exactly the same computation.

        Note:
            Rows of ``assign`` are a softmax over all ``max_groups``
            prototypes, but the quality gate may drop some of them
            afterwards without renormalising.  Consequently the rows of
            ``p^c`` sum to <= 1 rather than exactly 1.  This matches the
            original behaviour and is harmless for argmax/max, but it is
            worth keeping in mind when interpreting the blended
            probabilities.

        Args:
            img (Tensor): Input images (B, 3, H, W).
            out_hw (tuple[int, int], optional): Target spatial size. When
                given and different from the feature resolution, the
                result is bilinearly upsampled to it. Default: None.

        Returns:
            Tensor | None: Probabilities (B, num_classes, H, W), or None
                if the DynamicAnchorModule is not available.
        """
        if self.dynamic_anchor is None:
            return None

        # --- Step 1: Extract student features ---
        # Use the student model's encoder (no gradient needed for extraction)
        with torch.no_grad():
            student_feat = self.get_model().extract_feat(img)

        # Get features in the appropriate space for DynAnchor
        # Solution 1: encoder[-1] (512-d), Solution 2: fused (256-d)
        feat = self._get_anchor_features(student_feat)
        decode_head = self.get_model().decode_head
        B, C_feat, Hf, Wf = feat.shape

        # --- Step 2: Run DynamicAnchorModule to get prototypes and
        #     soft assignments ---
        # assign: (B*Hf*Wf, K), proto: (K, C), quality: (K,)
        assign, proto, quality = self.dynamic_anchor(feat)

        # --- Step 3: Project prototypes through the classifier f_theta ---
        # Prototypes are in encoder space (in_channels, e.g. 512).
        # conv_seg expects decoder space (channels, e.g. 256).
        # Bridge with proto_to_decoder if dimensions differ.
        proto_dec = proto  # (K, feature_dim)
        if self.proto_to_decoder is not None:
            proto_dec = self.proto_to_decoder(proto)  # (K, decoder_channels)
        proto_4d = proto_dec.unsqueeze(-1).unsqueeze(-1)  # (K, C_dec, 1, 1)
        proto_logits = decode_head.conv_seg(proto_4d)  # (K, num_classes, 1, 1)
        proto_probs = torch.softmax(
            proto_logits.squeeze(-1).squeeze(-1), dim=1)  # (K, num_classes)

        # --- Step 4: Aggregate per-pixel via the soft assignment ---
        # p^c_j = sum_i a_ij * f_theta(PT_i)
        # assign: (N, K), proto_probs: (K, num_classes)
        proto_pixel_probs = torch.mm(assign, proto_probs)  # (N, num_classes)

        # Reshape to spatial format (B, num_classes, Hf, Wf)
        num_classes = proto_pixel_probs.shape[1]
        proto_pixel_probs = proto_pixel_probs.reshape(
            B, Hf, Wf, num_classes).permute(0, 3, 1, 2)

        # Upsample to the requested resolution
        if out_hw is not None and (Hf, Wf) != tuple(out_hw):
            proto_pixel_probs = F.interpolate(
                proto_pixel_probs, size=tuple(out_hw),
                mode='bilinear', align_corners=False)

        return proto_pixel_probs

    def _correct_pseudo_labels(self, target_img, target_img_metas,
                               ema_softmax):
        """Correct pseudo-labels using prototype-derived class distributions.

        Blends the teacher's prediction with the prototype signal from
        ``_compute_prototype_probs``:

            p_final = (1 - alpha) * p_teacher + alpha * p^c

        Args:
            target_img (Tensor): Unlabeled images (B, 3, H, W).
            target_img_metas (list[dict]): Image metadata.
            ema_softmax (Tensor): Teacher's softmax predictions
                (B, num_classes, H, W).

        Returns:
            Tensor: Corrected softmax probabilities (B, num_classes, H, W).
        """
        _, _, H_tea, W_tea = ema_softmax.shape
        # No gradient can reach these parameters from here: the blended
        # result is consumed by ``torch.max`` to build pseudo-label
        # *targets*, never by a loss that is backwarded.  Building the
        # graph anyway is pure waste — and expensive under
        # ``anchor_after_fusion=True``, where this call runs a full
        # decoder fusion at stride 4.
        with torch.no_grad():
            proto_probs = self._compute_prototype_probs(
                target_img, out_hw=(H_tea, W_tea))
        if proto_probs is None:
            return ema_softmax

        alpha = self.proto_correction_alpha
        return (1 - alpha) * ema_softmax + alpha * proto_probs

    @torch.no_grad()
    def compute_correction_diagnostics(self, img, img_metas):
        """Return the three pseudo-label signals for offline diagnostics.

        DIAGNOSTIC ONLY — never called from ``forward_train`` and never
        part of any backward pass.  Used by ``ProtoCorrectionDebugHook``
        to answer, using the held-out ground truth of nominally
        unlabeled images:

          1. Is the EMA teacher any good on its own?      -> 'teacher'
          2. Does the prototype signal carry usable class
             information on its own?                      -> 'proto'
          3. Does blending the two beat both components?  -> 'blend'

        The caller is responsible for putting the module in eval mode.
        This matters: ``DynamicAnchorModule`` mutates its persistent
        prototypes in-place when ``ema_decay > 0`` and ``self.training``
        is True, which would make diagnostics perturb training state.

        Args:
            img (Tensor): Images (B, 3, H, W).
            img_metas (list[dict]): Image metadata.

        Returns:
            dict: With keys ``teacher`` (Tensor), ``proto`` (Tensor or
                None when no DynamicAnchorModule), ``blend`` (Tensor,
                equal to ``teacher`` when ``proto`` is None) and
                ``correction_active`` (bool — whether training is
                currently applying the correction at this iteration).
        """
        ema_logits = self.get_ema_model().encode_decode(img, img_metas)
        p_teacher = torch.softmax(ema_logits, dim=1)

        out_hw = (p_teacher.shape[2], p_teacher.shape[3])
        p_proto = self._compute_prototype_probs(img, out_hw=out_hw)

        if p_proto is None:
            p_blend = p_teacher
        else:
            alpha = self.proto_correction_alpha
            p_blend = (1 - alpha) * p_teacher + alpha * p_proto

        correction_active = bool(
            self.proto_correction
            and self.dynamic_anchor is not None
            and self.local_iter >= self.proto_correction_start_iter)

        return dict(
            teacher=p_teacher,
            proto=p_proto,
            blend=p_blend,
            correction_active=correction_active)

    def _compute_dapcn_losses(self, logits, seg_label, decoder_features,
                              is_labeled=True, pseudo_weight=None):
        """Compute DAPCN losses (boundary + prototype grouping).

        Args:
            logits (Tensor): Segmentation logits (N, C, H, W).
            seg_label (Tensor): Labels (N, H, W) or (N, 1, H, W).
            decoder_features (list[Tensor]): Multi-scale decoder features.
            is_labeled (bool): Whether this is the labeled subset.
            pseudo_weight (Tensor, optional): Confidence weights.

        Returns:
            dict: Loss components.
        """
        losses = {}

        # Resize labels to match logits resolution
        _, _, H, W = logits.shape
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)
        seg_label_resized = F.interpolate(
            seg_label.float().unsqueeze(1), size=(H, W), mode='nearest',
        ).long().squeeze(1)

        # Resize pseudo_weight if provided
        if pseudo_weight is not None:
            if pseudo_weight.dim() == 3:
                pseudo_weight = pseudo_weight.unsqueeze(1)
            pseudo_weight_resized = F.interpolate(
                pseudo_weight, size=(H, W), mode='bilinear',
                align_corners=False)
        else:
            pseudo_weight_resized = None

        # --- Boundary loss ---
        if self.boundary_lambda > 0:
            if is_labeled or self.apply_boundary_on_target:
                boundary_loss = self._compute_boundary_loss(
                    logits, seg_label_resized, decoder_features,
                    is_labeled, pseudo_weight_resized)
                losses['loss_boundary'] = self.boundary_lambda * boundary_loss

        # --- Prototype grouping loss ---
        if self.proto_lambda > 0:
            if is_labeled or self.apply_proto_on_target:
                # Get features in the DynAnchor's native space
                anchor_feat = self._get_anchor_features(decoder_features)
                B, C, Hf, Wf = anchor_feat.shape
                feats_flat = anchor_feat.permute(0, 2, 3, 1).reshape(-1, C)

                assign, proto, quality = self.dynamic_anchor(anchor_feat)
                loss_proto, loss_dict = self.dapg_loss_fn(
                    feats_flat, assign, proto, quality)

                losses['loss_proto'] = self.proto_lambda * loss_proto
                losses.update({
                    f'proto_{k}': v for k, v in loss_dict.items()})

        return losses

    def _resize_to_feature(self, feat, seg_label, pseudo_weight):
        """Match label/weight to a feature map's spatial size.

        The affinity loss compares each pixel of ``feat`` with its
        neighbours and scores that against label agreement, so the two
        must live on the same grid. ``seg_label`` arrives at the decode
        head's output resolution (stride 4), while the feature handed to
        the affinity loss is the last encoder stage (stride 32) — a 8x
        difference that has to be reconciled here.

        Args:
            feat (Tensor): Reference feature map (N, C, H, W).
            seg_label (Tensor): Labels (N, H', W').
            pseudo_weight (Tensor | None): Weights (N, 1, H', W').

        Returns:
            tuple: (seg_label, pseudo_weight) on the feature's grid.
        """
        h, w = feat.shape[2], feat.shape[3]
        if seg_label.shape[-2:] != (h, w):
            seg_label = F.interpolate(
                seg_label.float().unsqueeze(1), size=(h, w),
                mode='nearest').long().squeeze(1)
        if pseudo_weight is not None and pseudo_weight.shape[-2:] != (h, w):
            pseudo_weight = F.interpolate(
                pseudo_weight, size=(h, w), mode='bilinear',
                align_corners=False)
        return seg_label, pseudo_weight

    def _compute_boundary_loss(self, logits, seg_label_resized,
                               decoder_features, is_labeled,
                               pseudo_weight_resized):
        """Compute boundary loss based on configured mode."""
        mode = self.boundary_loss_mode

        if mode == 'binary':
            b_pred = extract_boundary_map(logits, mode=self.boundary_mode)
            b_gt = compute_boundary_gt(
                seg_label_resized, ignore_index=self.ignore_index)
            if not is_labeled and pseudo_weight_resized is not None:
                w = pseudo_weight_resized.squeeze(1).unsqueeze(1)
                return F.binary_cross_entropy(
                    b_pred, b_gt.float(), weight=w)
            return F.binary_cross_entropy(b_pred, b_gt.float())

        elif mode == 'affinity':
            feat = decoder_features[-1]
            aff_label, aff_weight = self._resize_to_feature(
                feat, seg_label_resized, pseudo_weight_resized)
            return self.affinity_loss_fn(
                feat, aff_label, pseudo_weight=aff_weight)

        elif mode == 'hybrid':
            b_pred = extract_boundary_map(logits, mode=self.boundary_mode)
            b_gt = compute_boundary_gt(
                seg_label_resized, ignore_index=self.ignore_index)
            if not is_labeled and pseudo_weight_resized is not None:
                w = pseudo_weight_resized.squeeze(1)
                binary_l = F.binary_cross_entropy(
                    b_pred, b_gt.float(), weight=w)
            else:
                binary_l = F.binary_cross_entropy(b_pred, b_gt.float())
            feat = decoder_features[-1]
            aff_label, aff_weight = self._resize_to_feature(
                feat, seg_label_resized, pseudo_weight_resized)
            affinity_l = self.affinity_loss_fn(
                feat, aff_label, pseudo_weight=aff_weight)
            hw = self.hybrid_binary_weight
            return hw * binary_l + (1 - hw) * affinity_l

        else:
            raise ValueError(f"Unknown boundary_loss_mode: {mode}")

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Forward function for semi-supervised training.

        Training pipeline:
          1. Supervised loss on labeled images (with GT)
          2. DAPCN losses on labeled images (boundary + prototype)
          3. EMA teacher generates pseudo-labels on unlabeled images
          4. ClassMix: labeled patches pasted onto unlabeled
          5. Mixed supervised loss (GT for labeled regions, pseudo
             for unlabeled regions)
          6. DAPCN losses on mixed images

        Args:
            img (Tensor): Labeled images.
            img_metas (list[dict]): Labeled image info.
            gt_semantic_seg (Tensor): Ground truth labels.
            target_img (Tensor): Unlabeled images.
            target_img_metas (list[dict]): Unlabeled image info.

        Returns:
            dict[str, Tensor]: Loss components.
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device
        has_dapcn = self.boundary_lambda > 0 or self.proto_lambda > 0
        # The mixed branch runs the DAPCN losses only if explicitly asked
        # to (off by default). Tracking it separately matters for memory:
        # otherwise Step 5 retains the whole mixed-forward graph and Step 6
        # runs an extra decode-head pass, both to produce nothing.
        has_dapcn_on_target = (
            (self.boundary_lambda > 0 and self.apply_boundary_on_target)
            or (self.proto_lambda > 0 and self.apply_proto_on_target))

        # Init/update EMA teacher
        if self.local_iter == 0:
            self._init_ema_weights()
        if self.local_iter > 0:
            self._update_ema(self.local_iter)

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),
            'std': stds[0].unsqueeze(0)
        }

        # === Step 1: Supervised loss on labeled images ===
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=has_dapcn)

        # === Step 2: DAPCN losses on labeled images ===
        if has_dapcn:
            src_logits = self.get_model().decode_head(src_feat)
            src_dapcn_losses = self._compute_dapcn_losses(
                src_logits, gt_semantic_seg, src_feat,
                is_labeled=True, pseudo_weight=None)
            if src_dapcn_losses:
                src_dapcn_loss, src_dapcn_log = self._parse_losses(
                    src_dapcn_losses)
                log_vars.update(src_dapcn_log)
                src_dapcn_loss.backward()

        # === Step 3: Pseudo-label generation on unlabeled ===
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        ema_logits = self.get_ema_model().encode_decode(
            target_img, target_img_metas)
        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)

        # === Step 3b: Prototype-based pseudo-label correction ===
        # Correct teacher predictions using prototype class distributions.
        # p^c_j = sum_i f_theta(PT_i) * a_ij
        # Activated only when:
        #   - proto_correction is enabled
        #   - DynamicAnchorModule exists (proto_lambda > 0)
        #   - Training has passed proto_correction_start_iter
        use_correction = (
            self.proto_correction
            and self.dynamic_anchor is not None
            and self.local_iter >= self.proto_correction_start_iter
        )
        if use_correction:
            ema_softmax = self._correct_pseudo_labels(
                target_img, target_img_metas, ema_softmax)

        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)

        # Apply warmup scaling to pseudo-label weight
        warmup_scale = self._get_pseudo_weight_scale()
        pseudo_weight = pseudo_weight * warmup_scale

        if self.psweight_ignore_top > 0:
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones(pseudo_weight.shape, device=dev)

        # === Step 4: ClassMix augmentation ===
        # Mix labeled patches into unlabeled images for consistency
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack(
                    (gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack(
                    (gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # === Step 5: Mixed supervised loss ===
        model_output = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight,
            return_feat=True)
        tgt_feat = model_output.pop('features')
        target_loss, target_log_vars = self._parse_losses(model_output)
        log_vars.update(add_prefix(target_log_vars, 'mix'))
        target_loss.backward(retain_graph=has_dapcn_on_target)

        # === Step 6: DAPCN losses on mixed (unlabeled) ===
        if has_dapcn_on_target:
            tgt_logits = self.get_model().decode_head(tgt_feat)
            tgt_dapcn_losses = self._compute_dapcn_losses(
                tgt_logits, mixed_lbl, tgt_feat,
                is_labeled=False, pseudo_weight=pseudo_weight)
            if tgt_dapcn_losses:
                tgt_dapcn_loss, tgt_dapcn_log = self._parse_losses(
                    tgt_dapcn_losses)
                log_vars.update(add_prefix(tgt_dapcn_log, 'mix'))
                tgt_dapcn_loss.backward()

        self.local_iter += 1
        return log_vars
