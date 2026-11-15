# ---------------------------------------------------------------
# DAPCN-SSL: Semi-Supervised Learning with DAPCN
#
# Adapts the DAPCN self-training framework from UDA to SSL.
# Key differences from UDA (dapcn.py):
#   - Labeled and unlabeled data come from the SAME domain
#   - No domain gap → ClassMix is used for consistency regularisation
#     (not domain bridging)
#   - pseudo_weight_ignore_top/bottom set to 0 (no rectification
#     artifacts in satellite imagery)
#   - No ImageNet Feature Distance (removed in UDA version too)
#   - Stronger pseudo-label confidence threshold (same domain →
#     teacher predictions are more reliable earlier)
#   - Prototype-based pseudo-label correction: prototypes learned
#     by DynamicAnchorModule are projected through the decoder's
#     classifier to produce class distributions, which are then
#     blended with teacher predictions via pixel-prototype affinity.
#
# The training loop:
#   1. Supervised loss on labeled images
#   2. DAPCN losses (boundary + prototype) on labeled
#   3. EMA teacher generates pseudo-labels on unlabeled
#   3b. Prototype correction refines pseudo-labels (when enabled)
#   4. ClassMix: labeled patches pasted onto unlabeled
#   5. Mixed supervised loss (confidence- and warmup-weighted)
#   6. DAPCN losses on mixed — DISABLED BY DEFAULT, see below
#
# Loss scope (default):
#   The boundary and prototype (DAPG) losses run on labeled images
#   ONLY.  On mixed images the supervision comes from noisy
#   pseudo-labels, and DAPGLoss in particular has no confidence
#   weighting mechanism (unlike the CE and boundary losses, which
#   accept ``pseudo_weight``), so applying it there mixes unweighted
#   noisy signal into the prototype geometry.  The mixed branch
#   therefore keeps exactly two mechanisms: prototype-based
#   pseudo-label correction (step 3b) and pseudo-label confidence
#   weighting (steps 3-5).  Set ``apply_boundary_on_target`` /
#   ``apply_proto_on_target`` to True to restore the old behaviour
#   for ablations.
# ---------------------------------------------------------------

import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.utils.checkpoint import checkpoint
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
class DAPCN_SSL(UDADecorator):
    """Semi-Supervised DAPCN for satellite image segmentation.

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
        pseudo_weight_mode (str): How teacher confidence weights the
            pseudo-labels. 'ratio' (DACS/DAFormer): every unlabeled pixel
            is trained on its argmax label with one scalar weight, the
            share of pixels above ``pseudo_threshold``. 'mask'
            (FixMatch/UniMatch): per-pixel weight 1 above the threshold,
            0 below. Default: 'ratio'.
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
                 pseudo_weight_mode='ratio',
                 # Prototype pseudo-label correction
                 proto_correction=True,
                 proto_correction_alpha=0.5,
                 proto_correction_start_iter=1000,
                 anchor_after_fusion=False,
                 # Learned corrector (step (a))
                 corrector_lambda=0.0,
                 # Frozen-prototype feature alignment
                 align_lambda=0.0,
                 align_start_iter=1000,
                 align_on='target',
                 # Sub-module configs
                 dynamic_anchor=None,
                 dapg_loss=None,
                 affinity_loss=None,
                 **cfg):
        super(DAPCN_SSL, self).__init__(**cfg)

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
        assert pseudo_weight_mode in ('ratio', 'mask'), (
            "pseudo_weight_mode must be 'ratio' or 'mask', got %r"
            % pseudo_weight_mode)
        self.pseudo_weight_mode = pseudo_weight_mode

        # Prototype-based pseudo-label correction
        self.proto_correction = proto_correction
        self.proto_correction_alpha = proto_correction_alpha
        self.proto_correction_start_iter = proto_correction_start_iter
        self.anchor_after_fusion = anchor_after_fusion
        self.corrector_lambda = corrector_lambda
        self.align_lambda = align_lambda
        self.align_start_iter = align_start_iter
        assert align_on in ('target', 'mixed'), (
            "align_on must be 'target' or 'mixed', got %r" % align_on)
        self.align_on = align_on

        # EMA teacher model
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        # Store sub-module configs
        self._dynamic_anchor_cfg = dynamic_anchor
        self._dapg_loss_cfg = dapg_loss

        # DynamicAnchorModule + DAPGLoss
        self.dynamic_anchor = None
        self.dapg_loss_fn = None
        self.conv_seg_proto = None
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

        # ---- Independent prototype classifier (corrector head) ----
        # Parameter-independent from ``decode_head.conv_seg`` on purpose.
        # conv_seg is, through the EMA teacher, very nearly the same
        # function the teacher uses to classify pixels, so projecting
        # prototypes through it makes p^c inherit the teacher's biases —
        # measured as correlated errors (p^c echoed the teacher's exact
        # mistake on 72% of the pixels the teacher got wrong).  Giving the
        # correction path its own head is DST's "the classifier head that
        # generates pseudo labels is never trained with pseudo labels"
        # applied here.
        #
        # Nonlinear rather than a single 1x1 conv: DST's own ablation
        # (Table 3) reports a nonlinear pseudo head consistently beating a
        # linear one, attributed to reduced representation degeneration
        # under biased supervision.  It also makes this head structurally
        # different from conv_seg, not merely differently initialised.
        #
        # It maps the prototype's NATIVE dimension straight to classes, so
        # it needs no proto_to_decoder bridge (the frozen-random-matrix
        # bug that made Solution 1's correction inert cannot recur here).
        if self.corrector_lambda > 0:
            hidden = max(feature_dim // 2, decode_head.num_classes)
            self.conv_seg_proto = nn.Sequential(
                nn.Conv2d(feature_dim, hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, decode_head.num_classes, kernel_size=1),
            )
        else:
            self.conv_seg_proto = None

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

        The reset is SKIPPED when the learned corrector is enabled: with
        ``corrector_lambda > 0`` the DynAnchor parameters receive real
        gradient from ``L_corrector`` starting at iteration 0, so their
        Adam moments are already meaningful and wiping them at iteration
        ``proto_correction_start_iter`` would discard that history rather
        than protect against a spike.
        """
        # Reset Adam state for DynAnchor params at activation boundary
        if (self.dynamic_anchor is not None
                and self.corrector_lambda == 0
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
        if self.conv_seg_proto is not None:
            # Corrector head: maps the prototype's native dimension
            # straight to classes, bypassing proto_to_decoder entirely.
            proto_4d = proto.unsqueeze(-1).unsqueeze(-1)  # (K, C, 1, 1)
            proto_logits = self.conv_seg_proto(proto_4d)
        else:
            # Original path: reuse the segmentation classifier.
            proto_dec = proto  # (K, feature_dim)
            if self.proto_to_decoder is not None:
                proto_dec = self.proto_to_decoder(proto)
            proto_4d = proto_dec.unsqueeze(-1).unsqueeze(-1)
            proto_logits = decode_head.conv_seg(proto_4d)
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

                # The only call that writes to the global memory bank,
                # and only for labeled images: the bank has to represent
                # the LABELED distribution, or the correction and
                # alignment paths that read it mean nothing.
                assign, proto, quality = self.dynamic_anchor(
                    anchor_feat, update_bank=is_labeled)
                loss_proto, loss_dict = self.dapg_loss_fn(
                    feats_flat, assign, proto, quality)

                losses['loss_proto'] = self.proto_lambda * loss_proto
                losses.update({
                    f'proto_{k}': v for k, v in loss_dict.items()})

        # --- Learned corrector loss (labeled images only) ---
        # Deliberately not gated by apply_proto_on_target: this loss needs
        # real ground truth, which mixed images do not have.
        if self.corrector_lambda > 0 and is_labeled:
            losses['loss_corrector'] = self.corrector_lambda * \
                self._compute_corrector_loss(decoder_features, seg_label)

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

    def _compute_corrector_loss(self, decoder_features, seg_label):
        """Supervise the correction path directly with ground truth.

        Trains the learned assignment (``query_proj``/``key_proj`` inside
        DynamicAnchorModule) and the independent ``conv_seg_proto`` head to
        make p^c itself a correct segmentation, rather than hoping a
        geometrically-tidy prototype bank happens to project into useful
        class distributions.  Without this, nothing in the objective ever
        tells the correction path what "right" looks like: its output is
        consumed by ``torch.max`` as a pseudo-label *target* and never
        backwarded.

            p^c_j = sum_i a_ij * softmax(conv_seg_proto(PT_i))
            L_corrector = CE(p^c, y_gt)          [labeled images only]

        The features are DETACHED first, so gradient reaches only
        ``query_proj``, ``key_proj``, ``conv_seg_proto`` and
        ``dynamic_anchor.prototypes`` — never the shared backbone or the
        main decode head.  This keeps a second classification objective
        from pulling the primary segmentation representation toward a
        different optimum.  (Note the existing DAPG loss deliberately does
        NOT detach — it is meant to shape the backbone's feature geometry.)

        Args:
            decoder_features (list[Tensor]): Multi-scale encoder features.
            seg_label (Tensor): Ground truth (N, H, W) or (N, 1, H, W).

        Returns:
            Tensor: Scalar cross-entropy loss.
        """
        # no_grad, not just .detach(): the result is detached either way, so
        # autograd was recording the whole stride-4 decoder fusion and then
        # throwing the graph away.  Those retained activations were enough to
        # OOM a 24 GB card at ~22.4 GB steady state.  Semantically identical,
        # roughly a gigabyte cheaper.
        with torch.no_grad():
            anchor_feat = self._get_anchor_features(decoder_features)
        B, C, Hf, Wf = anchor_feat.shape

        assign, proto, _ = self.dynamic_anchor(anchor_feat)

        proto_4d = proto.unsqueeze(-1).unsqueeze(-1)  # (K, C, 1, 1)
        proto_probs = torch.softmax(
            self.conv_seg_proto(proto_4d).squeeze(-1).squeeze(-1), dim=1)

        pc = torch.mm(assign, proto_probs)            # (N, num_classes)
        # Rows of `assign` are a softmax over all K prototypes, but the
        # quality gate may drop columns without renormalising, so pc can
        # sum to slightly under 1.  Renormalise here: cross-entropy needs
        # a proper distribution, and an improper one would silently add a
        # constant offset that varies with how many prototypes survived.
        pc = pc / pc.sum(dim=1, keepdim=True).clamp_min(self.EPS)

        num_classes = pc.shape[1]
        pc = pc.reshape(B, Hf, Wf, num_classes).permute(0, 3, 1, 2)

        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)
        gt = F.interpolate(
            seg_label.float().unsqueeze(1), size=(Hf, Wf),
            mode='nearest').long().squeeze(1)

        # pc holds probabilities, not logits, so take the log and use NLL
        # rather than cross_entropy (which would apply a second softmax).
        return F.nll_loss(
            torch.log(pc.clamp_min(self.EPS)), gt,
            ignore_index=self.ignore_index)

    def _compute_align_loss(self, decoder_features, unlabeled_mask=None):
        """Pull UNLABELED features toward frozen prototype anchors.

        Called on the features of the unlabeled image itself (Step 7 of
        forward_train), not on a ClassMix composite: the loss exists to
        close the gap between the unlabeled and labeled feature
        distributions, so both sides of it have to be one domain each.

        Labeled and unlabeled images here are the same domain — only which
        ones carry annotation differs — so their feature distributions
        ought to coincide.  The prototypes, shaped only by labeled images
        (``apply_proto_on_target=False``), stand in for the labeled
        distribution; this loss is the one mechanism that pulls unlabeled
        features toward them.  Nothing else does: the correction path only
        READS the prototypes, under ``no_grad``.

            L_align = mean_j sum_k A_jk * (1 - cos(f_j, stopgrad(q_k)))

        The prototypes are detached, so gradient flows to the FEATURES
        (and the assignment projections), never back into
        ``self.prototypes``.  That is the point: the anchors stay put and
        the unlabeled features move to them, rather than the anchors
        drifting to accommodate whatever the unlabeled features currently
        look like.

        Known risk, documented rather than designed around: weighting the
        pull by the current assignment ``A_jk`` makes this mechanically a
        form of entropy minimisation over cluster assignments.  It sharpens
        whatever the model already believes, so on a pixel whose feature
        has landed near the wrong cluster it will entrench that error
        rather than fix it.  ``align_start_iter`` is a hard requirement,
        not a tuning knob, for a related reason: prototypes measurably
        collapse to a single direction at initialisation (cos_mean 0.9999
        at iter 25 in every run so far), and aligning to collapsed anchors
        would pull every feature to one point.

        Args:
            decoder_features (list[Tensor]): Multi-scale encoder features.
            unlabeled_mask (Tensor | None): (N, 1, H, W), 1 where the
                pixel is to be pulled.  None -- the normal case now that
                the caller passes the unlabeled image itself rather than
                a ClassMix composite -- applies the loss to every pixel.

        Returns:
            Tensor: Scalar alignment loss.
        """
        # Gradient checkpointing on the fusion: this re-runs the decoder's
        # stride-4 feature fusion, and storing its activations for backward
        # pushes the step over the 24 GB card (the corrector run already
        # peaks at 21.9 GB).  Recomputing the fusion during backward instead
        # trades ~30% compute for that memory and keeps batch size at 4, so
        # this run stays comparable with every other one.
        # The features are unpacked into separate positional arguments, and
        # this is not cosmetic.  torch.utils.checkpoint decides whether to
        # build a backward graph by scanning its *tensor* arguments for
        # requires_grad; a single list argument is not a tensor, so the
        # scan finds nothing, and it returns an output with
        # requires_grad=False after warning "None of the inputs have
        # requires_grad=True".  Passing `decoder_features` as one list
        # therefore silently severed L_align from the graph.  Nothing
        # failed, because _parse_losses summed it with losses that did
        # carry grad_fn -- the alignment term was simply dropped.  Runs C
        # and D both trained this way and neither ever applied alignment.
        anchor_feat = checkpoint(
            lambda *feats: self._get_anchor_features(list(feats)),
            *decoder_features)
        B, C, Hf, Wf = anchor_feat.shape

        # Anchors: the global memory bank itself, not this image's EM
        # refinement of it.  Refining here would rebuild the anchors out
        # of the very features being pulled -- measured on run C, that
        # put all 96 anchors at cos 0.78 from the current image's own
        # feature mean, so every unlabeled pixel was dragged toward the
        # mean of whatever image it came from, and loss_align never
        # descended once in 9000 iterations.  The bank is the
        # dataset-level quantity, so it is what the features move toward.
        with torch.no_grad():
            proto_n = self.dynamic_anchor._get_initial_prototypes()  # (K, C)

        feats = anchor_feat.permute(0, 2, 3, 1).reshape(-1, C)
        feats_n = F.normalize(feats, dim=1)                  # (N, C), has grad

        cos = torch.mm(feats_n, proto_n.t())                 # (N, K')
        assign = torch.softmax(cos / self.dynamic_anchor.temperature, dim=1)
        per_pixel = (assign * (1.0 - cos)).sum(dim=1)        # (N,)

        if unlabeled_mask is not None:
            m = F.interpolate(
                unlabeled_mask.float(), size=(Hf, Wf),
                mode='nearest').reshape(-1)
            denom = m.sum().clamp_min(1.0)
            return (per_pixel * m).sum() / denom
        return per_pixel.mean()

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
        if self.pseudo_weight_mode == 'mask':
            pseudo_weight = ps_large_p.float()
        else:
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

        # === Step 7: Frozen-prototype alignment, unlabeled image ===
        # The point of this loss is to make the unlabeled feature
        # distribution match the labeled one, so it belongs on the
        # unlabeled DOMAIN -- not on the ClassMix composite.  It used to
        # run on `tgt_feat` with the pasted labeled patches masked out,
        # which reused an existing forward pass but measured a mixture:
        # those features come from an image that is part source, and
        # ClassMix's strong augmentation had been applied to all of it.
        # Here the student sees `target_img` as it is, so the features
        # being pulled are the unlabeled domain's own.
        #
        # The anchors are the global prototype bank, which
        # DynamicAnchorModule only ever writes to from the labeled DAPG
        # path (update_bank=is_labeled).  Labeled distribution on one
        # side, unlabeled on the other, and nothing in between.
        #
        # Backwarded on its own, after the mixed branch has released its
        # graph.  This costs one extra encoder forward, but the two
        # graphs never coexist, so the peak is set by whichever is
        # larger rather than by their sum.  extract_feat stops at the
        # encoder -- no decode head, no auxiliary losses -- and
        # _compute_align_loss checkpoints the fusion on top of that.
        if (self.align_lambda > 0
                and self.dynamic_anchor is not None
                and self.local_iter >= self.align_start_iter):
            # 'target' pulls the unlabeled image's own features.  'mixed'
            # reproduces the earlier placement -- the ClassMix composite,
            # with the pasted labeled patches masked out -- so the two can
            # be compared under an identical gradient path.  Both re-run
            # the encoder here rather than reusing Step 5's activations:
            # that graph is already released by target_loss.backward(),
            # and paying the same forward on both sides is what keeps the
            # comparison about WHICH image, not about how it was wired.
            if self.align_on == 'mixed':
                align_feat = self.get_model().extract_feat(mixed_img)
                align_mask = 1.0 - torch.cat(mix_masks, dim=0).float()
            else:
                align_feat = self.get_model().extract_feat(target_img)
                align_mask = None
            loss_align = (
                self.align_lambda
                * self._compute_align_loss(align_feat, align_mask))
            loss_align.backward()
            log_vars['%s.loss_align' % self.align_on] = loss_align.item()

        self.local_iter += 1
        return log_vars
