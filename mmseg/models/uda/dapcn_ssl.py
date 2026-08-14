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
#   5. Mixed supervised loss
#   6. DAPCN losses on mixed (target)
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
from mmseg.models.uda.dwpc_mixin import DWPCMixin
from mmseg.models.utils.dapcn_utils import (
    compute_boundary_gt,
    extract_boundary_map,
)
from mmseg.models.utils.dacs_transforms import (
    get_class_masks,
    get_mean_std,
    strong_transform,
)
from mmseg.models.utils.prototype_memory import (
    PrototypeMemory,
    prototype_contrastive_loss,
    prototype_contrastive_loss_extended,
)


@UDA.register_module()
class DAPCN_SSL(DWPCMixin, UDADecorator):
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
            (mixed) images. Default: True.
        apply_proto_on_target (bool): Apply prototype loss on unlabeled
            (mixed) images. Default: True.
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
                 apply_boundary_on_target=True,
                 apply_proto_on_target=True,
                 boundary_loss_mode='affinity',
                 hybrid_binary_weight=0.5,
                 ignore_index=255,
                 pseudo_label_warmup_iters=0,
                 # Prototype pseudo-label correction
                 proto_correction=True,
                 proto_correction_mode='class_prototype',
                 proto_correction_alpha=0.5,
                 proto_correction_temperature=0.1,
                 proto_correction_start_iter=1000,
                 anchor_after_fusion=False,
                 # PrototypeMemory + class-conditioned contrastive loss
                 contrastive_lambda=0.0,
                 contrastive_temp=0.07,
                 num_prototypes_per_class=1,
                 prototype_ema=0.999,
                 prototype_init_strategy='zeros',
                 # Extended contrastive objective (v2):
                 #   - contrastive_use_teacher: feed EMA-teacher confident
                 #     pseudo-labels into the memory bank as additional
                 #     supervision. Without this, the bank only sees the
                 #     labeled subset and quickly self-equilibrates.
                 #   - contrastive_use_dynanchor_negatives: use DynAnchor's
                 #     dataset-level prototypes as a hard-negative bank,
                 #     turning the N-way (N=num_classes) discrimination
                 #     into (N + K_dyn)-way against class-agnostic but
                 #     manifold-spanning negatives.
                 contrastive_use_teacher=False,
                 contrastive_use_dynanchor_negatives=False,
                 # Sub-module configs
                 dynamic_anchor=None,
                 dapg_loss=None,
                 affinity_loss=None,
                 # DWPC dual-witness pseudo-label correction
                 dwpc=None,
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

        # Prototype-based pseudo-label correction
        self.proto_correction = proto_correction
        if proto_correction_mode not in ('class_prototype', 'dynamic_anchor'):
            raise ValueError(
                'proto_correction_mode must be "class_prototype" or '
                '"dynamic_anchor"')
        if not 0.0 <= proto_correction_alpha <= 1.0:
            raise ValueError('proto_correction_alpha must be in [0, 1]')
        if proto_correction_temperature <= 0:
            raise ValueError('proto_correction_temperature must be positive')
        self.proto_correction_mode = proto_correction_mode
        self.proto_correction_alpha = proto_correction_alpha
        self.proto_correction_temperature = proto_correction_temperature
        self.proto_correction_start_iter = proto_correction_start_iter
        self.anchor_after_fusion = anchor_after_fusion

        # Class-conditioned PrototypeMemory + InfoNCE contrastive
        self.contrastive_lambda = contrastive_lambda
        self.contrastive_temp = contrastive_temp
        self.num_prototypes_per_class = num_prototypes_per_class
        self.prototype_ema = prototype_ema
        self.prototype_init_strategy = prototype_init_strategy
        self.contrastive_use_teacher = contrastive_use_teacher
        self.contrastive_use_dynanchor_negatives = (
            contrastive_use_dynanchor_negatives)

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

        # PrototypeMemory bank for class-conditioned contrastive loss AND
        # the ``class_prototype`` pseudo-label correction.
        #
        # The bank lives in the decoder's FUSED feature space
        # (``decode_head.channels``-d, 1/4 resolution), NOT the raw
        # encoder stage-4 space. Measured offline (tools/proto_correction_probe.py):
        # fused space gives a ~10x larger cosine margin (+0.076 vs +0.007)
        # and lifts nearest-prototype accuracy 0.45 -> 0.61. This is
        # independent of ``anchor_after_fusion``, which controls only the
        # DynamicAnchorModule / DAPG placement.
        #
        # NOTE: the bank dim changed from in_channels[-1] (512) to
        # channels (256), so old checkpoints' ``proto_memory.*`` buffers
        # will not resume into this layout — start a fresh run.
        self.proto_memory = None
        if (self.contrastive_lambda > 0
                or self.proto_correction_mode == 'class_prototype'):
            decode_head = self.get_model().decode_head
            num_classes = decode_head.num_classes
            proto_dim = decode_head.channels
            self.proto_memory = PrototypeMemory(
                num_classes=num_classes,
                feature_dim=proto_dim,
                num_prototypes_per_class=num_prototypes_per_class,
                ema=prototype_ema,
                init_strategy=prototype_init_strategy,
            )

        self.class_probs = {}

        # DWPC dual-witness pseudo-label correction (mixin). Reuses
        # self.proto_memory (Witness A) + new structural witness buffers.
        self._dwpc_init(dwpc, self.get_model().decode_head.num_classes)

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
        """Run one optimizer step without resetting established moments."""
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

        Every DAPCN consumer routes through here -- the DAPG loss (which flattens
        this map into ``feats_flat``), the DynamicAnchorModule EM, and the
        prototype pseudo-label correction -- so centering here keeps features and
        prototypes in the same space. Centering inside DynamicAnchorModule alone
        would leave DAPG comparing raw features against centered prototypes.
        """
        if self.anchor_after_fusion:
            # Solution 2: run features through decoder fusion
            decode_head = self.get_model().decode_head
            feat = decode_head._fuse_features(encoder_features)
        else:
            # Solution 1: use raw encoder features (last scale)
            decode_head = self.get_model().decode_head
            feat = decode_head._transform_inputs(encoder_features)
            if isinstance(feat, list):
                feat = feat[-1]

        # Centre at the source so the flattened feats_flat handed to DAPGLoss and
        # the EM input share one space. See DynamicAnchorModule.center().
        if self.dynamic_anchor is not None:
            feat = self.dynamic_anchor.center(feat)

        return feat

    def _get_proto_features(self, encoder_features):
        """Get features in the class-prototype bank's native space.

        The ``PrototypeMemory`` bank and the ``class_prototype``
        pseudo-label correction operate in the decoder's FUSED feature
        space (``decode_head.channels``-d, 1/4 resolution), obtained via
        ``DAFormerHead._fuse_features()``. This is distinct from the
        DynamicAnchorModule space returned by ``_get_anchor_features``
        (controlled by ``anchor_after_fusion``).

        The map is centred per batch (subtract the mean over dim (0,2,3))
        so the cosine geometry between pixels and class centroids is
        meaningful. Without it the shared common-mode direction dominates
        and every cosine similarity collapses toward 1 (measured 0.99999
        on MiT-B5 stage-4; see DynamicAnchorModule.center()).

        Args:
            encoder_features: Multi-scale encoder features (list of tensors).

        Returns:
            Tensor: Fused decoder feature map (B, channels, H/4, W/4),
                centred per batch.
        """
        decode_head = self.get_model().decode_head
        feat = decode_head._fuse_features(encoder_features)
        return feat - feat.mean(dim=(0, 2, 3), keepdim=True)

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

    @torch.no_grad()
    def _correct_pseudo_labels_from_class_prototypes(self, student_feat,
                                                     ema_softmax):
        """Blend teacher probabilities with class-prototype evidence.

        Every target pixel is compared against the persistent, semantic
        ``PrototypeMemory`` bank in the same anchor-feature space. For a
        class with multiple prototypes, its evidence is the maximum cosine
        similarity to any of that class's prototypes.
        """
        if self.proto_memory is None or not self.proto_memory.is_initialised():
            self._last_proto_correction_stats = {}
            return ema_softmax

        feat = self._get_proto_features(student_feat)
        B, C, Hf, Wf = feat.shape
        pixel_features = feat.permute(0, 2, 3, 1).reshape(-1, C)
        pixel_features = F.normalize(pixel_features, dim=1)
        pixel_features = torch.nan_to_num(pixel_features, nan=0.0)

        memory = self.proto_memory.get_all_normalised()
        similarities = torch.mm(pixel_features, memory.t())
        num_classes = self.proto_memory.num_classes
        similarities = similarities.reshape(
            -1, num_classes, self.proto_memory.K)
        class_scores = similarities.max(dim=2).values
        prototype_probs = torch.softmax(
            class_scores / self.proto_correction_temperature, dim=1)
        prototype_probs = prototype_probs.reshape(
            B, Hf, Wf, num_classes).permute(0, 3, 1, 2)

        if prototype_probs.shape[-2:] != ema_softmax.shape[-2:]:
            prototype_probs = F.interpolate(
                prototype_probs, size=ema_softmax.shape[-2:],
                mode='bilinear', align_corners=False)

        alpha = self.proto_correction_alpha
        blended = (1 - alpha) * ema_softmax + alpha * prototype_probs
        teacher_label = ema_softmax.argmax(dim=1)
        prototype_label = prototype_probs.argmax(dim=1)
        corrected_label = blended.argmax(dim=1)
        update_counts = self.proto_memory.update_counts.reshape(
            num_classes, self.proto_memory.K)
        self._last_proto_correction_stats = {
            'initialized_classes': (
                update_counts.gt(0).any(dim=1).float().sum()),
            'mean_updates_per_prototype': update_counts.float().mean(),
            'mean_nearest_similarity': class_scores.max(dim=1).values.mean(),
            'teacher_prototype_agreement': (
                teacher_label == prototype_label).float().mean(),
            'pseudo_label_flip_rate': (
                teacher_label != corrected_label).float().mean(),
        }
        return blended

    @torch.no_grad()
    def _correct_pseudo_labels_dynamic_anchor(self, target_img,
                                              target_img_metas, ema_softmax,
                                              student_feat=None):
        """Correct pseudo-labels using prototype-derived class distributions.

        The DynamicAnchorModule learns K prototypes {PT_i} that capture
        dataset-level feature structure without class supervision. By
        passing these prototypes through the decoder's classifier f_theta
        (conv_seg), we obtain per-prototype class distributions.

        The corrected probability for each pixel j is computed as:

            p^c_j = sum_{i=1}^{K} f_theta(PT_i) * a_ij

        where a_ij is the soft assignment (adjacency) between pixel j
        and prototype i, measuring their cosine similarity.

        The final blended probability is:

            p_final = (1 - alpha) * p_teacher + alpha * p^c

        Args:
            target_img (Tensor): Unlabeled images (B, 3, H, W).
            target_img_metas (list[dict]): Image metadata.
            ema_softmax (Tensor): Teacher's softmax predictions
                (B, num_classes, H, W).

        Returns:
            Tensor: Corrected softmax probabilities (B, num_classes, H, W).
        """
        # --- Step 1: Extract student features from unlabeled images ---
        # Caller may pre-extract and pass in to avoid a redundant forward
        # when the v2 contrastive teacher path also needs these features.
        if student_feat is None:
            with torch.no_grad():
                student_feat = self.get_model().extract_feat(target_img)

        # Get features in the appropriate space for DynAnchor
        # Solution 1: encoder[-1] (512-d), Solution 2: fused (256-d)
        feat = self._get_anchor_features(student_feat)
        decode_head = self.get_model().decode_head
        B, C_feat, Hf, Wf = feat.shape

        # --- Step 2: Run DynamicAnchorModule to get prototypes and
        #     soft assignments ---
        # assign: (B*Hf*Wf, K), proto: (K, C), quality: (K,)
        assign, proto, quality = self.dynamic_anchor(feat)

        K = proto.shape[0]
        N = B * Hf * Wf

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

        # --- Step 4: Compute corrected pseudo-label probability ---
        # p^c_j = sum_i a_ij * f_theta(PT_i)
        # assign: (N, K), proto_probs: (K, num_classes)
        corrected_probs = torch.mm(assign, proto_probs)  # (N, num_classes)

        # Reshape to spatial format (B, num_classes, Hf, Wf)
        num_classes = corrected_probs.shape[1]
        corrected_probs = corrected_probs.reshape(
            B, Hf, Wf, num_classes).permute(0, 3, 1, 2)

        # Upsample to match teacher's resolution
        _, _, H_tea, W_tea = ema_softmax.shape
        if (Hf, Wf) != (H_tea, W_tea):
            corrected_probs = F.interpolate(
                corrected_probs, size=(H_tea, W_tea),
                mode='bilinear', align_corners=False)

        # --- Step 5: Blend teacher prediction with prototype correction ---
        alpha = self.proto_correction_alpha
        blended = (1 - alpha) * ema_softmax + alpha * corrected_probs

        return blended

    def _correct_pseudo_labels(self, target_img, target_img_metas,
                               ema_softmax, student_feat=None):
        """Dispatch to the configured pseudo-label correction mechanism."""
        if self.proto_correction_mode == 'class_prototype':
            if student_feat is None:
                with torch.no_grad():
                    student_feat = self.get_model().extract_feat(target_img)
            return self._correct_pseudo_labels_from_class_prototypes(
                student_feat, ema_softmax)
        return self._correct_pseudo_labels_dynamic_anchor(
            target_img, target_img_metas, ema_softmax, student_feat)

    @torch.no_grad()
    def _eval_pseudo_label_correction(self, raw_label, corrected_label,
                                      target_gt):
        """Score raw vs prototype-corrected pseudo-labels against hidden GT.

        The unlabeled ground truth is used purely as an evaluation oracle to
        measure whether the prototype correction fixes real label noise. It
        never enters any loss, backward pass, or memory-bank update.

        Reports overall and per-class metrics. For each class ``c`` it emits
        ``pl_acc.c{c}.raw`` / ``.corrected`` / ``.delta`` (corrected - raw)
        and ``pl_acc.c{c}.flip_correct`` (fraction of that class's flipped
        pixels that landed on the correct class; omitted when there were no
        flips in that class).

        Args:
            raw_label (Tensor): (B, H, W) teacher pseudo-label *before*
                correction.
            corrected_label (Tensor): (B, H, W) pseudo-label *after* the
                prototype blend (argmax).
            target_gt (Tensor | None): (B, 1, H, W) or (B, H, W) hidden GT.

        Returns:
            dict[str, float]: overall + per-class ``pl_acc.*`` metrics.
            Empty dict when no GT is provided or no valid pixels exist.
        """
        if target_gt is None:
            return {}
        if target_gt.dim() == 4:
            target_gt = target_gt.squeeze(1)
        valid = target_gt != self.ignore_index
        if not valid.any():
            return {}

        raw_correct = (raw_label == target_gt) & valid
        cor_correct = (corrected_label == target_gt) & valid
        flipped = (raw_label != corrected_label) & valid
        flip_correct_mask = (corrected_label == target_gt) & flipped

        n_valid = valid.sum().float()
        out = {
            'pl_acc.raw': (raw_correct.sum().float() / n_valid).item(),
            'pl_acc.corrected': (cor_correct.sum().float() / n_valid).item(),
            'pl_acc.flip_correct': (
                flip_correct_mask.sum().float()
                / flipped.sum().float().clamp(min=1.0)).item(),
        }
        out['pl_acc.delta'] = out['pl_acc.corrected'] - out['pl_acc.raw']

        for c in range(self.num_classes):
            cls_mask = (target_gt == c) & valid
            n_c = cls_mask.sum().float()
            if n_c.item() == 0:
                continue
            raw_c = (raw_correct & cls_mask).sum().float() / n_c
            cor_c = (cor_correct & cls_mask).sum().float() / n_c
            out[f'pl_acc.c{c}.raw'] = raw_c.item()
            out[f'pl_acc.c{c}.corrected'] = cor_c.item()
            out[f'pl_acc.c{c}.delta'] = (cor_c - raw_c).item()
            n_flip_c = (flipped & cls_mask).sum().float()
            if n_flip_c.item() > 0:
                out[f'pl_acc.c{c}.flip_correct'] = (
                    (flip_correct_mask & cls_mask).sum().float()
                    / n_flip_c).item()

        return out

    def _compute_dapcn_losses(self, logits, seg_label, decoder_features,
                              is_labeled=True, pseudo_weight=None,
                              unlabeled_decoder_features=None,
                              teacher_pseudo_label=None,
                              teacher_high_conf_mask=None):
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
        # Cache the EM-refined DA prototypes from this step so the
        # contrastive block below can reuse them as live hard negatives
        # (they're adapted to THIS batch's feature distribution rather
        # than the static xavier seed).
        dyn_proto_refined = None
        if self.proto_lambda > 0:
            if is_labeled or self.apply_proto_on_target:
                # Get features in the DynAnchor's native space
                anchor_feat = self._get_anchor_features(decoder_features)
                B, C, Hf, Wf = anchor_feat.shape
                feats_flat = anchor_feat.permute(0, 2, 3, 1).reshape(-1, C)

                assign, proto, quality = self.dynamic_anchor(anchor_feat)
                dyn_proto_refined = proto
                loss_proto, loss_dict = self.dapg_loss_fn(
                    feats_flat, assign, proto, quality)

                losses['loss_proto'] = self.proto_lambda * loss_proto
                losses.update({
                    f'proto_{k}': v for k, v in loss_dict.items()})

        # --- Class-conditioned prototype contrastive loss (labeled only) ---
        # Skipped on mixed/unlabeled CE step because the query features
        # need reliable supervision; teacher pseudo-labels enter via the
        # memory-bank update path instead, not as queries.
        if is_labeled and self.proto_memory is not None:
            anchor_feat = self._get_proto_features(decoder_features)
            _, C_proto, Hf, Wf = anchor_feat.shape
            feats_flat = anchor_feat.permute(0, 2, 3, 1).reshape(-1, C_proto)
            if seg_label_resized.shape[-2:] != (Hf, Wf):
                labels_proto = F.interpolate(
                    seg_label_resized.float().unsqueeze(1),
                    size=(Hf, Wf), mode='nearest').long().squeeze(1)
            else:
                labels_proto = seg_label_resized
            labels_flat = labels_proto.reshape(-1)
            valid_mask = labels_flat != self.ignore_index

            # 1. Update memory bank from LABELED GT features (always)
            self.proto_memory.update(
                feats_flat.detach(), labels_flat, mask=valid_mask)

            # 2. Optionally update bank from TEACHER-confident pseudo-
            #    labeled features (v2). This injects 3500-image worth of
            #    feature statistics into the bank, breaking the self-
            #    referential equilibrium of the labeled-only bank.
            if (self.contrastive_use_teacher
                    and unlabeled_decoder_features is not None
                    and teacher_pseudo_label is not None
                    and teacher_high_conf_mask is not None):
                un_anchor = self._get_proto_features(
                    unlabeled_decoder_features)
                _, _, Hu, Wu = un_anchor.shape
                un_feats_flat = un_anchor.permute(
                    0, 2, 3, 1).reshape(-1, C_proto)
                if teacher_pseudo_label.shape[-2:] != (Hu, Wu):
                    tp = F.interpolate(
                        teacher_pseudo_label.float().unsqueeze(1),
                        size=(Hu, Wu), mode='nearest').long().squeeze(1)
                else:
                    tp = teacher_pseudo_label
                if teacher_high_conf_mask.shape[-2:] != (Hu, Wu):
                    tm = F.interpolate(
                        teacher_high_conf_mask.float().unsqueeze(1),
                        size=(Hu, Wu), mode='nearest').squeeze(1).bool()
                else:
                    tm = teacher_high_conf_mask.bool()
                un_labels_flat = tp.reshape(-1)
                un_mask_flat = tm.reshape(-1)
                self.proto_memory.update(
                    un_feats_flat.detach(), un_labels_flat, mask=un_mask_flat)

            # 3. Compose the InfoNCE objective.
            use_ext = (
                self.contrastive_use_dynanchor_negatives
                and self.dynamic_anchor is not None
                and self.dynamic_anchor.feature_dim == C_proto
            )
            if use_ext:
                # Hard-negative bank: the *EM-refined* DA prototypes
                # produced in the L_proto block above. These cluster
                # in the regions of feature-space the queries actually
                # occupy — genuine hard negatives, not static random
                # vectors. ``detach()`` so the contrastive gradient
                # does NOT reshape the DA geometry (DAPGLoss owns that).
                if dyn_proto_refined is not None:
                    extra_neg = dyn_proto_refined.detach()
                else:
                    # Cold path: L_proto wasn't computed this step
                    # (proto_lambda=0). Run DA on the labeled features
                    # ourselves, no_grad since we're only using it as
                    # a negative bank, not optimizing through it.
                    with torch.no_grad():
                        _, dyn_p, _ = self.dynamic_anchor(anchor_feat)
                    extra_neg = dyn_p.detach()
                loss_c = prototype_contrastive_loss_extended(
                    features=feats_flat,
                    class_prototypes=self.proto_memory(),
                    extra_negatives=extra_neg,
                    labels=labels_flat,
                    num_classes=self.proto_memory.num_classes,
                    num_prototypes_per_class=self.num_prototypes_per_class,
                    temperature=self.contrastive_temp,
                    ignore_index=self.ignore_index,
                )
            else:
                loss_c = prototype_contrastive_loss(
                    features=feats_flat,
                    prototypes=self.proto_memory(),
                    labels=labels_flat,
                    num_classes=self.proto_memory.num_classes,
                    num_prototypes_per_class=self.num_prototypes_per_class,
                    temperature=self.contrastive_temp,
                    ignore_index=self.ignore_index,
                )
            losses['loss_contrastive'] = self.contrastive_lambda * loss_c

        return losses

    def _align_affinity_inputs(self, feat, seg_label_resized,
                               pseudo_weight_resized):
        """Resize labels (and pseudo_weight) to match feat's spatial dims.

        AffinityBoundaryLoss requires features and seg_label at the same
        H, W. decoder_features[-1] is the 1/32-scale encoder output but
        seg_label_resized is at logits resolution (1/4). Without this
        alignment the loss errors with a size mismatch.
        """
        _, _, Hf, Wf = feat.shape
        if seg_label_resized.shape[-2:] != (Hf, Wf):
            aff_label = F.interpolate(
                seg_label_resized.float().unsqueeze(1),
                size=(Hf, Wf), mode='nearest').long().squeeze(1)
        else:
            aff_label = seg_label_resized
        if pseudo_weight_resized is None:
            return aff_label, None
        pw = pseudo_weight_resized
        if pw.dim() == 4:
            pw = pw.squeeze(1)  # (B, H, W)
        if pw.shape[-2:] != (Hf, Wf):
            pw = F.interpolate(
                pw.unsqueeze(1), size=(Hf, Wf),
                mode='bilinear', align_corners=False).squeeze(1)
        return aff_label, pw

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
            aff_label, aff_pw = self._align_affinity_inputs(
                feat, seg_label_resized, pseudo_weight_resized)
            return self.affinity_loss_fn(
                feat, aff_label, pseudo_weight=aff_pw)

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
            aff_label, aff_pw = self._align_affinity_inputs(
                feat, seg_label_resized, pseudo_weight_resized)
            affinity_l = self.affinity_loss_fn(
                feat, aff_label, pseudo_weight=aff_pw)
            hw = self.hybrid_binary_weight
            return hw * binary_l + (1 - hw) * affinity_l

        else:
            raise ValueError(f"Unknown boundary_loss_mode: {mode}")

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas, target_gt_semantic_seg=None):
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
            target_gt_semantic_seg (Tensor, optional): Hidden GT for the
                unlabeled images (loaded but never used for training). Used
                only as an evaluation oracle for ``pl_acc.*`` metrics.

        Returns:
            dict[str, Tensor]: Loss components.
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device
        has_dapcn = (
            self.boundary_lambda > 0
            or self.proto_lambda > 0
            or self.contrastive_lambda > 0
        )

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

        # === Step 3 (moved before step 2): Pseudo-label generation ===
        # We compute pseudo-labels BEFORE the labeled DAPCN losses so
        # that the v2 contrastive objective can use teacher-confident
        # pseudo-labels to update the prototype memory bank from
        # unlabeled features at the labeled step (step 2).
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        ema_logits = self.get_ema_model().encode_decode(
            target_img, target_img_metas)
        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        # Raw teacher label, captured BEFORE any prototype correction, for
        # evaluation-only comparison against the hidden unlabeled GT.
        raw_pseudo_label = ema_softmax.argmax(dim=1)
        self._last_proto_correction_stats = {}
        if (self.proto_correction_mode == 'class_prototype'
                and self.proto_memory is not None):
            counts = self.proto_memory.update_counts.reshape(
                self.proto_memory.num_classes, self.proto_memory.K)
            initialized = counts.gt(0).any(dim=1)
            log_vars.update({
                'proto_bank.initialized_classes': (
                    initialized.float().sum().item()),
                'proto_bank.ready': initialized.all().float().item(),
                'proto_bank.mean_updates_per_prototype': (
                    counts.float().mean().item()),
                'proto_bank.min_updates_per_class': (
                    counts.sum(dim=1).min().float().item()),
                'proto_bank.drift': self.proto_memory.last_drift.item(),
            })

        # Determine whether we need student features on target_img.
        # Shared by (a) v2 contrastive teacher-pseudo-label bank update,
        # (b) prototype-based pseudo-label correction. One no_grad
        # forward serves both — avoid duplicate work.
        need_contrastive_teacher = (
            self.contrastive_use_teacher
            and self.proto_memory is not None
            and self.contrastive_lambda > 0)
        correction_ready = (
            self.proto_memory is not None
            if self.proto_correction_mode == 'class_prototype'
            else self.dynamic_anchor is not None)
        need_proto_correction = (
            self.proto_correction and correction_ready
            and self.local_iter >= self.proto_correction_start_iter)
        need_dwpc = self._dwpc_needed(self.local_iter)
        student_unlab_feat = None
        if need_contrastive_teacher or need_proto_correction or need_dwpc:
            with torch.no_grad():
                student_unlab_feat = self.get_model().extract_feat(target_img)

        # === Step 3b: Prototype-based pseudo-label correction ===
        if need_proto_correction:
            ema_softmax = self._correct_pseudo_labels(
                target_img, target_img_metas, ema_softmax,
                student_feat=student_unlab_feat)
        if self._last_proto_correction_stats:
            log_vars.update({
                f'proto_corr.{name}': value.item()
                for name, value in self._last_proto_correction_stats.items()
            })

        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        # Evaluation-only: score raw vs prototype-corrected pseudo-labels
        # against the hidden unlabeled GT. This GT never enters any loss,
        # backward pass, or memory-bank update.
        log_vars.update(self._eval_pseudo_label_correction(
            raw_pseudo_label, pseudo_label, target_gt_semantic_seg))
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        scalar_pseudo_weight = torch.sum(ps_large_p).item() / ps_size

        warmup_scale = self._get_pseudo_weight_scale()

        # Teacher-confident mask feeds the (uncorrected) bank update at
        # step 2 — keep it on the raw teacher prediction.
        teacher_high_conf_mask = ps_large_p if need_contrastive_teacher else None

        # === Step 3b: DWPC dual-witness correction (flip + per-pixel w_i) ===
        # Replaces the scalar broadcast weight with a genuine per-pixel
        # weight and the teacher pseudo-label with the corrected label.
        if need_dwpc:
            corrected_label, pseudo_weight = self._dwpc_correct(
                ema_softmax, student_unlab_feat, self.local_iter,
                scalar_pseudo_weight)
            pseudo_label = corrected_label
            pseudo_weight = pseudo_weight * warmup_scale
        else:
            pseudo_weight = scalar_pseudo_weight * torch.ones(
                pseudo_prob.shape, device=dev)
            pseudo_weight = pseudo_weight * warmup_scale

        if self.psweight_ignore_top > 0:
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones(pseudo_weight.shape, device=dev)

        # === Step 2: DAPCN losses on labeled images ===
        if has_dapcn:
            src_logits = self.get_model().decode_head(src_feat)
            src_dapcn_losses = self._compute_dapcn_losses(
                src_logits, gt_semantic_seg, src_feat,
                is_labeled=True, pseudo_weight=None,
                unlabeled_decoder_features=(
                    student_unlab_feat if need_contrastive_teacher else None),
                teacher_pseudo_label=(
                    pseudo_label if need_contrastive_teacher else None),
                teacher_high_conf_mask=teacher_high_conf_mask)
            if src_dapcn_losses:
                src_dapcn_loss, src_dapcn_log = self._parse_losses(
                    src_dapcn_losses)
                log_vars.update(src_dapcn_log)
                src_dapcn_loss.backward()

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
        target_loss.backward(retain_graph=has_dapcn)

        # === Step 6: DAPCN losses on mixed (unlabeled) ===
        if has_dapcn:
            tgt_logits = self.get_model().decode_head(tgt_feat)
            tgt_dapcn_losses = self._compute_dapcn_losses(
                tgt_logits, mixed_lbl, tgt_feat,
                is_labeled=False, pseudo_weight=pseudo_weight)
            if tgt_dapcn_losses:
                tgt_dapcn_loss, tgt_dapcn_log = self._parse_losses(
                    tgt_dapcn_losses)
                log_vars.update(add_prefix(tgt_dapcn_log, 'mix'))
                tgt_dapcn_loss.backward()

        # Refresh the bank-drift metric once per iteration (after the
        # labeled-step update has run), so proto_bank.drift reflects the
        # bank's movement over the last full iteration.
        if self.proto_memory is not None:
            self.proto_memory.refresh_drift()

        self.local_iter += 1
        return log_vars
