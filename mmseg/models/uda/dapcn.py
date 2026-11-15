# ---------------------------------------------------------------
# DAPCN: Dynamic Attention-based Prototype Clustering Network
# for Unsupervised Domain Adaptation in Satellite Image Segmentation
#
# Core components:
#   1. Affinity-based boundary loss (orientation-invariant)
#   2. Dynamic anchor prototype grouping (DAPGLoss)
#   3. Self-training with EMA teacher and ClassMix
# ---------------------------------------------------------------

import random
from copy import deepcopy

import numpy as np
import torch
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
class DAPCN(UDADecorator):
    """Dynamic Attention-based Prototype Clustering Network for UDA.

    DAPCN integrates geometry-driven dynamic anchor discovery with
    boundary-aware supervision into a unified UDA framework, designed
    for satellite image segmentation where:
      - Land categories have no repeatable shape
      - Multiple categories co-occur in a single image
      - Severe class imbalance with few labels for rare classes

    Args:
        boundary_lambda (float): Weight for boundary loss. Default: 0.3.
        proto_lambda (float): Weight for prototype grouping loss. Default: 0.1.
        boundary_mode (str): Boundary extraction mode
            ('sobel', 'laplacian', 'diff'). Default: 'sobel'.
        apply_boundary_on_target (bool): Apply boundary loss on target
            domain. Default: True.
        apply_proto_on_target (bool): Apply prototype loss on target
            domain. Default: True.
        boundary_loss_mode (str): Boundary loss mode
            ('binary', 'affinity', 'hybrid'). Default: 'affinity'.
        hybrid_binary_weight (float): Binary weight in hybrid mode.
            Default: 0.5.
        ignore_index (int): Ignore index for loss computation.
            Default: 255.
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
                 # Sub-module configs (MMSeg registry style)
                 dynamic_anchor=None,
                 dapg_loss=None,
                 affinity_loss=None,
                 **cfg):
        super(DAPCN, self).__init__(**cfg)

        # Self-training parameters
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
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

        # EMA teacher model
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        # Store sub-module configs for lazy init
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
        """Build DynamicAnchorModule, auto-detecting feature_dim."""
        decode_head_cfg = self.get_model().decode_head
        in_channels = decode_head_cfg.in_channels
        if isinstance(in_channels, (list, tuple)):
            in_channels = in_channels[-1]

        if self._dynamic_anchor_cfg is not None:
            da_cfg = deepcopy(self._dynamic_anchor_cfg)
            da_cfg.setdefault('feature_dim', in_channels)
            self.dynamic_anchor = MODELS.build(da_cfg)
        else:
            from mmseg.models.uda.dynamic_anchor import DynamicAnchorModule
            self.dynamic_anchor = DynamicAnchorModule(
                feature_dim=in_channels,
                max_groups=64,
                temperature=0.1,
                num_iters=3,
                init_method='xavier',
                min_quality=0.1,
            )

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
        """Single training iteration."""
        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def _compute_dapcn_losses(self, logits, seg_label, decoder_features,
                              is_source=True, pseudo_weight=None):
        """Compute DAPCN losses (boundary + prototype grouping).

        Args:
            logits (Tensor): Segmentation logits (N, C, H, W).
            seg_label (Tensor): Labels (N, H, W) or (N, 1, H, W).
            decoder_features (list[Tensor]): Multi-scale decoder features.
            is_source (bool): Whether this is source domain.
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
            if is_source or self.apply_boundary_on_target:
                boundary_loss = self._compute_boundary_loss(
                    logits, seg_label_resized, decoder_features,
                    is_source, pseudo_weight_resized)
                losses['loss_boundary'] = self.boundary_lambda * boundary_loss

        # --- Prototype grouping loss ---
        if self.proto_lambda > 0:
            if is_source or self.apply_proto_on_target:
                feat = (decoder_features[-1] if isinstance(
                    decoder_features, list) else decoder_features)
                B, C, Hf, Wf = feat.shape
                feats_flat = feat.permute(0, 2, 3, 1).reshape(-1, C)

                assign, proto, quality = self.dynamic_anchor(feat)
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
        the affinity loss is the last encoder stage (stride 32).
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
                               decoder_features, is_source,
                               pseudo_weight_resized):
        """Compute boundary loss based on configured mode."""
        mode = self.boundary_loss_mode

        if mode == 'binary':
            b_pred = extract_boundary_map(logits, mode=self.boundary_mode)
            b_gt = compute_boundary_gt(
                seg_label_resized, ignore_index=self.ignore_index)
            if not is_source and pseudo_weight_resized is not None:
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
            if not is_source and pseudo_weight_resized is not None:
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
        """Forward function for training.

        Training pipeline:
          1. Source supervised loss (cross-entropy)
          2. DAPCN losses on source (boundary + prototype)
          3. EMA teacher generates pseudo-labels on target
          4. ClassMix: source patches pasted onto target
          5. Mixed supervised loss
          6. DAPCN losses on target (boundary + prototype)

        Args:
            img (Tensor): Source images.
            img_metas (list[dict]): Source image info.
            gt_semantic_seg (Tensor): Source ground truth labels.
            target_img (Tensor): Target images.
            target_img_metas (list[dict]): Target image info.

        Returns:
            dict[str, Tensor]: Loss components.
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device
        has_dapcn = self.boundary_lambda > 0 or self.proto_lambda > 0

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

        # === Step 1: Source supervised loss ===
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=has_dapcn)

        # === Step 2: DAPCN losses on source ===
        if has_dapcn:
            src_logits = self.get_model().decode_head(src_feat)
            src_dapcn_losses = self._compute_dapcn_losses(
                src_logits, gt_semantic_seg, src_feat,
                is_source=True, pseudo_weight=None)
            if src_dapcn_losses:
                src_dapcn_loss, src_dapcn_log = self._parse_losses(
                    src_dapcn_losses)
                log_vars.update(src_dapcn_log)
                src_dapcn_loss.backward()

        # === Step 3: Pseudo-label generation ===
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        ema_logits = self.get_ema_model().encode_decode(
            target_img, target_img_metas)
        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)

        if self.psweight_ignore_top > 0:
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones(pseudo_weight.shape, device=dev)

        # === Step 4: ClassMix augmentation ===
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

        # === Step 6: DAPCN losses on target ===
        if has_dapcn:
            tgt_logits = self.get_model().decode_head(tgt_feat)
            tgt_dapcn_losses = self._compute_dapcn_losses(
                tgt_logits, mixed_lbl, tgt_feat,
                is_source=False, pseudo_weight=pseudo_weight)
            if tgt_dapcn_losses:
                tgt_dapcn_loss, tgt_dapcn_log = self._parse_losses(
                    tgt_dapcn_losses)
                log_vars.update(add_prefix(tgt_dapcn_log, 'mix'))
                tgt_dapcn_loss.backward()

        self.local_iter += 1
        return log_vars
