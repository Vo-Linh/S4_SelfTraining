# ---------------------------------------------------------------
# DAFormerDAPCNHead: Supervised segmentation with DAPCN losses
# Integrates boundary-aware loss, dynamic anchor prototypes,
# DAPG loss, and a persistent prototype memory bank with
# contrastive regularisation — all within the standard
# MMSegmentation decode-head interface (no UDA decorator).
#
# Implementation: thin subclass that inherits DAPCNHeadMixin and
# delegates all auxiliary-loss computation to the mixin.  Only the
# DAFormer-specific fused-feature reconstruction lives here.
# ---------------------------------------------------------------

import torch
import torch.nn.functional as F

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.daformer_head import DAFormerHead
from mmseg.models.decode_heads.dapcn_head_mixin import DAPCNHeadMixin


@HEADS.register_module()
class DAFormerDAPCNHead(DAPCNHeadMixin, DAFormerHead):
    """DAFormer decode head augmented with DAPCN auxiliary losses.

    All auxiliary-loss logic lives in :class:`DAPCNHeadMixin`.  This
    subclass only forwards constructor arguments into ``init_dapcn``
    and reconstructs the fused decoder feature (DAFormer-specific) so
    the mixin's contrastive path can operate on it.

    Loss budget (total = CE + auxiliary):
        L = L_ce
            + boundary_lambda   * L_boundary
            + proto_lambda      * L_dapg
            + contrastive_lambda * L_contrastive
    """

    def __init__(self,
                 # --- DAPCN loss weights ---
                 boundary_lambda=0.3,
                 proto_lambda=0.1,
                 contrastive_lambda=0.1,
                 # --- DA placement ---
                 da_position='before_fusion',
                 da_feature_dim=None,
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
                 **kwargs):
        super().__init__(**kwargs)

        self.init_dapcn(
            da_position=da_position,
            da_feature_dim=da_feature_dim,
            boundary_lambda=boundary_lambda,
            proto_lambda=proto_lambda,
            contrastive_lambda=contrastive_lambda,
            boundary_mode=boundary_mode,
            boundary_loss_mode=boundary_loss_mode,
            hybrid_binary_weight=hybrid_binary_weight,
            contrastive_temperature=contrastive_temperature,
            contrastive_sample_ratio=contrastive_sample_ratio,
            warmup_iters=warmup_iters,
            num_prototypes_per_class=num_prototypes_per_class,
            prototype_ema=prototype_ema,
            prototype_init_strategy=prototype_init_strategy,
            dynamic_anchor=dynamic_anchor,
            dapg_loss=dapg_loss,
            affinity_loss=affinity_loss,
        )

    # ------------------------------------------------------------------
    # Override forward_train to inject DAPCN auxiliary losses
    # ------------------------------------------------------------------
    def forward_train(self, inputs, img_metas, gt_semantic_seg,
                      train_cfg, seg_weight=None):
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)

        fused_feature = self._get_fused_feature(inputs)
        losses.update(self.dapcn_forward_train(
            inputs, seg_logits, gt_semantic_seg, fused_feature))

        return losses

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_fused_feature(self, inputs):
        """Re-derive the fused decoder feature (before cls_seg).

        Mirrors ``DAFormerHead.forward`` but stops before the
        classification convolution, returning the (B, channels, H, W)
        representation that the memory bank stores.
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
