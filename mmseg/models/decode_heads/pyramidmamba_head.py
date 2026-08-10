# ---------------------------------------------------------------
# PyramidMamba decode head for mmsegmentation.
#
# Wraps the ported GeoSeg PyramidMamba decoder (see
# pyramidmamba_modules.py) as an mmseg BaseDecodeHead so it plugs into the
# stock EncoderDecoder + TIMMBackbone pipeline, exactly like UNetFormerHead.
#
# The backbone (e.g. ResNeXt101_32x16d via TIMMBackbone, out_indices=(1,2,3,4))
# emits four feature maps; this head selects the stride-4 (shallow) and
# stride-32 (deep) features via in_index, runs the Pyramid-Pooling + Mamba
# decoder, and classifies with cls_seg.
#
# PyramidMambaDAPCNHead adds the repo's DAPCN auxiliary losses through
# DAPCNHeadMixin, mirroring UNetFormerDAPCNHead.
# ---------------------------------------------------------------

from mmseg.models.builder import HEADS
from .decode_head import BaseDecodeHead
from .pyramidmamba_modules import PyramidMambaDecoder


@HEADS.register_module()
class PyramidMambaHead(BaseDecodeHead):
    """PyramidMamba decode head (EfficientPyramidMamba decoder).

    Consumes two backbone features — a shallow (stride-4) and a deep
    (stride-32) map selected through ``in_index`` — and fuses them with a
    Pyramid-Pooling + Mamba decoder. The fused feature has
    ``decode_channels // 2`` channels at the input resolution; ``cls_seg``
    produces the ``num_classes`` logits.

    Args:
        in_channels (list[int]): Channels of the selected backbone features,
            i.e. ``[shallow_ch, deep_ch]`` (e.g. ``[256, 2048]``).
        channels (int): Kept for API compatibility; the effective
            classifier width is ``decode_channels // 2``.
        encoder_channels (tuple[int]): Same as ``in_channels`` as a tuple
            ``(shallow_ch, deep_ch)``; ``[0]`` feeds the skip ``pre_conv`` and
            ``[-1]`` feeds the Mamba block.
        decode_channels (int): Decoder hidden width. Default: 128.
        last_feat_size (int): Deepest-feature spatial size at the train crop
            (= crop_size // 32). Default: 16 (for a 512 crop).
        d_state, d_conv, expand: Mamba SSM hyper-parameters.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 encoder_channels=(256, 2048),
                 decode_channels=128,
                 last_feat_size=16,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 in_index=(0, 3),
                 input_transform='multiple_select',
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                                  loss_weight=1.0),
                 backbone_nhwc=False,
                 ignore_index=255,
                 init_cfg=dict(type='Normal', std=0.01,
                               override=dict(name='conv_seg'))):
        super().__init__(
            in_channels=in_channels,
            channels=decode_channels // 2,
            num_classes=num_classes,
            in_index=in_index,
            input_transform=input_transform,
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_corners=align_corners,
            loss_decode=loss_decode,
            ignore_index=ignore_index,
            init_cfg=init_cfg,
        )
        self.encoder_channels = tuple(encoder_channels)
        self.decode_channels = decode_channels
        self.last_feat_size = last_feat_size
        # NHWC backbones (e.g. timm Swin) emit (B,H,W,C); permute to NCHW in _decode.
        self.backbone_nhwc = backbone_nhwc

        self.decoder = PyramidMambaDecoder(
            encoder_channels=self.encoder_channels,
            decoder_channels=decode_channels,
            last_feat_size=last_feat_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def _decode(self, inputs):
        """Run the PyramidMamba decoder, returning the fused feature.

        Args:
            inputs (list[Tensor]): Raw multi-level backbone features.

        Returns:
            Tensor: Fused feature (B, decode_channels // 2, H, W) at the
            input image resolution (before cls_seg).
        """
        feats = self._transform_inputs(inputs)  # [shallow (x0), deep (x3)]
        if self.backbone_nhwc:
            # timm Swin & co. output (B, H, W, C); the decoder expects NCHW.
            feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
        x0, x3 = feats[0], feats[-1]
        return self.decoder(x0, x3)

    def forward(self, inputs):
        """Forward pass returning segmentation logits."""
        return self.cls_seg(self._decode(inputs))

    def _fuse_features(self, inputs):
        """Return the fused decoder feature (pre-``cls_seg``).

        Mirrors ``DAFormerHead._fuse_features`` so the ``DAPCN_SSL`` UDA
        wrapper can source anchor features from the decoder's fused space
        (``anchor_after_fusion=True``).

        Note: unlike UNetFormer (H/4), this decoder emits its fused map at the
        FULL input resolution, so anchoring after fusion runs the EM clustering
        over B*H*W pixels. Prefer the default Solution 1 (anchor on the
        stride-32 encoder feature) for PyramidMamba.
        """
        return self._decode(inputs)
