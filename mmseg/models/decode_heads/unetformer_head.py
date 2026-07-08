import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.models.builder import HEADS
from mmseg.ops import resize
from .decode_head import BaseDecodeHead
from .unetformer_modules import (
    ConvBNReLU, ConvBN, Conv, SeparableConvBNReLU, SeparableConvBN,
    Mlp, GlobalLocalAttention, Block, WF, FeatureRefinementHead,
)


@HEADS.register_module()
class UNetFormerHead(BaseDecodeHead):
    """UNetFormer decode head with GlobalLocalAttention decoder.
    
    Paper-faithful implementation: CNN encoder + Transformer decoder.
    
    Args:
        encoder_channels (tuple): Channel dims from backbone (64, 128, 256, 512 for B0).
        decode_channels (int): Decoder hidden channels (64 for B0).
        window_size (int): Window attention size. Default: 8.
        num_heads (int): Number of attention heads. Default: 8.
        mlp_ratio (float): MLP hidden dim ratio. Default: 4.0.
        drop_path_rate (float): DropPath rate. Default: 0.1.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 window_size=8,
                 num_heads=8,
                 mlp_ratio=4.0,
                 drop_path_rate=0.1,
                 in_index=[0, 1, 2, 3],
                 input_transform='multiple_select',
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 init_cfg=dict(type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super().__init__(
            in_channels=in_channels,
            channels=decode_channels,
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
            sampler=sampler,
            init_cfg=init_cfg,
        )
        self.encoder_channels = encoder_channels
        self.decode_channels = decode_channels
        self.align_corners = align_corners

        # Decoder components (paper architecture)
        # pre_conv4: project deepest features to decode_channels
        self.pre_conv4 = ConvBN(encoder_channels[3], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=num_heads, window_size=window_size,
                        mlp_ratio=mlp_ratio, drop_path=drop_path_rate)

        # WF3 + Block3: weighted fusion with res3 + transformer
        self.wf3 = WF(in_channels=encoder_channels[2], decode_channels=decode_channels)
        self.b3 = Block(dim=decode_channels, num_heads=num_heads, window_size=window_size,
                        mlp_ratio=mlp_ratio, drop_path=drop_path_rate)

        # WF2 + Block2: weighted fusion with res2 + transformer
        self.wf2 = WF(in_channels=encoder_channels[1], decode_channels=decode_channels)
        self.b2 = Block(dim=decode_channels, num_heads=num_heads, window_size=window_size,
                        mlp_ratio=mlp_ratio, drop_path=drop_path_rate)

        # FeatureRefinementHead: final refinement with res1
        self.frh = FeatureRefinementHead(in_channels=encoder_channels[0], decode_channels=decode_channels)

    def _decode(self, inputs):
        """Run the GLA decoder, returning the fused feature before cls_seg.

        Args:
            inputs (list[Tensor]): 4 feature maps from backbone [res1, res2, res3, res4].

        Returns:
            Tensor: Fused feature (B, decode_channels, H/4, W/4).
        """
        res1, res2, res3, res4 = inputs[0], inputs[1], inputs[2], inputs[3]

        x = self.pre_conv4(res4)       # (B, decode_channels, H/32, W/32)
        x = self.b4(x)                 # transformer at H/32
        x = self.wf3(x, res3)          # upsample + weighted fusion with res3 -> H/16
        x = self.b3(x)                 # transformer at H/16
        x = self.wf2(x, res2)          # upsample + weighted fusion with res2 -> H/8
        x = self.b2(x)                 # transformer at H/8
        x = self.frh(x, res1)          # feature refinement with res1 -> H/4
        return x

    def _fuse_features(self, inputs):
        """Return the fused decoder feature (pre-``cls_seg``).

        Mirrors ``DAFormerHead._fuse_features`` so the ``DAPCN_SSL`` UDA
        wrapper can extract anchor features in the decoder's fused space
        (Solution 2 / ``anchor_after_fusion=True``). ``inputs`` are the raw
        multi-scale backbone features; ``_decode`` indexes the 4 stages
        directly, matching ``forward``.

        Args:
            inputs (list[Tensor]): 4 backbone feature maps.

        Returns:
            Tensor: Fused feature (B, decode_channels, H/4, W/4).
        """
        return self._decode(inputs)

    def forward(self, inputs):
        """Forward pass.

        Args:
            inputs (list[Tensor]): 4 feature maps from backbone [res1, res2, res3, res4].

        Returns:
            Tensor: Segmentation logits (B, num_classes, H/4, W/4).
        """
        x = self._decode(inputs)
        return self.cls_seg(x)
