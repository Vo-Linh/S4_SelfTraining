# ---------------------------------------------------------------
# PyramidMamba decoder modules.
#
# Ported (near-verbatim) from GeoSeg's EfficientPyramidMamba decoder:
#   https://github.com/WangLibo1995/GeoSeg/blob/main/geoseg/models/PyramidMamba.py
#   "PyramidMamba: Rethinking Pyramid Feature Fusion with Selective Space
#    State Model for Semantic Segmentation of Remote Sensing Imagery"
#    (arXiv:2406.10828)
#
# The decoder fuses a shallow (stride-4) and a deep (stride-32) backbone
# feature using a Pyramid-Pooling + Mamba (PPM) block, three upsampling
# stages, a skip connection, and a head. Unlike the original GeoSeg head,
# ``PyramidMambaDecoder`` returns the fused feature *before* the final
# class projection (channels = ``decoder_channels // 2``); the mmseg
# ``BaseDecodeHead.cls_seg`` produces the ``num_classes`` logits instead,
# keeping the head idiomatic and compatible with the DAPCN mixin.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
try:
    from timm.layers import trunc_normal_
except ImportError:  # older timm
    from timm.models.layers import trunc_normal_


def _build_mamba(d_model, d_state, d_conv, expand):
    """Lazily import and build the Mamba SSM block.

    ``mamba_ssm`` (and its ``causal_conv1d`` companion) are CUDA-compiled
    packages that are only required when a PyramidMamba model is actually
    instantiated. Importing them lazily keeps ``import mmseg`` working in
    environments where Mamba is not installed (e.g. CPU-only tooling).
    """
    try:
        from mamba_ssm import Mamba
    except ImportError as e:  # pragma: no cover - depends on env
        raise ImportError(
            'PyramidMamba requires the `mamba_ssm` package (and '
            '`causal_conv1d`). Install matching CUDA wheels, e.g.\n'
            '  pip install causal-conv1d==1.2.0.post2 mamba-ssm==1.2.0.post1\n'
            'and a compatible `transformers<4.41`. See '
            'https://github.com/state-spaces/mamba for details.'
        ) from e
    return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      bias=bias, dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      bias=bias, dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      bias=bias, dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class MambaLayer(nn.Module):
    """Pyramid-Pooling + Mamba (PPM) layer.

    Builds a set of pyramid-pooled context features (sizes derived from
    ``last_feat_size``), concatenates them with the residual input, and
    sweeps the flattened sequence with a Mamba selective-scan block.
    """

    def __init__(self, in_chs=512, dim=128, d_state=16, d_conv=4, expand=2,
                 last_feat_size=16):
        super().__init__()
        pool_scales = self.generate_arithmetic_sequence(
            1, last_feat_size, max(1, last_feat_size // 4))
        self.pool_len = len(pool_scales)
        self.pool_layers = nn.ModuleList()
        self.pool_layers.append(nn.Sequential(
            ConvBNReLU(in_chs, dim, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        ))
        for pool_scale in pool_scales[1:]:
            self.pool_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_scale),
                ConvBNReLU(in_chs, dim, kernel_size=1)
            ))
        self.mamba = _build_mamba(
            d_model=dim * self.pool_len + in_chs,  # Model dimension d_model
            d_state=d_state,    # SSM state expansion factor
            d_conv=d_conv,      # Local convolution width
            expand=expand,      # Block expansion factor
        )

    def forward(self, x):  # B, C, H, W
        res = x
        B, C, H, W = res.shape
        ppm_out = [res]
        for p in self.pool_layers:
            pool_out = p(x)
            pool_out = F.interpolate(
                pool_out, (H, W), mode='bilinear', align_corners=False)
            ppm_out.append(pool_out)
        x = torch.cat(ppm_out, dim=1)
        _, chs, _, _ = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c', b=B, c=chs, h=H, w=W)
        x = self.mamba(x)
        x = x.transpose(2, 1).view(B, chs, H, W)
        return x

    def generate_arithmetic_sequence(self, start, stop, step):
        sequence = []
        for i in range(start, stop, step):
            sequence.append(i)
        return sequence


class ConvFFN(nn.Module):
    def __init__(self, in_ch=128, hidden_ch=512, out_ch=128, drop=0.):
        super(ConvFFN, self).__init__()
        self.conv = ConvBNReLU(in_ch, in_ch, kernel_size=3)
        self.fc1 = Conv(in_ch, hidden_ch, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = Conv(hidden_ch, out_ch, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """PPM-Mamba block followed by a convolutional FFN."""

    def __init__(self, in_chs=512, dim=128, hidden_ch=512, out_ch=128, drop=0.1,
                 d_state=16, d_conv=4, expand=2, last_feat_size=16):
        super(Block, self).__init__()
        self.mamba = MambaLayer(
            in_chs=in_chs, dim=dim, d_state=d_state, d_conv=d_conv,
            expand=expand, last_feat_size=last_feat_size)
        self.conv_ffn = ConvFFN(
            in_ch=dim * self.mamba.pool_len + in_chs,
            hidden_ch=hidden_ch, out_ch=out_ch, drop=drop)

    def forward(self, x):
        x = self.mamba(x)
        x = self.conv_ffn(x)
        return x


class PyramidMambaDecoder(nn.Module):
    """PyramidMamba decoder returning the fused feature before classification.

    Args:
        encoder_channels (tuple[int]): (shallow_ch, deep_ch). ``encoder_channels[0]``
            is the stride-4 feature channel, ``encoder_channels[-1]`` the
            stride-32 feature channel.
        decoder_channels (int): Decoder hidden width. Default: 128.
        last_feat_size (int): Spatial size of the deepest feature at train
            crop (= crop_size // 32). Sets the pyramid pool scales. Default: 16.
        d_state, d_conv, expand: Mamba SSM hyper-parameters.

    forward(x0, x3) -> Tensor of shape (B, decoder_channels // 2, H, W) at the
    input image resolution.
    """

    def __init__(self, encoder_channels=(256, 2048), decoder_channels=128,
                 last_feat_size=16, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.b3 = Block(
            in_chs=encoder_channels[-1], dim=decoder_channels,
            hidden_ch=4 * decoder_channels, out_ch=decoder_channels,
            d_state=d_state, d_conv=d_conv, expand=expand,
            last_feat_size=last_feat_size)
        self.up_conv = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels),
            nn.Upsample(scale_factor=2),
            ConvBNReLU(decoder_channels, decoder_channels),
            nn.Upsample(scale_factor=2),
            ConvBNReLU(decoder_channels, decoder_channels),
            nn.Upsample(scale_factor=2),
        )
        self.pre_conv = ConvBNReLU(encoder_channels[0], decoder_channels)
        # NOTE: trailing Conv(decoder_channels // 2 -> num_classes) of the
        # original GeoSeg head is intentionally dropped; mmseg's cls_seg
        # (conv_seg) produces the class logits from these features.
        self.head = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels // 2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(decoder_channels // 2, decoder_channels // 2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self._init_decoder_weights()

    def _init_decoder_weights(self):
        """Initialize conv/norm/linear layers WITHOUT touching the Mamba block.

        ``Mamba`` carries a carefully tuned internal initialization (notably
        ``dt_proj.bias`` ~= inverse-softplus, ``A_log``, ``D``) that governs the
        SSM dynamics. A blanket ``self.apply(self._init_weights)`` would recurse
        into the Mamba's ``nn.Linear`` layers and zero ``dt_proj.bias`` /
        reset ``dt_proj.weight``, blowing up the SSM timestep and crippling the
        global-context path. Skip the Mamba subtree to preserve that init.
        """
        def _recurse(module):
            for child in module.children():
                # Skip any module defined by mamba_ssm (the Mamba SSM block)
                # and everything inside it.
                if type(child).__module__.startswith('mamba_ssm'):
                    continue
                self._init_weights(child)
                _recurse(child)
        _recurse(self)

    def forward(self, x0, x3):
        x3 = self.b3(x3)
        x3 = self.up_conv(x3)
        skip = self.pre_conv(x0)
        # up_conv uses fixed scale_factor=2 upsampling, so the 8x-upsampled deep
        # feature only matches the stride-4 skip exactly when the input H/W are
        # divisible by 32. Under whole-image inference the eval image can have
        # an arbitrary (resized) size, leaving the two off by a few pixels —
        # align before the residual add. No-op at the train crop (512 -> /32).
        if x3.shape[-2:] != skip.shape[-2:]:
            x3 = F.interpolate(
                x3, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = x3 + skip
        x = self.head(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
