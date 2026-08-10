# ---------------------------------------------------------------
# PyramidMamba (ResNet18 encoder + EfficientPyramidMamba decoder)
#
# Ported from the supervised SatelliteImageSegSupevised repo so DAPCN-SSL
# can run on a Mamba decoder. Deliberately matched to
# _base_/models/unetformer_r18.py: same TIMM ResNet18 encoder and the same
# effective classifier width (channels=64), so an architecture comparison
# under DAPCN-SSL changes only the decoder.
#
# The head consumes TWO scales (stride-4 and stride-32) via in_index=(0, 3);
# BaseDecodeHead.channels is set internally to decode_channels // 2 = 64.
#
# Requires `mamba_ssm` + `causal_conv1d` (CUDA wheels) in the venv -- they are
# imported lazily, so `import mmseg` still works without them, but building
# this model does not.
# ---------------------------------------------------------------

norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='TIMMBackbone',
        model_name='resnet18.fb_swsl_ig1b_ft_in1k',
        features_only=True,
        pretrained=True,
        out_indices=(1, 2, 3, 4),
    ),
    decode_head=dict(
        type='PyramidMambaHead',
        # stride-4 (64ch) and stride-32 (512ch) of ResNet18
        in_channels=[64, 512],
        in_index=(0, 3),
        encoder_channels=(64, 512),
        channels=64,            # cosmetic; effective width = decode_channels // 2
        decode_channels=128,
        last_feat_size=16,      # 512 crop // 32
        d_state=16,
        d_conv=4,
        expand=2,
        num_classes=9,
        dropout_ratio=0.1,
        input_transform='multiple_select',
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)
