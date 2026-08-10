# ---------------------------------------------------------------
# PyramidMamba (ResNeXt101-32x16d SWSL encoder + EfficientPyramidMamba decoder)
#
# Production recipe from the supervised repo, matching
# _base_/models/unetformer_resnext101.py on the encoder so the decoder is
# the only variable in the Table XIII comparison.
#
# The head takes TWO scales (stride-4 and stride-32) via in_index=(0, 3).
# BaseDecodeHead.channels is set internally to decode_channels // 2 = 128,
# so under DAPCN_SSL: feature_dim = in_channels[-1] = 2048 and
# proto_to_decoder = Linear(2048 -> 128).
#
# NOTE the classifier width differs from UNetFormer-ResNeXt101 (128 vs 256)
# because PyramidMamba halves decode_channels internally. That is the
# supervised repo's production recipe for each decoder; if you want the two
# decoders width-matched for a stricter Table XIII, set decode_channels=512
# here (-> channels=256).
#
# Requires mamba_ssm + causal_conv1d (installed on the box).
# ---------------------------------------------------------------

norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='TIMMBackbone',
        model_name='resnext101_32x16d.fb_swsl_ig1b_ft_in1k',
        features_only=True,
        pretrained=True,
        out_indices=(1, 2, 3, 4),
    ),
    decode_head=dict(
        type='PyramidMambaHead',
        in_channels=[256, 2048],
        in_index=(0, 3),
        encoder_channels=(256, 2048),
        channels=128,           # cosmetic; effective width = decode_channels // 2
        decode_channels=256,
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
