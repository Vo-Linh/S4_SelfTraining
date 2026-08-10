# ---------------------------------------------------------------
# UNetFormer (ResNeXt101-32x16d SWSL encoder + GlobalLocalAttention decoder)
#
# Production recipe from the supervised repo (the one used for the OEM
# results there), scaled up from unetformer_r18.py:
#   encoder feats  (64,128,256,512)  ->  (256,512,1024,2048)
#   decode_channels        64        ->  256
#
# Under DAPCN_SSL this makes feature_dim = in_channels[-1] = 2048 and
# proto_to_decoder = Linear(2048 -> 256).
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
        type='UNetFormerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        channels=256,
        num_classes=9,
        encoder_channels=(256, 512, 1024, 2048),
        decode_channels=256,
        window_size=8,
        num_heads=8,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
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
