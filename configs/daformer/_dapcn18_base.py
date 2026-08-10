log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
custom_hooks = [
    dict(
        type='EmailNotificationHook',
        subject_prefix='[daformer]',
        milestone_every=1,
        log_tail_lines=50,
        nan_guard=True)
]
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(type='mit_b5', style='pytorch'),
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=dict(type='BN', requires_grad=True))),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(work_dir='work_dirs/T_dapcn_oem_mitb5_1_8'),
    test_cfg=dict(mode='whole'))
dataset_type = 'OpenEarthMapDataset'
data_root = '/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
labeled_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomRotate', prob=0.5, degree=180, pad_val=0, seg_pad_val=255),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
unlabeled_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='SSLDataset',
        labeled=dict(
            type='OpenEarthMapDataset',
            data_root='/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/',
            img_dir='images/train',
            ann_dir='annotations/train',
            split='train_1_8_labeled.txt',
            ignore_index=255,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='Resize',
                    img_scale=(1024, 1024),
                    ratio_range=(0.5, 2.0),
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_size=(512, 512),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='RandomRotate',
                    prob=0.5,
                    degree=180,
                    pad_val=0,
                    seg_pad_val=255),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ]),
        unlabeled=dict(
            type='OpenEarthMapDataset',
            data_root='/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/',
            img_dir='images/train',
            ann_dir='annotations/train',
            split='train_1_8_unlabeled.txt',
            ignore_index=255,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='Resize',
                    img_scale=(1024, 1024),
                    ratio_range=(0.5, 2.0),
                    keep_ratio=True),
                dict(type='RandomCrop', crop_size=(512, 512)),
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ])),
    val=dict(
        type='OpenEarthMapDataset',
        data_root='/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/',
        img_dir='images/val',
        ann_dir='annotations/val',
        split='val_2000_fixed.txt',
        ignore_index=255,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(1024, 1024),
                        keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size=(512, 512),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='OpenEarthMapDataset',
        data_root='/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/',
        img_dir='images/val',
        ann_dir='annotations/val',
        split='val_2000_fixed.txt',
        ignore_index=255,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(1024, 1024),
                        keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size=(512, 512),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
uda = dict(
    type='DAPCN_SSL',
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=1000,
    pseudo_label_warmup_iters=1000,
    proto_correction=True,
    proto_correction_alpha=0.5,
    proto_correction_start_iter=1000,
    boundary_lambda=0.5,
    proto_lambda=0.1,
    contrastive_lambda=0.1,
    contrastive_temp=0.07,
    num_prototypes_per_class=1,
    prototype_ema=0.999,
    prototype_init_strategy='zeros',
    boundary_loss_mode='affinity',
    boundary_mode='sobel',
    apply_boundary_on_target=True,
    apply_proto_on_target=True,
    hybrid_binary_weight=0.5,
    ignore_index=255,
    dynamic_anchor=dict(
        type='DynamicAnchorModule',
        max_groups=96,
        temperature=0.1,
        num_iters=3,
        init_method='xavier',
        min_quality=0.1,
        use_quality_gate=True,
        use_mask_predictor=False,
        ema_decay=0.0),
    dapg_loss=dict(
        type='DAPGLoss',
        margin=0.3,
        lambda_inter=0.5,
        lambda_quality=0.1,
        loss_weight=1.0),
    affinity_loss=dict(
        type='AffinityBoundaryLoss',
        temperature=0.5,
        scale=2,
        num_neighbors=4,
        ignore_index=255,
        loss_weight=1.0))
use_ddp_wrapper = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            {
                'head': dict(lr_mult=10.0),
                'dynamic_anchor.prototypes': dict(
                    lr_mult=10.0, decay_mult=0.0),
                'dynamic_anchor.quality_net': dict(lr_mult=10.0),
                'proto_to_decoder': dict(lr_mult=10.0, decay_mult=0.0),
                'pos_block': dict(decay_mult=0.0),
                'norm': dict(decay_mult=0.0)
            })))
optimizer_config = None
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
seed = 0
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
evaluation = dict(interval=4000, metric='mIoU')
name = 'ssl_oem_dapcn_daformer_mitb5'
exp = 'ssl_oem'
name_dataset = 'openearthmap_ssl_500_3500'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dapcn_ssl_affinity_contrastive'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
work_dir = 'work_dirs/T_dapcn_oem_mitb5_1_8'
gpu_ids = [0]
