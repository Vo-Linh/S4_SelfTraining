# ---------------------------------------------------------------
# Semi-Supervised OpenEarthMap Dataset Configuration
#
# Pairs:
#   - labeled subset    = train_500_fixed.txt   (500 samples w/ GT)
#   - unlabeled subset  = train_3500_fixed.txt  (3500 samples; the
#                         labeled 500 are a subset and reappear here
#                         — they still benefit from consistency reg)
#   - val               = val_2000_fixed.txt
#
# 9 OpenEarthMap land-cover classes, 512x512 crops, ignore_index=255.
# Mirrors /home/ubuntu/SatelliteImageSegSupevised/configs/_base_/
# datasets/openearthmap_val2000.py for the supervised path.
# ---------------------------------------------------------------

dataset_type = 'OpenEarthMapDataset'
data_root = '/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

crop_size = (512, 512)

labeled_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024),
         ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=180,
         pad_val=0, seg_pad_val=255),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# Unlabeled images still have GT on disk (loaded for shape only by
# the pipeline); training never reads them — pseudo-labels from the
# EMA teacher replace them. We omit PhotoMetricDistortion so the
# teacher sees the same photometric range as evaluation.
unlabeled_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024),
         ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='SSLDataset',
        labeled=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/train',
            ann_dir='annotations/train',
            split='train_500_fixed.txt',
            ignore_index=255,
            pipeline=labeled_train_pipeline),
        unlabeled=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/train',
            ann_dir='annotations/train',
            split='train_3500_fixed.txt',
            ignore_index=255,
            pipeline=unlabeled_train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        split='val_2000_fixed.txt',
        ignore_index=255,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        split='val_2000_fixed.txt',
        ignore_index=255,
        pipeline=test_pipeline))
