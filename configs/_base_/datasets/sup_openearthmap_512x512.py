# ---------------------------------------------------------------
# Supervised (SupOnly) OpenEarthMap -- the Table IX lower bound.
#
# Identical to ssl_openearthmap_512x512.py except there is no SSLDataset
# wrapper and no unlabeled branch: the model sees ONLY the labeled split.
# Same labeled pipeline, same val/test, same crop -- so the difference
# between this and the DAPCN-SSL run is exactly "use of unlabeled data".
#
# samples_per_gpu=4 matches the SSL config's *labeled* branch (which is 4
# labeled + 4 unlabeled). Giving SupOnly 8 labeled images per iteration
# would hand it 2x the labelled supervision per step and stop it being a
# clean baseline.
#
# The split is overridden per ratio at launch, e.g.
#   --options data.train.split=train_5pct_labeled.txt
# ---------------------------------------------------------------

dataset_type = 'OpenEarthMapDataset'
data_root = '/home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

crop_size = (512, 512)

train_pipeline = [
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
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='annotations/train',
        split='train_5pct_labeled.txt',
        ignore_index=255,
        pipeline=train_pipeline),
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
