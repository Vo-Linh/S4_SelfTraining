# ---------------------------------------------------------------
# Supervised (SupOnly) LoveDA dataset base.
#
# Mirror of sup_openearthmap_512x512.py, swapped to LoveDA (7 classes).
# Used for the Table X lower bound (SupOnly @ 1/5/10%) and upper bound
# (FullSup oracle @ 100%). Split is overridden per run at launch.
#
# PLACE AT: S4_SelfTraining/configs/_base_/datasets/sup_loveda_512x512.py
#
# !!! VERIFY ON BOX (server was down when authored) !!!
#   [A] data_root + img_dir/ann_dir match the actual LoveDA layout on disk.
#   [B] LoveDA masks are stored as 0=no-data(ignore), 1..7=classes  ->
#       reduce_zero_label=True (below) remaps to 0..6 + 255 ignore.
#       If masks are ALREADY 0..6 with 255 ignore, set reduce_zero_label=False.
#   [C] split list file names exist (see paper_configs_s4/README.md for gen).
# ---------------------------------------------------------------

dataset_type = 'LoveDADataset'
data_root = '/home/ubuntu/data/LoveDA/LoveDA_flat/'   # TODO[A] verify

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),   # TODO[B]
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
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',        # TODO[A]
        ann_dir='annotations/train',   # TODO[A]
        split='loveda_train_5pct_labeled.txt',   # overridden per run
        reduce_zero_label=True,        # TODO[B] keep in sync with LoadAnnotations
        ignore_index=255,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',          # TODO[A]
        ann_dir='annotations/val',     # TODO[A]
        split='loveda_val_fixed.txt',  # TODO[C]
        reduce_zero_label=True,        # TODO[B]
        ignore_index=255,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        split='loveda_val_fixed.txt',
        reduce_zero_label=True,
        ignore_index=255,
        pipeline=test_pipeline))
