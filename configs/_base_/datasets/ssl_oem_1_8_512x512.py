# ---------------------------------------------------------------
# OpenEarthMap — Semi-Supervised split 1/8 (187 labeled / 1313 unlabeled)
#
# Data: /home/c405/Data_10TB/project/data/OpenEarthMap_wo_xBD_coco_1500_2000
#       1500 train + 2000 val .tif tiles (1000x1000, some 1024x1024)
#       labels uint8 with raw values 0..8
#
# Three dataset-specific decisions are baked in here; changing any of
# them silently invalidates comparisons with the other runs.
#
# 1. reduce_zero_label=True.  Raw label 0 is OpenEarthMap's "unknown"
#    class; it is mapped to ignore and classes 1..8 shift down to 0..7,
#    so num_classes=8.  This is what the official OpenEarthMap
#    benchmark reports, which keeps our mIoU directly comparable with
#    published numbers.  It must be set in BOTH the pipeline
#    (LoadAnnotations) and the dataset (for evaluation) — setting only
#    one produces silently wrong metrics.
#
# 2. The unlabeled pool is the COMPLEMENT of the labeled split
#    (1407 = 1500 - 93; the two are disjoint).  Note this departs from
#    the SSLDataset docstring, which describes an unlabeled pool that
#    also contains the labeled images.  Both are defensible; state the
#    choice explicitly in the paper.
#
# 3. Split files: only the root-level `*_filename_only*` variants work.
#    `splits/1_8/labeled.txt` is a paired two-column format and
#    `splits/train_filename_only_1_8.txt` carries a `.tif` suffix;
#    mmseg does `line.strip() + img_suffix`, so neither loads.
#
# RCS statistics were generated for this split with:
#   python tools/convert_datasets/generate_rcs_stats.py <data_root> \
#     --ann-dir annotations/train --split train_filename_only_1_8.txt \
#     --seg-map-suffix .tif --reduce-zero-label \
#     --min-pixels 3000 --auto-min-pixels --out-suffix _1_8
# All 8 classes clear min_pixels=3000 in this split.
# ---------------------------------------------------------------

dataset_type = 'CustomDataset'
data_root = '/home/p40/data_hdd_4TB/data/data/OpenEarthMap_wo_xBD_coco_1500_2000/'

num_classes = 8

# OpenEarthMap taxonomy AFTER reduce_zero_label (raw 1..8 -> 0..7).
classes = [
    'bareland', 'rangeland', 'developed', 'road',
    'tree', 'water', 'agriculture', 'building',
]

palette = [
    [128, 0, 0], [0, 255, 36], [148, 148, 148], [255, 255, 255],
    [34, 97, 38], [0, 69, 255], [75, 181, 73], [222, 31, 7],
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

crop_size = (512, 512)

# Tiles are ~1000x1000, so a 512 crop keeps roughly a quarter of a tile
# at native resolution — no downscaling of the imagery.
labeled_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# Labels are loaded but never used as supervision for this branch —
# pseudo-labels from the EMA teacher replace them.
unlabeled_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
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
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Deterministic pipeline for ProtoCorrectionDebugHook: no random crop
# or flip, so the same pixels are compared at every evaluation.
debug_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    # meta_keys is spelled out to drop 'flip'/'flip_direction': those are
    # set only by RandomFlip, which this pipeline deliberately omits so
    # the same pixels are compared at every evaluation. Adding a
    # RandomFlip(prob=0) instead would fix the KeyError but would draw
    # from the global numpy RNG on every debug image, perturbing the
    # augmentation stream that training itself depends on.
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'img_norm_cfg')),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='SSLDataset',
        labeled=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/train',
            ann_dir='annotations/train',
            img_suffix='.tif',
            seg_map_suffix='.tif',
            split='train_filename_only_1_8.txt',   # 187 images
            reduce_zero_label=True,
            classes=classes,
            palette=palette,
            pipeline=labeled_train_pipeline),
        unlabeled=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/train',
            ann_dir='annotations/train',
            img_suffix='.tif',
            seg_map_suffix='.tif',
            # Complement of the labeled split — see note 2 above.
            split='train_filename_only_unlabeled_1_8.txt',  # 1313 images
            reduce_zero_label=True,
            classes=classes,
            palette=palette,
            pipeline=unlabeled_train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        img_suffix='.tif',
        seg_map_suffix='.tif',
        split='val_filename_only_fixed.txt',       # 2000 images
        reduce_zero_label=True,
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        img_suffix='.tif',
        seg_map_suffix='.tif',
        split='val_filename_only_fixed.txt',
        reduce_zero_label=True,
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))

# ---------------------------------------------------------------
# Prototype-correction diagnostics — see EXPERIMENTS_ASSIGNMENT.md
#
# Reads the ground truth of the UNLABELED pool to answer whether the
# correction actually helps: debug/teacher_mIoU vs debug/proto_mIoU vs
# debug/blend_mIoU, plus flip_acc_before/after on the pixels the
# correction changes.
#
# That ground truth is hidden from training by design, so these numbers
# must never drive checkpoint selection or early stopping — report
# final results from the val split. To disable: `custom_hooks = []`.
# ---------------------------------------------------------------
custom_hooks = [
    dict(
        type='ProtoCorrectionDebugHook',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/train',
            ann_dir='annotations/train',
            img_suffix='.tif',
            seg_map_suffix='.tif',
            split='train_filename_only_unlabeled_1_8.txt',
            reduce_zero_label=True,
            classes=classes,
            palette=palette,
            pipeline=debug_pipeline),
        num_debug_samples=64,
        ce_num_samples=8,
        ce_log_interval=200,
        miou_eval_interval=1000,
        batch_size=2,
        subset_seed=0,
    )
]
