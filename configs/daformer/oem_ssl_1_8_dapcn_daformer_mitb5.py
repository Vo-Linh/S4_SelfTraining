# ---------------------------------------------------------------
# DAPCN-SSL + DAFormer — OpenEarthMap, 1/8 labeled split
#
# Architecture: DAFormer (MiT-B5 encoder + SepASPP decoder)
# Data:         187 labeled / 1313 unlabeled / 2000 val
# Classes:      8 (raw label 0 = "unknown" is ignored)
#
# Sibling of oem_ssl_1_16_dapcn_daformer_mitb5.py at a higher label
# ratio (1/8 instead of 1/16) — same architecture, same DAPCN/RCS
# settings, only the split files and RCS stats differ.
#
# Usage:
#   python tools/train.py \
#       configs/daformer/oem_ssl_1_8_dapcn_daformer_mitb5.py
#
# Prerequisite: pretrained/mit_b5.pth
# RCS statistics for this split already generated as
# sample_class_stats_1_8.json / samples_with_class_1_8.json.
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/ssl_oem_1_8_512x512.py',
    '../_base_/ssl/dapcn_ssl.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py',
]

seed = 0

# 8 classes: OpenEarthMap 1..8 shifted to 0..7, raw 0 ignored.
model = dict(
    decode_head=dict(
        num_classes=8,
    ))

uda = dict(
    alpha=0.999,
    pseudo_threshold=0.968,
    # Overhead imagery — no rectification artefacts to crop away.
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    # With 187 labeled images the teacher is still unreliable early
    # on, so ramp the pseudo-label weight in over the first 1000 iters.
    pseudo_label_warmup_iters=1000,
    # --- Prototype-based pseudo-label correction ---
    proto_correction=True,
    proto_correction_alpha=0.5,
    proto_correction_start_iter=1000,
    # --- DAPCN losses (labeled images only; see dapcn_ssl.py) ---
    boundary_lambda=0.5,
    proto_lambda=0.1,
    boundary_loss_mode='affinity',
    boundary_mode='sobel',
    apply_boundary_on_target=False,
    apply_proto_on_target=False,
    ignore_index=255,
    # --- DynamicAnchor: plain cosine assignment (the baseline that
    #     ablations D1-D4 vary) ---
    dynamic_anchor=dict(
        type='DynamicAnchorModule',
        max_groups=96,
        temperature=0.1,
        num_iters=3,
        init_method='xavier',
        min_quality=0.1,
        use_quality_gate=True,
    ),
    dapg_loss=dict(
        type='DAPGLoss',
        margin=0.3,
        lambda_inter=0.5,
        lambda_quality=0.1,
        loss_weight=1.0,
    ),
    affinity_loss=dict(
        type='AffinityBoundaryLoss',
        temperature=0.5,
        scale=2,
        num_neighbors=4,
        ignore_index=255,
        loss_weight=1.0,
    ),
)

# --- Rare Class Sampling on the labeled subset ---
# Statistics are split-specific: they were computed from the 187 labeled
# images only, with reduce_zero_label applied, so the class ids match
# what the pipeline produces. All 8 classes clear min_pixels=3000.
#
# class_temp=0.1, NOT the 0.01 used by DAFormer/the satellite template,
# for the same reason as the 1/16 config: RCS draws classes from
# softmax((1 - freq) / class_temp), and this split's distribution (8
# classes spanning 1.7%-22.4% pixel share) is far flatter than the
# 19-class Cityscapes tail 0.01 was tuned for, so 0.01 makes the
# softmax degenerate:
#
#   temp    bareland water  road   agri   dev.   build  range  tree
#   0.01    0.685    0.311  0.004  0.000  0.000  0.000  0.000  0.000
#   0.1     0.280    0.259  0.166  0.083  0.069  0.067  0.042  0.035
#
# At 0.01, 68.5% of crops would come from the 31 images containing
# bareland — out of 187 labeled images total. That is a memorisation
# risk, not rare-class balancing. At 0.1 the rare classes are still
# favoured (bareland/water get ~3-16x their pixel share) while the
# sampling stays spread across a much larger image pool.
data = dict(
    train=dict(
        rare_class_sampling=dict(
            min_pixels=3000,
            class_temp=0.1,
            min_crop_ratio=0.5,
            stats_file='sample_class_stats_1_8.json',
            samples_file='samples_with_class_1_8.json')))

# --- Optimizer ---
# DynAnchor prototypes: no weight decay (cluster centres should not be
# pulled toward zero) and 10x LR (randomly initialised, and gradients
# are attenuated through 3 EM iterations).
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.0),
            'dynamic_anchor.prototypes': dict(lr_mult=10.0, decay_mult=0.0),
            'dynamic_anchor.quality_net': dict(lr_mult=10.0),
            # Only present when the corresponding flag is enabled;
            # unmatched custom_keys are simply ignored.
            'dynamic_anchor.logit_scale': dict(lr_mult=10.0, decay_mult=0.0),
            'dynamic_anchor.query_proj': dict(lr_mult=10.0, decay_mult=0.0),
            'dynamic_anchor.key_proj': dict(lr_mult=10.0, decay_mult=0.0),
            'proto_to_decoder': dict(lr_mult=10.0, decay_mult=0.0),
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)

checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
# 2000 val tiles is a slow evaluation; keep the interval coarse and
# read debug/* for the fine-grained view between checkpoints.
evaluation = dict(interval=4000, metric='mIoU')

name = 'oem_ssl_1_8_dapcn_daformer_mitb5'
exp = 'oem_ssl'
name_dataset = 'oem_ssl_1_8'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dapcn_ssl_affinity_rcs'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
