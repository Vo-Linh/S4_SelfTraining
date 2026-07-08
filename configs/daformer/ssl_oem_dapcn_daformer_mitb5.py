# ---------------------------------------------------------------
# DAPCN-SSL + DAFormer on OpenEarthMap (9-class land cover)
#
# Real-data SSL training using the OpenEarthMap dataset:
#   labeled    = train_500_fixed.txt   (500 images with GT)
#   unlabeled  = train_3500_fixed.txt  (3500 images, labels ignored)
#   val        = val_2000_fixed.txt    (2000 images)
#
# Adapted from satellite_ssl_dapcn_daformer_mitb5.py (generic
# satellite layout) to point at the OEM data root and 9-class taxonomy.
# The reference supervised counterpart is
# /home/ubuntu/SatelliteImageSegSupevised/configs/segformer/
# segformer_mit-b5_openearthmap_train500_40k.py.
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/ssl_openearthmap_512x512.py',
    '../_base_/ssl/dapcn_ssl.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]

seed = 0

# OEM has 9 classes. The MiT-B5 backbone runs from random init here
# (matching the reference supervised configs which also do not load
# external ImageNet weights). Drop a real mit_b5.pth at
# /home/ubuntu/S4_SelfTraining/pretrained/mit_b5.pth and re-enable
# `pretrained=` if you want a warm start.
model = dict(
    pretrained=None,
    decode_head=dict(num_classes=9))

uda = dict(
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
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

# No RCS — sample_class_stats.json is not provided for OEM in this
# layout. Re-enable with rare_class_sampling=dict(...) once stats are
# generated.

optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.0),
            'dynamic_anchor.prototypes': dict(lr_mult=10.0, decay_mult=0.0),
            'dynamic_anchor.quality_net': dict(lr_mult=10.0),
            'proto_to_decoder': dict(lr_mult=10.0, decay_mult=0.0),
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }))

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
