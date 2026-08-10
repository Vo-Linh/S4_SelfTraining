# ---------------------------------------------------------------
# DAPCN-SSL + UNetFormer (ResNet18) on OpenEarthMap (9-class land cover)
#
# Semi-supervised self-training with the DAPCN_SSL UDA wrapper driving
# an EncoderDecoder(TIMMBackbone + UNetFormerHead). DAPCN (dynamic anchor,
# DAPG grouping, affinity boundary, prototype-based pseudo-label
# correction) is applied by the wrapper on the UNetFormer decoder
# features -- no decode-head DAPCN variant is used.
#
#   labeled    = train_500_fixed.txt   (500 images with GT)
#   unlabeled  = train_3500_fixed.txt  (3500 images, labels ignored)
#   val        = val_2000_fixed.txt    (2000 images)
#
# Mirrors ssl_oem_dapcn_daformer_mitb5.py; only the architecture base
# and the optimizer paramwise keys (no MiT pos_block) differ.
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # Architecture: UNetFormer (ResNet18 encoder + GLA decoder)
    '../_base_/models/unetformer_resnext101.py',
    # Data: OpenEarthMap SSL 500/3500 split
    '../_base_/datasets/ssl_openearthmap_512x512.py',
    # DAPCN-SSL training loop (EMA teacher + prototype correction)
    '../_base_/ssl/dapcn_ssl.py',
    # Optimizer + schedule
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py',
]

seed = 0

# ResNet18 encoder loads SWSL ImageNet weights via timm (backbone
# pretrained=True in the model base). num_classes=9 is already set in the
# model base; kept here for clarity.
model = dict(
    pretrained=None,
    decode_head=dict(num_classes=9))

# DAPCN-SSL hyperparameters. Inherits the full uda dict from
# _base_/ssl/dapcn_ssl.py; the overrides below match the DAFormer OEM
# reference run. DynamicAnchorModule.feature_dim is auto-injected by
# DAPCN_SSL from the head channels (Solution 1: in_channels[-1]=2048, with
# a learned proto_to_decoder Linear(2048->256) bridge to conv_seg).
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

# Per-parameter optimizer tuning (STIC_semi Appendix): 10x LR for the
# randomly-initialised decode head and the DynamicAnchor prototypes /
# quality-net / projection; zero weight-decay on prototype-space params
# to preserve clustering geometry. 'head' matches the UNetFormer
# decode_head.* params; no MiT 'pos_block' key here.
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.0),
            'dynamic_anchor.prototypes': dict(lr_mult=10.0, decay_mult=0.0),
            'dynamic_anchor.quality_net': dict(lr_mult=10.0),
            'proto_to_decoder': dict(lr_mult=10.0, decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
evaluation = dict(interval=4000, metric='mIoU')

name = 'ssl_oem_dapcn_unetformer_resnext101'
exp = 'ssl_oem'
name_dataset = 'openearthmap_ssl_500_3500'
name_architecture = 'unetformer_resnext101'
name_encoder = 'resnext101_32x16d'
name_decoder = 'unetformer_gla'
name_uda = 'dapcn_ssl_affinity_contrastive'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
