# ---------------------------------------------------------------
# DAPCN-SSL + PyramidMamba (ResNet18) on OpenEarthMap (9-class)
#
# Architecture counterpart to ssl_oem_dapcn_unetformer_r18.py: same encoder
# (TIMM ResNet18/SWSL), same DAPCN-SSL loop, same optimizer/schedule/iters.
# The ONLY difference is the decoder (EfficientPyramidMamba vs UNetFormer
# GLA), which is what makes the two directly comparable for the
# backbone/decoder-agnostic study (paper Table XIII).
#
# Because both heads expose channels=64 and in_channels[-1]=512, DAPCN_SSL
# auto-derives the SAME feature_dim (512) and the SAME proto_to_decoder
# Linear(512 -> 64) for both -- so the prototype machinery is identical and
# only the decoder varies.
#
# Anchor placement: Solution 1 (anchor_after_fusion=False, the default) --
# the dynamic anchor runs on the stride-32 encoder feature (16x16). Do NOT
# flip to Solution 2 here: PyramidMamba's fused map is at FULL input
# resolution, so EM clustering would run over B*512*512 pixels.
#
# Requires mamba_ssm + causal_conv1d in the venv.
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/pyramidmamba_r18.py',
    '../_base_/datasets/ssl_openearthmap_512x512.py',
    '../_base_/ssl/dapcn_ssl.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py',
]

seed = 0

model = dict(
    pretrained=None,
    decode_head=dict(num_classes=9))

# Identical to the UNetFormer config so the comparison is clean.
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

name = 'ssl_oem_dapcn_pyramidmamba_r18'
exp = 'ssl_oem'
name_dataset = 'openearthmap_ssl_500_3500'
name_architecture = 'pyramidmamba_r18'
name_encoder = 'resnet18'
name_decoder = 'pyramidmamba'
name_uda = 'dapcn_ssl_affinity_contrastive'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
