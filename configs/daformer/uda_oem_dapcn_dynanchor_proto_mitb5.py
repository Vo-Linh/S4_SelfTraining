# ---------------------------------------------------------------
# DAPCN + DAFormer for OpenEarthMap UDA (DWPC base leaf)
#
# OpenEarthMap source->target UDA with the DAPCN self-training loop.
# Serves as the shared base for the uda_oem_dwpc_a{0..7} ablation rows;
# on its own (no dwpc block) it is the A0 UDA baseline.
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/uda_oem_512x512.py',
    '../_base_/uda/dapcn.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]

seed = 0

model = dict(
    pretrained=None,
    decode_head=dict(num_classes=9))

uda = dict(
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    boundary_lambda=0.5,
    proto_lambda=0.1,
    boundary_loss_mode='affinity',
    boundary_mode='sobel',
    apply_boundary_on_target=True,
    apply_proto_on_target=True,
    # PrototypeMemory for Witness A (created when dwpc+witness_a enabled)
    num_prototypes_per_class=1,
    prototype_ema=0.999,
    prototype_init_strategy='zeros',
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
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
evaluation = dict(interval=4000, metric='mIoU')

name = 'uda_oem_dapcn_dynanchor_proto_mitb5'
exp = 'uda_oem'
name_dataset = 'openearthmap_uda'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dapcn_affinity'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
