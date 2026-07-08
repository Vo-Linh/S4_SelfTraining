# ---------------------------------------------------------------
# SSL ladder — TIER 2: DAPCN (dynamic-anchor + prototype-memory).
#
# The DAPCN method on top of the DAFormer self-training baseline:
# affinity boundary loss + DAPG prototype loss + v2 contrastive
# (PrototypeMemory bank) + DynamicAnchorModule. NO pseudo-label
# correction of any kind (proto_correction=False, dwpc.enabled=False).
#
# Ladder:
#   tier 1  ssl_oem_daformer_selftrain_mitb5.py   (pure self-training)
#   tier 2  THIS FILE                              (+ DynAnchor + ProtoMem)
#   tier 3  ssl_oem_dapcn_dwpc_full_mitb5.py       (+ DWPC dual-witness)
#
# This is the clean isolating anchor for DWPC:
#   tier3 - tier2  ==  the effect of the DWPC block, ALONE
#                      (the dwpc block here is byte-identical to tier 3,
#                       only enabled=False).
#
#   CUDA_VISIBLE_DEVICES=0 \
#   /home/ubuntu/SatelliteImageSegSupevised/.venv/bin/python tools/train.py \
#     configs/daformer/ssl_oem_dapcn_dynanchor_proto_mitb5.py \
#     --work-dir work_dirs/dwpc/dapcn_dynanchor_proto_s0 --seed 0 --deterministic
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

model = dict(
    pretrained=None,
    decode_head=dict(num_classes=9))

uda = dict(
    # ---- self-training / EMA teacher (identical to tier 3) ----
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    pseudo_label_warmup_iters=1000,

    # ---- legacy soft proto-correction: OFF (same as tier 3) ----
    proto_correction=False,
    proto_correction_alpha=0.5,
    proto_correction_start_iter=1000,

    # ---- DAPCN losses: ALL ON (identical to tier 3) ----
    boundary_lambda=0.5,
    proto_lambda=0.1,
    contrastive_lambda=0.1,
    contrastive_temp=0.07,
    num_prototypes_per_class=1,
    prototype_ema=0.999,
    prototype_init_strategy='zeros',
    contrastive_use_teacher=True,
    contrastive_use_dynanchor_negatives=True,
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

    # ---- DWPC: DISABLED (the only difference vs tier 3) ----
    # Block kept byte-identical to tier 3 so the comparison is one bit.
    # With enabled=False, _dwpc_needed() is always False: _dwpc_correct
    # is never called, pseudo_weight stays the scalar baseline, and no
    # Witness-B buffers are registered.
    dwpc=dict(
        enabled=False,
        start_iter=1500,
        ramp_iters=1000,
        rare_class_ids=[0, 1, 6],
        class_freq=[0.00654, 0.01488, 0.22473, 0.16005, 0.06658,
                    0.20029, 0.03246, 0.13799, 0.15648],
        witness_a_enabled=True,
        witness_a_symmetric=False,
        witness_a_rare_prior_gamma=1.0,
        witness_a_beta=1.0,
        witness_a_sigma2=0.5,
        witness_a_standardize=True,
        witness_b_enabled=True,
        witness_b_ema=0.99,
        witness_b_cc_iters=64,
        witness_b_resolution=128,
        witness_b_enable_containment=True,
        witness_b_min_count=5,
        target_adaptive=True,
        rare_gate=True,
        flip_veto=True,
        tau_flip=0.5,
        flip_budget=0.005,
        beta_s=5.0,
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
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
evaluation = dict(interval=4000, metric='mIoU')

name = 'ssl_oem_dapcn_dynanchor_proto_mitb5'
exp = 'ssl_oem_dwpc'
name_dataset = 'openearthmap_ssl_500_3500'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dapcn_dynanchor_proto'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
