# ---------------------------------------------------------------
# DWPC FULL — every component enabled, in one self-contained config.
#
# DAPCN-SSL stack (boundary + DAPG prototype grouping + v2 contrastive +
# DynamicAnchor) AND the full DWPC dual-witness pseudo-label correction
# (Witness A appearance + Witness B structure + rare-gate + target-
# adaptive + flip-veto). This is the "everything on" config (== row A5,
# but with all knobs written out explicitly rather than inherited).
#
#   python tools/train.py \
#     configs/daformer/ssl_oem_dapcn_dwpc_full_mitb5.py
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
    # ---- self-training / EMA teacher ----
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    pseudo_label_warmup_iters=1000,

    # ---- legacy soft proto-correction: OFF (DWPC replaces it) ----
    proto_correction=False,
    proto_correction_alpha=0.5,
    proto_correction_start_iter=1000,

    # ---- DAPCN losses: ALL ON ----
    boundary_lambda=0.5,
    proto_lambda=0.1,
    contrastive_lambda=0.1,
    contrastive_temp=0.07,
    num_prototypes_per_class=1,
    prototype_ema=0.999,
    prototype_init_strategy='zeros',
    contrastive_use_teacher=True,            # bank sees target pixels (Witness A precondition)
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

    # ---- DWPC dual-witness correction: ALL COMPONENTS ON ----
    dwpc=dict(
        enabled=True,
        start_iter=1500,
        ramp_iters=1000,
        rare_class_ids=[0, 1, 6],
        class_freq=[0.00654, 0.01488, 0.22473, 0.16005, 0.06658,
                    0.20029, 0.03246, 0.13799, 0.15648],
        # Witness A — appearance (prototype-density likelihood ratio)
        witness_a_enabled=True,
        witness_a_symmetric=False,            # asymmetric flip-to-rare
        witness_a_rare_prior_gamma=1.0,       # inverse-freq rare prior on
        witness_a_beta=1.0,
        witness_a_sigma2=0.5,
        witness_a_standardize=True,
        # Witness B — structure (target-adaptive plausibility)
        witness_b_enabled=True,
        witness_b_ema=0.99,
        witness_b_cc_iters=64,
        witness_b_resolution=128,
        witness_b_enable_containment=True,
        witness_b_min_count=5,
        target_adaptive=True,
        # fusion / decision
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

name = 'ssl_oem_dapcn_dwpc_full_mitb5'
exp = 'ssl_oem_dwpc'
name_dataset = 'openearthmap_ssl_500_3500'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dwpc_full'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
