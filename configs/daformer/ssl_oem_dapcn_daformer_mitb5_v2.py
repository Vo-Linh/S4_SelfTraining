# ---------------------------------------------------------------
# DAPCN-SSL + DAFormer on OpenEarthMap — v2 contrastive objective
#
# Differs from ssl_oem_dapcn_daformer_mitb5.py only in the contrastive
# loss configuration. The v2 objective addresses the structural reason
# L_contr plateaus in v1:
#
#   v1: queries = labeled features,
#       positives = class-EMA centroid of the SAME labeled features,
#       negatives = the other 8 class centroids.
#       → self-referential target, 9-way easy task,
#         saturates after ~150 iters.
#
#   v2: queries = labeled features,
#       bank = labeled GT + EMA-teacher-confident pseudo features from
#              the 3500 unlabeled images (decouples target from queries,
#              and brings unlabeled distribution into the loss),
#       extra negatives = DynamicAnchorModule prototypes detached
#              (96 dataset-level, class-agnostic, manifold-spanning
#              hard negatives; turns the task from 9-way into 105-way),
#       loss = prototype_contrastive_loss_extended.
#
# Two switches do this: ``contrastive_use_teacher=True`` and
# ``contrastive_use_dynanchor_negatives=True``.
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
    # --- v2 contrastive: extended objective ---
    contrastive_lambda=0.1,
    contrastive_temp=0.07,
    num_prototypes_per_class=1,
    prototype_ema=0.999,
    prototype_init_strategy='zeros',
    contrastive_use_teacher=True,            # ← v2: bank gets teacher pseudos
    contrastive_use_dynanchor_negatives=True,  # ← v2: extra hard negatives
    # ------------------------------------------
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
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
evaluation = dict(interval=4000, metric='mIoU')

name = 'ssl_oem_dapcn_daformer_mitb5_v2'
exp = 'ssl_oem_v2'
name_dataset = 'openearthmap_ssl_500_3500'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dapcn_ssl_affinity_contrastiveV2'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
