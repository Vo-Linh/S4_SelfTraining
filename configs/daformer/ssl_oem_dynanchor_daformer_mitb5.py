# ---------------------------------------------------------------
# Baseline + DynamicAnchor on OpenEarthMap
#
# Adds the DynamicAnchorModule on top of the baseline self-training
# pipeline. Boundary loss and class-conditioned contrastive remain
# OFF so this config isolates the DynAnchor contribution.
#
# What changes vs. ssl_oem_daformer_selftrain_mitb5.py:
#   - proto_lambda          = 0.1   (enables DynamicAnchorModule
#                                    + DAPGLoss on labeled & mixed
#                                    features — L_intra + L_inter
#                                    + L_quality)
#   - proto_correction      = True  (after start_iter, blend teacher
#                                    softmax with prototype-derived
#                                    p^c_j = Σ f_θ(PT_i) · a_ij)
#   - dynamic_anchor / dapg_loss configured
#   - paramwise_cfg adds custom_keys for dynamic_anchor.*
#     and proto_to_decoder
#
# Still off (kept for clean isolation):
#   - boundary_lambda    = 0
#   - contrastive_lambda = 0
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
    # --- DynamicAnchor + prototype grouping ON ---
    proto_lambda=0.1,
    apply_proto_on_target=True,
    proto_correction=True,
    proto_correction_alpha=0.5,
    proto_correction_start_iter=1000,
    anchor_after_fusion=False,
    # --- Still off ---
    boundary_lambda=0.0,
    contrastive_lambda=0.0,
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
    affinity_loss=None,
)

optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.0),
            # DynAnchor prototype bank: random init, EM-attenuated
            # gradient → 10x LR, no weight decay.
            'dynamic_anchor.prototypes': dict(lr_mult=10.0, decay_mult=0.0),
            'dynamic_anchor.quality_net': dict(lr_mult=10.0),
            # Solution-1 projection from encoder space to decoder space
            # (used by prototype pseudo-label correction).
            'proto_to_decoder': dict(lr_mult=10.0, decay_mult=0.0),
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
evaluation = dict(interval=4000, metric='mIoU')

name = 'ssl_oem_dynanchor_daformer_mitb5'
exp = 'ssl_oem'
name_dataset = 'openearthmap_ssl_500_3500'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'ssl_baseline_plus_dynanchor'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
