# ---------------------------------------------------------------
# Baseline SSL on OpenEarthMap (no DAPCN components)
#
# Pure EMA-teacher self-training:
#   - Supervised CE on labeled (500 images)
#   - EMA teacher pseudo-labels on unlabeled (3500 images)
#   - ClassMix augmentation
#   - Linear pseudo-label warmup
#
# Disabled vs full DAPCN-SSL:
#   - boundary_lambda      = 0  (no AffinityBoundaryLoss)
#   - proto_lambda         = 0  (no DAPGLoss, no DynamicAnchorModule)
#   - contrastive_lambda   = 0  (no PrototypeMemory / InfoNCE)
#   - proto_correction     = False
#
# This is the reference point for measuring the contribution of
# DynamicAnchor (see ssl_oem_dynanchor_daformer_mitb5.py).
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
    # --- All DAPCN components off ---
    proto_correction=False,
    boundary_lambda=0.0,
    proto_lambda=0.0,
    contrastive_lambda=0.0,
    # Sub-module configs are ignored when their lambdas are 0,
    # but we null them out for clarity.
    dynamic_anchor=None,
    dapg_loss=None,
    affinity_loss=None,
)

optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.0),
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
evaluation = dict(interval=4000, metric='mIoU')

name = 'ssl_oem_daformer_selftrain_mitb5'
exp = 'ssl_oem'
name_dataset = 'openearthmap_ssl_500_3500'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'ssl_baseline_selftraining'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
