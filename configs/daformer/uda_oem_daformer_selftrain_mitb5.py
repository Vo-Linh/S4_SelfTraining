# ---------------------------------------------------------------
# UDA ladder — TIER 1: DAFormer self-training baseline (no DAPCN).
#
# Pure EMA-teacher cross-domain self-training on OpenEarthMap UDA:
#   - Supervised CE on the source domain
#   - EMA teacher pseudo-labels on the target domain
#   - ClassMix augmentation
# All DAPCN components OFF (boundary / DAPG prototype / DynamicAnchor),
# no pseudo-label correction. The UDA counterpart of
# ssl_oem_daformer_selftrain_mitb5.py.
#
# Ladder:
#   tier 1  THIS FILE                            (self-training)
#   tier 2  uda_oem_dapcn_dynanchor_proto_mitb5  (+ DynAnchor + ProtoMem)
#   tier 3  uda_oem_dapcn_dwpc_full_mitb5         (+ DWPC dual-witness)
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
    # --- All DAPCN components off (pure self-training) ---
    boundary_lambda=0.0,
    proto_lambda=0.0,
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

name = 'uda_oem_daformer_selftrain_mitb5'
exp = 'uda_oem'
name_dataset = 'openearthmap_uda'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'uda_baseline_selftraining'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
