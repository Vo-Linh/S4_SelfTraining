# ---------------------------------------------------------------
# SupOnly baseline: pyramidmamba (ResNeXt101) on OpenEarthMap -- Table IX lower bound.
#
# NO 'uda' block => build_train_model builds a plain EncoderDecoder, with no
# EMA teacher, no ClassMix, no DAPCN. The model sees ONLY the labeled split.
# Architecture, optimizer, schedule, iters and val protocol are identical to
# ssl_oem_dapcn_pyramidmamba_resnext101.py, so the gap between the two is exactly
# the contribution of the unlabeled data + DAPCN.
#
# Set the ratio at launch:
#   --options data.train.split=train_5pct_labeled.txt
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/pyramidmamba_resnext101.py',
    '../_base_/datasets/sup_openearthmap_512x512.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py',
]

seed = 0

model = dict(
    pretrained=None,
    decode_head=dict(num_classes=9))

# Same 10x head LR as the SSL runs. No dynamic_anchor / proto_to_decoder keys
# here -- those modules only exist inside the DAPCN_SSL wrapper.
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.0),
            'norm': dict(decay_mult=0.0),
        }))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
evaluation = dict(interval=4000, metric='mIoU')

name = 'sup_oem_pyramidmamba_resnext101'
exp = 'sup_oem'
name_dataset = 'openearthmap_sup'
name_architecture = 'pyramidmamba_resnext101'
name_encoder = 'resnext101_32x16d'
name_decoder = 'pyramidmamba'
name_uda = 'none_suponly'
name_opt = 'adamw_6e-05_poly10warm_1x4_40k'
