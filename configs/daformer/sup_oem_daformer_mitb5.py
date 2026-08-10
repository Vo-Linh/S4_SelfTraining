# ---------------------------------------------------------------
# SupOnly baseline: DAFormer(MiT-B5) on OpenEarthMap.
#   Lower bound  -> data.train.split = train_{1,5,10}pct_labeled.txt
#   Upper bound  -> data.train.split = train_3500_fixed.txt   (100% oracle)
#
# No `uda` block => plain EncoderDecoder, labeled split only. Arch / schedule /
# aug / val are identical to ssl_oem_dapcn_daformer_mitb5.py, so the gap between
# this and the DAPCN-SSL run = exactly the contribution of unlabeled data + DAPCN.
#
# PLACE AT: S4_SelfTraining/configs/daformer/sup_oem_daformer_mitb5.py
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/sup_openearthmap_512x512.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py',
]

seed = 0

# NOTE: set pretrained='pretrained/mit_b5.pth' for paper-valid (ImageNet-init) runs.
# Left None here so the config builds before the weights are dropped in.
model = dict(pretrained=None, decode_head=dict(num_classes=9))

# NOT None: a plain EncoderDecoder (SupOnly) needs the OptimizerHook to call
# optimizer.step(). None is only correct for UDA models (DAPCN_SSL) whose
# train_step does backward/step manually — copying None here froze the weights
# (collapse to 1/num_classes). dict() inherits the base OptimizerHook.
optimizer_config = dict()
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(custom_keys={
        'head': dict(lr_mult=10.0),
        'pos_block': dict(decay_mult=0.0),
        'norm': dict(decay_mult=0.0),
    }))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=3)
evaluation = dict(interval=4000, metric='mIoU')

name = 'sup_oem_daformer_mitb5'
exp = 'sup_oem'
name_dataset = 'openearthmap_sup'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'none_suponly'
name_opt = 'adamw_6e-05_poly10warm_1x4_40k'
