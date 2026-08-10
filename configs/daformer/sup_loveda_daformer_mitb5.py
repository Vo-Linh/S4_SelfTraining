# ---------------------------------------------------------------
# SupOnly baseline: DAFormer(MiT-B5) on LoveDA (7 classes).
#   Lower bound  -> data.train.split = loveda_train_{1,5,10}pct_labeled.txt
#   Upper bound  -> data.train.split = loveda_train_full.txt   (100% oracle)
#
# Identical to sup_oem_daformer_mitb5.py except the dataset base (LoveDA) and
# num_classes=7. PLACE AT: S4_SelfTraining/configs/daformer/sup_loveda_daformer_mitb5.py
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/sup_loveda_512x512.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py',
]

seed = 0

# set pretrained='pretrained/mit_b5.pth' for paper-valid (ImageNet-init) runs.
model = dict(pretrained=None, decode_head=dict(num_classes=7))

# NOT None: plain EncoderDecoder (SupOnly) needs the OptimizerHook to step().
# None only fits UDA models (DAPCN_SSL) that step manually in train_step.
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

name = 'sup_loveda_daformer_mitb5'
exp = 'sup_loveda'
name_dataset = 'loveda_sup'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'none_suponly'
name_opt = 'adamw_6e-05_poly10warm_1x4_40k'
