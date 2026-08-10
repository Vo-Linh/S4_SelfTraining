# ---------------------------------------------------------------
# OEM LOWER BOUND (SupOnly) — DAFormer(MiT-B5), 5% labels.
# Table IX "SupOnly" row @ 5% (the ablation anchor ratio).
#
# PLACE AT: S4_SelfTraining/configs/daformer/sup_oem_daformer_mitb5_5pct.py
# RUN:
#   python tools/train.py configs/daformer/sup_oem_daformer_mitb5_5pct.py \
#     --work-dir work_dirs/E0.1_sup_oem_mitb5_5pct --seed 0 --gpu-ids 0
# ---------------------------------------------------------------

_base_ = ['./sup_oem_daformer_mitb5.py']

model = dict(pretrained='pretrained/mit_b5.pth')
data = dict(train=dict(split='train_5pct_labeled.txt'))

name = 'sup_oem_daformer_mitb5_5pct'
