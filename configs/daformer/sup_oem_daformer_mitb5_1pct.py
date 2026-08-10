# ---------------------------------------------------------------
# OEM LOWER BOUND (SupOnly) — DAFormer(MiT-B5), 1% labels.
# Table IX "SupOnly" row @ 1%.
#
# PLACE AT: S4_SelfTraining/configs/daformer/sup_oem_daformer_mitb5_1pct.py
# RUN:
#   python tools/train.py configs/daformer/sup_oem_daformer_mitb5_1pct.py \
#     --work-dir work_dirs/E0.1_sup_oem_mitb5_1pct --seed 0 --gpu-ids 0
# ---------------------------------------------------------------

_base_ = ['./sup_oem_daformer_mitb5.py']

model = dict(pretrained='pretrained/mit_b5.pth')
data = dict(train=dict(split='train_1pct_labeled.txt'))

name = 'sup_oem_daformer_mitb5_1pct'
