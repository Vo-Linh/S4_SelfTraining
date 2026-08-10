# ---------------------------------------------------------------
# OEM UPPER BOUND (FullSup oracle) — DAFormer(MiT-B5), 100% labels.
# Table IX "100% FullSup (oracle)" row.
#
# Inherits sup_oem_daformer_mitb5.py; only pins the split to the full
# labeled train pool + turns on ImageNet init.
#
# PLACE AT: S4_SelfTraining/configs/daformer/sup_oem_daformer_mitb5_full.py
# RUN:
#   python tools/train.py configs/daformer/sup_oem_daformer_mitb5_full.py \
#     --work-dir work_dirs/E0.2_fullsup_oem_mitb5 --seed 0 --gpu-ids 0
# ---------------------------------------------------------------

_base_ = ['./sup_oem_daformer_mitb5.py']

model = dict(pretrained='pretrained/mit_b5.pth')          # requires the weights file present
data = dict(train=dict(split='train_3500_fixed.txt'))     # 100% of the labeled train pool

name = 'sup_oem_daformer_mitb5_full'
