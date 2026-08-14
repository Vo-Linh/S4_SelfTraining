# ---------------------------------------------------------------
# DAPCN-SSL on OpenEarthMap with 1/16 labeled data (6.25%).
#
# Requires disjoint split files in the configured data root:
#   train_1_16_labeled.txt
#   train_1_16_unlabeled.txt
# ---------------------------------------------------------------

_base_ = ['./ssl_oem_dapcn_daformer_mitb5.py']

# Use ImageNet-pretrained MiT-B5 when the checkpoint is available.
model = dict(pretrained='pretrained/mit_b5.pth')

data = dict(
    train=dict(
        labeled=dict(split='train_1_16_labeled.txt'),
        unlabeled=dict(split='train_1_16_unlabeled.txt')))

name = 'ssl_oem_dapcn_mitb5_1_16'
name_dataset = 'openearthmap_ssl_1_16'
