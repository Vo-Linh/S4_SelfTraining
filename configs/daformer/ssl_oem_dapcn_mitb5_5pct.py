# ---------------------------------------------------------------
# P1 / Table IX: DAPCN-SSL on OpenEarthMap @ 5% labeled
#
# Paper protocol (STIC_semi §V-A): SegFormer MiT-B5 (ImageNet-1K
# pretrained), 512x512 crops, batch 8 (4 labeled + 4 unlabeled, i.e.
# samples_per_gpu=4), AdamW 6e-5 + poly(0.9).
#
# Everything except the labeled/unlabeled splits and the pretrained
# backbone is inherited from the proven DAPCN OEM config, so the
# training recipe is identical across the 1/5/10% ratios and the only
# variable is the label budget.
#
# Splits are DISJOINT (labeled images do NOT reappear in the unlabeled
# pool). Generate them first:
#   python tools/make_ssl_splits.py \
#     --pool $DATA/train_3500_fixed.txt --out-dir $DATA --ratios 1 5 10
# ---------------------------------------------------------------

_base_ = ['./ssl_oem_dapcn_daformer_mitb5.py']

# Paper uses an ImageNet-1K pretrained MiT-B5 encoder. Drop the weights at
# pretrained/mit_b5.pth -- without them the numbers are NOT paper-faithful.
model = dict(pretrained='pretrained/mit_b5.pth')

data = dict(
    train=dict(
        labeled=dict(split='train_5pct_labeled.txt'),
        unlabeled=dict(split='train_5pct_unlabeled.txt')))

name = 'ssl_oem_dapcn_mitb5_5pct'
name_dataset = 'openearthmap_ssl_5pct'
