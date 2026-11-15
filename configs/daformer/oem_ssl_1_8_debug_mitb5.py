# ---------------------------------------------------------------
# DEBUG RUN — DAPCN-SSL + DAFormer/MiT-B5, OpenEarthMap 1/8
#
# Purpose: confirm the pipeline runs end to end and converges
# sensibly.  NOT a result-producing run — do not report numbers from
# it, and do not compare them against the 40k baseline.
#
# Differences from oem_ssl_1_8_dapcn_daformer_mitb5.py:
#   max_iters        40000 -> 10000
#   evaluation       every 4000 -> every 1000
#   val set          2000 -> 100 fixed tiles (val_filename_only_debug100.txt)
#   samples_per_gpu  2 -> 4
#
# Usage:
#   python tools/train.py configs/daformer/oem_ssl_1_8_debug_mitb5.py
#
# What to look for:
#   - decode.loss_seg falls steadily
#   - debug/teacher_mIoU rises; debug/proto_mIoU is not stuck near 0
#   - no spike in loss right after iter 1000, where BOTH the
#     pseudo-label warmup ends and prototype correction switches on
#   - val mIoU at 10k is well above chance (1/8 = 0.125)
# ---------------------------------------------------------------

_base_ = ['./oem_ssl_1_8_dapcn_daformer_mitb5.py']

# Batch 4 instead of 2. This also gives DynamicAnchorModule 1024
# instead of 512 feature vectors per step to place its 96 prototypes,
# which is thin either way at a 16x16 encoder resolution.
# NOTE: batch size changes results — keep it fixed across every run
# you intend to compare. If this OOMs on a 24 GB card, drop to 2.
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    val=dict(split='val_filename_only_debug100.txt'),
    test=dict(split='val_filename_only_debug100.txt'))

runner = dict(type='IterBasedRunner', max_iters=10000)

# Frequent, cheap evaluation: 100 tiles instead of 2000 keeps each
# eval to well under a minute, so 10 of them cost almost nothing.
evaluation = dict(interval=1000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=2)

# Log more often than the default 50 — a short run needs a denser
# curve to judge convergence.
log_config = dict(interval=25)

name = 'oem_ssl_1_8_debug_mitb5'
exp = 'oem_ssl_debug'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x4_10k'
