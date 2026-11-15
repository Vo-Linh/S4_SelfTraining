# ---------------------------------------------------------------
# DualBank (cross), BIASED OpenEarthMap 1/8 split, full-length run.
#
# Same method and data as oem_biased_ssl_1_8_dualbank_cross_mitb5, but
# the 10k-iter debug settings that file inherits are put back to the
# standard 1/8 training recipe:
#   * 40k iterations, batch 2 labeled + 2 unlabeled per step
#   * evaluation every 2k iterations on the FULL 2000-image val split
#     (the debug chain uses a 100-image subset)
#   * best checkpoint (by mIoU) kept next to the periodic ones
#
# Evaluation is the standard decode path, i.e. branch P1.  P2 (the
# cross-attention branch) is not scored by it.
# ---------------------------------------------------------------

_base_ = ['./oem_biased_ssl_1_8_dualbank_cross_mitb5.py']

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    val=dict(split='val_filename_only_fixed.txt'),
    test=dict(split='val_filename_only_fixed.txt'))

runner = dict(type='IterBasedRunner', max_iters=40000)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=3)
log_config = dict(interval=50)

name = 'oem_biased_ssl_1_8_dualbank_cross_40k_mitb5'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
