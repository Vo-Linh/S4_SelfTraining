# ---------------------------------------------------------------
# DualBank (cross) on the BIASED OpenEarthMap 1/8 split.
# Data:     /home/p40/project/data/OpenEarthMap_wo_xBD_coco_1500_2000
#           splits_biased/train_filename_only_1_8.txt            (187 L)
#           splits_biased/train_filename_only_unlabeled_1_8.txt  (1313 U)
#           val: the same 2000-image val split as every other run
# ---------------------------------------------------------------

_base_ = ['./oem_ssl_1_8_dualbank_cross_mitb5.py']

_data_root = '/home/p40/project/data/OpenEarthMap_wo_xBD_coco_1500_2000/'

data = dict(
    train=dict(
        labeled=dict(
            data_root=_data_root,
            split='splits_biased/train_filename_only_1_8.txt'),
        unlabeled=dict(
            data_root=_data_root,
            split='splits_biased/train_filename_only_unlabeled_1_8.txt'),
        rare_class_sampling=dict(
            stats_file='/home/p40/data_hdd_4TB/S4_SelfTraining/rcs_stats/'
                       'oem_biased_1_8/sample_class_stats.json',
            samples_file='/home/p40/data_hdd_4TB/S4_SelfTraining/rcs_stats/'
                         'oem_biased_1_8/samples_with_class.json')),
    val=dict(data_root=_data_root),
    test=dict(data_root=_data_root))

name = 'oem_biased_ssl_1_8_dualbank_cross_mitb5'
exp = 'oem_biased_ssl'
name_uda = 'dapcn_ssl_dualbank_cross_biased'
