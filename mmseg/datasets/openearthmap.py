# ---------------------------------------------------------------
# OpenEarthMap dataset -- official 8-class benchmark taxonomy.
#
# Aerial RGB tiles (~1000x1000, some 1024x1024) stored as .tif, with
# .tif uint8 label maps whose RAW values are 0..8:
#
#   0  unknown  (nodata / unlabeled)   -> dropped
#   1  bareland        5  tree
#   2  rangeland       6  water
#   3  developed       7  agriculture
#   4  road            8  building
#
# reduce_zero_label=True is the default here: raw label 0 ("unknown") is
# mapped to ignore_index and classes 1..8 shift down to 0..7, so
# num_classes=8. This is the setup the official OpenEarthMap benchmark
# reports and the one the DAPCN-SSL configs use, which keeps every mIoU
# in this repo directly comparable.
#
# It must hold in BOTH places: this class (for evaluation, via
# get_gt_seg_maps) and the training pipeline's LoadAnnotations
# (reduce_zero_label=True). Setting only one silently corrupts either
# the metric or the training labels.
#
# Only the .tif suffixes, the CLASSES/PALETTE and the reduce_zero_label
# default differ from CustomDataset; everything else (split, data_root,
# ignore_index, pipeline, ...) passes straight through, and any of the
# defaults can still be overridden from the config.
# ---------------------------------------------------------------

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class OpenEarthMapDataset(CustomDataset):

    CLASSES = (
        'bareland', 'rangeland', 'developed', 'road',
        'tree', 'water', 'agriculture', 'building',
    )

    PALETTE = [
        [128, 0, 0], [0, 255, 36], [148, 148, 148], [255, 255, 255],
        [34, 97, 38], [0, 69, 255], [75, 181, 73], [222, 31, 7],
    ]

    def __init__(self, **kwargs):
        kwargs.setdefault('img_suffix', '.tif')
        kwargs.setdefault('seg_map_suffix', '.tif')
        kwargs.setdefault('reduce_zero_label', True)
        super(OpenEarthMapDataset, self).__init__(**kwargs)
