import os.path as osp

import mmcv
from mmcv.utils import print_log

from .builder import DATASETS
from .custom import CustomDataset
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class OpenEarthMapDataset(CustomDataset):
    """OpenEarthMap dataset - 9 land cover classes.

    Supports two data layouts:

    1. Flat layout (default)::

        data_root/
        ├── images/train/  (*.tif, e.g. tyrolw_25.tif)
        ├── annotations/train/  (*.tif)
        └── train_500_fixed.txt

       Split files use ``location/img_name`` entries (e.g. ``tyrolw/tyrolw_25``).
       The location prefix is stripped to produce flat filenames.

    2. Original nested layout (no split file)::

       All files in ``img_dir`` are scanned directly.
    """

    CLASSES = (
        'class_0', 'class_1', 'class_2', 'class_3', 'class_4',
        'class_5', 'class_6', 'class_7', 'class_8')

    PALETTE = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0]]

    def __init__(self,
                 pipeline,
                 img_dir='images/train',
                 img_suffix='.tif',
                 ann_dir='annotations/train',
                 seg_map_suffix='.tif',
                 split=None,
                 ann_file=None,
                 **kwargs):
        if ann_file is not None and split is None:
            split = ann_file
        super().__init__(
            pipeline=pipeline,
            img_dir=img_dir,
            img_suffix=img_suffix,
            ann_dir=ann_dir,
            seg_map_suffix=seg_map_suffix,
            split=split,
            **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    # Flatten subdirectory: "tyrolw/tyrolw_25" → "tyrolw_25"
                    img_name = osp.basename(img_name)
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(
            f'Loaded {len(img_infos)} images from {img_dir}',
            logger=get_root_logger())
        return img_infos
