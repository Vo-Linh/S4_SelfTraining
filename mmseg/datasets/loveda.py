# LoveDA dataset (7 land-cover classes) — mmseg 0.16 port.
#
# The standalone mmsegmentation ships a 1.x version (BaseSegDataset / METAINFO)
# that is NOT compatible with this 0.16 fork. This is a CustomDataset-based
# reimplementation styled after openearthmap.py.
#
# PLACE AT: S4_SelfTraining/mmseg/datasets/loveda.py
# and register in mmseg/datasets/__init__.py (import + __all__).
#
# LoveDA masks: 0 = no-data (ignore). reduce_zero_label=True maps 1..7 -> 0..6.
import os.path as osp

import mmcv
from mmcv.utils import print_log

from .builder import DATASETS
from .custom import CustomDataset
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class LoveDADataset(CustomDataset):
    """LoveDA dataset - 7 land cover classes (flat layout, like OpenEarthMap)."""

    CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest',
               'agricultural')

    PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
               [159, 129, 183], [0, 255, 0], [255, 195, 128]]

    def __init__(self,
                 pipeline,
                 img_dir='images/train',
                 img_suffix='.png',
                 ann_dir='annotations/train',
                 seg_map_suffix='.png',
                 split=None,
                 ann_file=None,
                 reduce_zero_label=True,
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
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = osp.basename(line.strip())
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        img_info['ann'] = dict(seg_map=img_name + seg_map_suffix)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    img_info['ann'] = dict(
                        seg_map=img.replace(img_suffix, seg_map_suffix))
                img_infos.append(img_info)
        print_log(
            f'Loaded {len(img_infos)} images from {img_dir}',
            logger=get_root_logger())
        return img_infos
