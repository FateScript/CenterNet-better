# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os
import os.path as osp

import dl_lib
from dl_lib.data import MetadataCatalog

from .builtin_meta import _get_builtin_metadata
from .pascal_voc import register_pascal_voc
from .register_coco import register_coco_instances

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train":
    ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val":
    ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival":
    ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_minival_100":
    ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017",
                        "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017",
                      "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017",
                       "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017",
                           "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017",
                          "coco/annotations/instances_val2017_100.json"),
}


def register_all_coco(root=osp.join(
        osp.split(osp.split(dl_lib.__file__)[0])[0], "datasets")):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file)
                if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root=osp.join(
        osp.split(osp.split(dl_lib.__file__)[0])[0], "datasets")):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


# Register them all under "./datasets"
register_all_coco()
register_all_pascal_voc()
