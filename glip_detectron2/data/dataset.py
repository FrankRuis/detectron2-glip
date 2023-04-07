import os
import torch
import logging

from transformers import AutoTokenizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

from .modulated_coco import GroundingMapper, GroundingMapDataset


def from_config(cfg, is_train=True, tokenizer=None):
    if is_train:
        dataset_list = cfg.DATASETS.TRAIN
    else:
        dataset_list = cfg.DATASETS.TEST

    if len(cfg.DATASETS.REGISTER) > 0:
        for task, attrs in cfg.DATASETS.REGISTER.items():
            if f'{attrs.name}_{task}' in DatasetCatalog:
                continue
            if attrs.type == "COCO":
                ann_file = os.path.join(cfg.DATASETS.DATA_ROOT, attrs.ann_file)
                img_dir = os.path.join(cfg.DATASETS.DATA_ROOT, attrs.img_dir)
                register_coco_instances(f'{attrs.name}_{task}', {}, ann_file, img_dir)
            else:
                raise NotImplementedError(f"Dataset type {attrs.type} not implemented")

    if tokenizer is None:
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            from transformers import CLIPTokenizerFast
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", from_slow=True, mask_token='ðŁĴĳ</w>')
            else:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", from_slow=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)

    mapper = GroundingMapper(cfg, is_train, tokenizer=tokenizer)
    datasets = []
    for dataset_name in dataset_list:
        if cfg.DATASETS.OVERRIDE_CLASS_NAMES and cfg.DATASETS.USE_OVERRIDE_CLASS_NAMES:
            if dataset_name in MetadataCatalog:
                meta = MetadataCatalog.get(dataset_name)
                if isinstance(cfg.DATASETS.OVERRIDE_CLASS_NAMES[0], dict):
                    meta.override_classes = [e['name'] for e in sorted(cfg.DATASETS.OVERRIDE_CLASS_NAMES,
                                                                       key=lambda x: x['id'])]
                else:
                    meta.override_classes = cfg.DATASETS.OVERRIDE_CLASS_NAMES
                logging.info(f"Overriding class names for {dataset_name}")
            else:
                raise ValueError(f"Dataset {dataset_name} not found in MetadataCatalog")

        if is_train and cfg.DATASETS.REPEAT_DATA > 1:
            DupDataset = create_duplicate_dataset(GroundingMapDataset, cfg.DATASETS.REPEAT_DATA)
            dataset = DupDataset(dataset_name, mapper)
        else:
            dataset = GroundingMapDataset(dataset_name, mapper)
        dataset.init_collate_fn(cfg.DATALOADER.SIZE_DIVISIBILITY)
        datasets.append(dataset)

    return datasets if len(datasets) > 1 else datasets[0]


def create_duplicate_dataset(DatasetBaseClass, repeat):
    class DupDataset(DatasetBaseClass):
        def __init__(self, *args, **kwargs):
            super(DupDataset, self).__init__(*args, **kwargs)

            self.repeat = repeat
            self.length = super(DupDataset, self).__len__()

        def __len__(self):
            return self.repeat * self.length

        def __getitem__(self, index):
            true_index = index % self.length
            return super(DupDataset, self).__getitem__(true_index)

        def init_collate_fn(self, size_divisibility):
            super(DupDataset, self).init_collate_fn(size_divisibility)

    return DupDataset
