import copy
import logging
import functools

import torch
import numpy as np
from detectron2.config import configurable
from detectron2.data import MapDataset, DatasetMapper, get_detection_dataset_dicts, MetadataCatalog, \
    detection_utils as utils, transforms as T

from .grounding import convert_od_to_grounding_simple
from .collate_batch import BatchCollator
from .transforms import Normalize


class GroundingMapper(DatasetMapper):
    @configurable
    def __init__(self, is_train, *, augmentations, image_format, use_instance_mask=False, use_keypoint=False,
                 instance_mask_format="polygon", keypoint_hflip_indices=None, precomputed_proposal_topk=None,
                 recompute_boxes=False, use_prompt_template=False, prompt_template=None, shuffle_caption=False,
                 return_tokens=True, tokenizer=None, token_separation=' ', max_query_len=256,
                 random_sample_negative=-1, normalize=None):
        super().__init__(is_train, augmentations=augmentations, image_format=image_format,
                                        use_instance_mask=use_instance_mask, use_keypoint=use_keypoint,
                                        instance_mask_format=instance_mask_format,
                                        keypoint_hflip_indices=keypoint_hflip_indices,
                                        precomputed_proposal_topk=precomputed_proposal_topk,
                                        recompute_boxes=recompute_boxes)
        self.use_prompt_template = use_prompt_template
        self.prompt_template = prompt_template
        self.shuffle_caption = shuffle_caption
        self.return_tokens = return_tokens
        self.tokenizer = tokenizer
        self.token_separation = token_separation
        self.max_query_len = max_query_len
        self.random_sample_negative = random_sample_negative
        self.normalize = normalize

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        if cfg.INPUT.FORMAT != '':
            input_format = cfg.INPUT.FORMAT
        elif cfg.INPUT.TO_BGR255:
            input_format = 'bgr255'
        else:
            input_format = 'rgb'
        ret = {
            **ret,
            "use_prompt_template": cfg.DATASETS.USE_PROMPT_TEMPLATE,
            "max_query_len": cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
            "random_sample_negative": cfg.DATASETS.RANDOM_SAMPLE_NEG,
            "shuffle_caption": cfg.DATASETS.SHUFFLE_CAPTION,
            "token_separation": cfg.DATASETS.TOKEN_SEPARATION,
            "normalize": Normalize(mean=cfg.MODEL.PIXEL_MEAN, std=cfg.MODEL.PIXEL_STD, format=input_format)
        }
        ret["augmentations"] = ret["augmentations"] + [
            # Normalize(mean=cfg.MODEL.PIXEL_MEAN, std=cfg.MODEL.PIXEL_STD, format=input_format)
        ]

        if cfg.DATASETS.USE_PROMPT_TEMPLATE and cfg.DATASETS.OVERRIDE_CLASS_NAMES \
                and 'prompt_template' in cfg.DATASETS.OVERRIDE_CLASS_NAMES[0]:
            prompt_template = []
            for cat in sorted(cfg.DATASETS.OVERRIDE_CLASS_NAMES, key=lambda x: x['id']):
                prompt_template.append(cat['prompt_template'] if is_train else cat.get('prompt_template_val', cat['prompt_template']))
            ret["prompt_template"] = prompt_template

        return ret

    def modified_call(self, dataset_dict):
        """
        Modified to not remove annotations and sem_seg_file_name.

        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format='BGR')
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        # if isinstance(image, np.ndarray):
        #     dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        #     dataset_dict["image"] = dataset_dict["image"][[2, 1, 0]]
        # else:
        #     dataset_dict["image"] = image
        image_shape = image.shape[:2]  # h, w
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict

    def __call__(self, dataset_dict, metadata=None, tokenizer=None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        assert metadata is not None, "Metadata must be provided for `GroundingMapper`"
        dataset_dict = self.modified_call(dataset_dict)
        # if not hasattr(self, 'caption'):
        self.init_prompt(metadata)
        dataset_dict["caption"] = self.caption
        self.update_instances(dataset_dict)
        dataset_dict['image'] = self.normalize.apply_image(dataset_dict['image'].float())

        return dataset_dict

    def init_prompt(self, metadata):
        if hasattr(metadata, 'override_classes'):
            classes = metadata.override_classes
        else:
            classes = metadata.thing_classes
        ind_to_class = {i: class_ for i, class_ in enumerate(classes)}
        self.label_to_positions, self.caption = convert_od_to_grounding_simple(
            ind_to_class=ind_to_class,
            shuffle_caption=self.shuffle_caption,
            token_separation=self.token_separation,
            prompt_template=self.prompt_template,
        )

    def update_instances(self, dataset_dict):
        if "instances" in dataset_dict:
            dataset_dict["instances"].set('tokens_positive', [
                [self.label_to_positions[e.item()]] for e in dataset_dict["instances"].gt_classes if e.item() in self.label_to_positions
            ])
        if self.return_tokens and self.tokenizer is not None and "instances" in dataset_dict:
            tokenized = self.tokenizer(dataset_dict["caption"], return_tensors="pt", max_length=self.max_query_len,
                                       truncation=True)
            dataset_dict['positive_map'] = create_positive_map(tokenized, dataset_dict["instances"].tokens_positive,
                                                               max_len=self.max_query_len)
            # positive_map_od was only used for shallow contrastive loss
            # dataset_dict['positive_map_od'] = create_positive_map_for_od_labels(tokenized, self.label_to_positions,
            #                                                                     max_len=self.max_query_len)


class GroundingMapDataset(MapDataset):
    def __init__(self, name, map_func):
        self.collate_fn = BatchCollator()
        dataset = get_detection_dataset_dicts(name, filter_empty=True)
        self.metadata = MetadataCatalog.get(name)
        if hasattr(self.metadata, 'override_classes'):
            classes = self.metadata.override_classes
        else:
            classes = self.metadata.thing_classes
        self.names = {i: class_ for i, class_ in enumerate(classes)}
        self.is_train = map_func.is_train
        assert isinstance(map_func, GroundingMapper), "`map_func` must be an instance of `GroundingMapper`"
        super().__init__(dataset, functools.partial(map_func, metadata=self.metadata))

    def init_collate_fn(self, size_divisibility=0):
        self.collate_fn = BatchCollator(size_divisibility)


def create_positive_map(tokenized, tokens_positive, max_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), max_len), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos: end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)
