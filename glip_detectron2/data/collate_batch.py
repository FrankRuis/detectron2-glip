# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from detectron2.structures import ImageList


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisibility=0):
        self.size_divisibility = size_divisibility

    def __call__(self, batch):
        # pytorch lightning adds the batch index to the batch, which does not work with the GLIP collator
        if isinstance(batch[-1], int):
            return batch

        images = ImageList.from_tensors([b['image'] for b in batch], self.size_divisibility)
        instances = [b['instances'] for b in batch] if 'instances' in batch[0] else None
        img_ids = [b['image_id'] for b in batch]
        captions = [b['caption'] for b in batch] if 'caption' in batch[0] else None
        positive_map = None
        positive_map_eval = None
        image_sizes = [(b['height'], b['width']) for b in batch] if 'height' in batch[0] else None

        if instances is not None and isinstance(instances[0], dict):
            return images, instances, img_ids, captions, positive_map, positive_map_eval, image_sizes

        if 'positive_map' in batch[0]:
            # we batch the positive maps here
            # Since in general each batch element will have a different number of boxes,
            # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
            max_len = max([b['positive_map'].shape[1] for b in batch])
            nb_boxes = sum([b['positive_map'].shape[0] for b in batch])
            batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
            cur_count = 0
            for b in batch:
                cur_pos = b['positive_map']
                batched_pos_map[cur_count: cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
                cur_count += len(cur_pos)

            assert cur_count == len(batched_pos_map)
            positive_map = batched_pos_map.float()

        if "positive_map_eval" in batch[0]:
            # we batch the positive maps here
            # Since in general each batch element will have a different number of boxes,
            # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
            max_len = max([b['positive_map_eval'].shape[1] for b in batch])
            nb_boxes = sum([b['positive_map_eval'].shape[0] for b in batch])
            batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
            cur_count = 0
            for b in batch:
                cur_pos = b['positive_map_eval']
                batched_pos_map[cur_count: cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
                cur_count += len(cur_pos)

            assert cur_count == len(batched_pos_map)
            # assert batched_pos_map.sum().item() == sum([v["positive_map"].sum().item() for v in batch[1]])
            positive_map_eval = batched_pos_map.float()

        return images, instances, img_ids, captions, positive_map, positive_map_eval, image_sizes
