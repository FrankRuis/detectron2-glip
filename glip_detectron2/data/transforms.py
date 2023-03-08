import numpy as np
import random
import cv2
import torch

from detectron2.data.transforms import HFlipTransform, Transform
from torchvision.transforms import functional as F


def build_transforms(cfg, is_train=True):
    if is_train:
        if len(cfg.AUGMENT.MULT_MIN_SIZE_TRAIN) > 0:
            min_size = cfg.AUGMENT.MULT_MIN_SIZE_TRAIN
        else:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_horizontal_prob = cfg.AUGMENT.FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.AUGMENT.VERTICAL_FLIP_PROB_TRAIN
        brightness = cfg.AUGMENT.BRIGHTNESS
        contrast = cfg.AUGMENT.CONTRAST
        saturation = cfg.AUGMENT.SATURATION
        hue = cfg.AUGMENT.HUE

        crop_prob = cfg.AUGMENT.CROP_PROB
        min_ious = cfg.AUGMENT.CROP_MIN_IOUS
        min_crop_size = cfg.AUGMENT.CROP_MIN_SIZE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0

    fix_res = cfg.INPUT.FIX_RES
    if cfg.INPUT.FORMAT is not '':
        input_format = cfg.INPUT.FORMAT
    elif cfg.INPUT.TO_BGR255:
        input_format = 'bgr255'
    else:
        input_format = 'rgb'
    normalize_transform = Normalize(
        mean=cfg.MODEL.PIXEL_MEAN, std=cfg.MODEL.PIXEL_STD, format=input_format
    )

    transform = Compose(
        [
            # Resize(min_size, max_size, restrict=fix_res),
            ToTensor(),
            # RandomHorizontalFlip(flip_horizontal_prob),
            normalize_transform,
        ]
    )
    return transform


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        if target is None:
            return image
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(Transform):
    def __init__(self, mean, std, format='rgb'):
        self.mean = mean
        self.std = std
        self.format = format.lower()

    def apply_image(self, image):
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).float()

        # if 'bgr' in self.format:
        #     image = image[[2, 1, 0]]
        # if '255' in self.format:
        #     image = image * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image

    def apply_coords(self, coords: np.ndarray):
        return coords


class Resize(object):
    def __init__(self, min_size, max_size, restrict=False):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.restrict = restrict

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if self.restrict:
            return (size, max_size)
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return oh, ow

    def __call__(self, image, target):
        if isinstance(image, np.ndarray):
            old_w, old_h = image.shape[:2]
            image_size = self.get_size(image.shape[:2])
            image = cv2.resize(image, image_size)
            new_size = image_size
        else:
            old_w, old_h = image.size
            image = F.resize(image, self.get_size(image.size))
            new_size = image.size
        if target is not None:
            new_width, new_height = new_size
            sy, sx = new_height / old_h, new_width / old_w
            target._image_size = (new_height, new_width)
            target.boxes.scale(sy, sx)
            # target = target.resize(new_size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            hflip = HFlipTransform(image.shape[1])
            if isinstance(image, np.ndarray):
                image = hflip.apply_image(image)
            else:
                image = F.hflip(image)
            if target is not None:
                target.boxes.tensor = hflip.apply_box(target.boxes.tensor)
        return image, target
