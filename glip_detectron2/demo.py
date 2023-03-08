import cv2
import torch
import random
from torchvision.transforms import functional as F
import re
import logging
import numpy as np
from typing import List, Union
import nltk
import inflect
from transformers import AutoTokenizer
from torchvision import transforms as T

from detectron2.modeling import build_model
from detectron2.structures.image_list import ImageList
from detectron2.data.transforms import ScaleTransform

from glip_detectron2.utils.checkpoint import translate_glip_dict

engine = inflect.engine()
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import timeit


class GLIPDemo(object):
    def __init__(self,
                 cfg,
                 confidence_threshold=0.7,
                 min_image_size=None,
                 show_mask_heatmaps=False,
                 masks_per_dim=5,
                 load_model=True
                 ):
        self.cfg = cfg.clone()
        if load_model:
            self.model = build_model(cfg)
            self.model.eval()
            self.device = torch.device(cfg.MODEL.DEVICE)
            self.model.to(self.device)
        self.min_image_size = min_image_size
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

        save_dir = cfg.OUTPUT_DIR
        if load_model:
            ckpt = torch.load(cfg.MODEL.WEIGHTS)
            ckpt = translate_glip_dict(ckpt['model'])
            self.model.load_state_dict(ckpt)
            logging.info("Loaded model from {}".format(cfg.MODEL.WEIGHTS))

        self.transforms = self.build_transform()

        # used to make colors for each tokens
        mask_threshold = -1 if show_mask_heatmaps else 0.5
        # self.masker = Masker(threshold=mask_threshold, padding=1)
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

        self.tokenizer = self.build_tokenizer()
        self.color = 255

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.MODEL.PIXEL_MEAN, std=cfg.MODEL.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                Resize(800, 1333),
                # T.Resize(self.min_image_size) if self.min_image_size is not None else lambda x: x,
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def build_tokenizer(self):
        cfg = self.cfg
        tokenizer = None
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "bert-base-uncased":
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            from transformers import CLIPTokenizerFast
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                              from_slow=True, mask_token='ðŁĴĳ</w>')
            else:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                              from_slow=True)
        return tokenizer

    def run_ner(self, caption):
        noun_phrases = find_noun_phrases(caption)
        noun_phrases = [remove_punctuation(phrase) for phrase in noun_phrases]
        noun_phrases = [phrase for phrase in noun_phrases if phrase != '']
        relevant_phrases = noun_phrases
        labels = noun_phrases
        self.entities = labels

        tokens_positive = []

        for entity, label in zip(relevant_phrases, labels):
            try:
                # search all occurrences and mark them as different entities
                for m in re.finditer(entity, caption.lower()):
                    tokens_positive.append([[m.start(), m.end()]])
            except:
                print("noun entities:", noun_phrases)
                print("entity:", entity)
                print("caption:", caption.lower())

        return tokens_positive

    def inference(self, original_image, original_caption):
        predictions = self.compute_prediction(original_image, original_caption)
        top_predictions = self._post_process_fixed_thresh(predictions)
        return top_predictions

    def run_on_web_image(self,
                         original_image,
                         original_caption,
                         thresh=0.5,
                         custom_entities=None,
                         alpha=0.0):
        predictions = self.compute_prediction(original_image, original_caption, custom_entities=custom_entities)
        top_predictions = self._post_process(predictions, thresh)

        result = original_image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        result = self.overlay_entity_names(result, top_predictions)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        return result, top_predictions

    def visualize_with_predictions(self,
                                   original_image,
                                   predictions,
                                   thresh=0.5,
                                   alpha=0.0,
                                   box_pixel=3,
                                   text_size=1,
                                   text_pixel=2,
                                   text_offset=10,
                                   text_offset_original=4,
                                   color=255):
        self.color = color
        height, width = original_image.shape[:-1]

        # TODO resize
        # scale_tf = ScaleTransform(height, width).apply_box(predictions.tensor)
        # predictions = predictions.resize((width, height))
        top_predictions = self._post_process(predictions, thresh)

        result = original_image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions, alpha=alpha, box_pixel=box_pixel)
        result = self.overlay_entity_names(result, top_predictions, text_size=text_size, text_pixel=text_pixel,
                                           text_offset=text_offset, text_offset_original=text_offset_original)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        return result, top_predictions

    def compute_prediction(self, original_image, original_caption, custom_entities=None):
        # image
        image = self.transforms(original_image)
        image_list = ImageList.from_tensors([image], self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # caption
        if isinstance(original_caption, list):
            # we directly provided a list of category names
            caption_string = ""
            tokens_positive = []
            seperation_tokens = " . "
            for word in original_caption:
                tokens_positive.append([len(caption_string), len(caption_string) + len(word)])
                caption_string += word
                caption_string += seperation_tokens

            tokenized = self.tokenizer([caption_string], return_tensors="pt")
            tokens_positive = [tokens_positive]

            original_caption = caption_string
            print(tokens_positive)
        else:
            tokenized = self.tokenizer([original_caption], return_tensors="pt")
            if custom_entities is None:
                tokens_positive = self.run_ner(original_caption)
            else:
                # first extract custom entities, then run NER
                tokens_positive = []
                _caption = original_caption.lower()
                for custom_entity in custom_entities:
                    # _caption.replace(custom_entity.lower(), "")
                    for m in re.finditer(custom_entity.lower(), original_caption.lower()):
                        tokens_positive.append([[m.start(), m.end()]])
                tokens_positive += self.run_ner(_caption)
            print(tokens_positive)
        # process positive map
        positive_map = create_positive_map(tokenized, tokens_positive)

        if self.cfg.MODEL.PROPOSAL_GENERATOR == "VLDyHeadModule":
            plus = 1
        else:
            plus = 0

        positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map, plus=plus)
        self.plus = plus
        self.positive_map_label_to_token = positive_map_label_to_token
        tic = timeit.time.perf_counter()

        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list, captions=[original_caption], positive_map=positive_map_label_to_token)
            predictions = [o.to(self.cpu_device) for o in predictions]
        print("inference time per image: {}".format(timeit.time.perf_counter() - tic))

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        # height, width = original_image.shape[:-1]
        # out_height, out_width = image.shape[1:]
        # sy, sx = height / out_height, width / out_width
        # prediction.pred_boxes.scale(sy, sx)

        # if prediction.has_field("mask"):
        #     # if we have masks, paste the masks in the right position
        #     # in the image, as defined by the bounding boxes
        #     masks = prediction.get("mask")
        #     # always single image is passed at a time
        #     masks = self.masker([masks], [prediction])[0]
        #     prediction.add_field("mask", masks)

        return prediction

    def _post_process_fixed_thresh(self, predictions):
        scores = predictions.get("scores")
        labels = predictions.get("pred_classes").tolist()
        thresh = scores.clone()
        for i, lb in enumerate(labels):
            if isinstance(self.confidence_threshold, float):
                thresh[i] = self.confidence_threshold
            elif len(self.confidence_threshold) == 1:
                thresh[i] = self.confidence_threshold[0]
            else:
                thresh[i] = self.confidence_threshold[lb]
        keep = torch.nonzero(scores > thresh).squeeze(1)
        predictions = predictions[keep]

        scores = predictions.get("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def _post_process(self, predictions, threshold=0.5):
        scores = predictions.scores
        labels = predictions.pred_classes.tolist()
        thresh = scores.clone()
        for i, lb in enumerate(labels):
            if isinstance(self.confidence_threshold, float):
                thresh[i] = threshold
            elif len(self.confidence_threshold) == 1:
                thresh[i] = threshold
            else:
                thresh[i] = self.confidence_threshold[lb]
        keep = torch.nonzero(scores > thresh).squeeze(1)
        predictions = predictions[keep]

        scores = predictions.scores
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = (30 * (labels[:, None] - 1) + 1) * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        try:
            colors = (colors * 0 + self.color).astype("uint8")
        except:
            pass
        return colors

    def overlay_boxes(self, image, predictions, alpha=0.5, box_pixel=3):
        labels = predictions.pred_classes
        boxes = predictions.pred_boxes.tensor

        colors = self.compute_colors_for_labels(labels).tolist()
        new_image = image.copy()
        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            new_image = cv2.rectangle(
                new_image, tuple(top_left), tuple(bottom_right), tuple(color), box_pixel)

        # Following line overlays transparent rectangle over the image
        image = cv2.addWeighted(new_image, alpha, image, 1 - alpha, 0)

        return image

    def overlay_scores(self, image, predictions):
        scores = predictions.get("scores")
        boxes = predictions.bbox

        for box, score in zip(boxes, scores):
            box = box.to(torch.int64)
            image = cv2.putText(image, '%.3f' % score,
                                (int(box[0]), int((box[1] + box[3]) / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        return image

    def overlay_entity_names(self, image, predictions, names=None, text_size=1.0, text_pixel=2, text_offset=10,
                             text_offset_original=4):
        scores = predictions.scores.tolist()
        labels = predictions.pred_classes.tolist()
        new_labels = []
        if self.cfg.MODEL.PROPOSAL_GENERATOR.NAME == "VLDyHeadModule":
            plus = 1
        else:
            plus = 0
        self.plus = plus
        if self.entities and self.plus:
            for i in labels:
                if i <= len(self.entities):
                    new_labels.append(self.entities[i - self.plus])
                else:
                    new_labels.append('object')
            # labels = [self.entities[i - self.plus] for i in labels ]
        else:
            new_labels = ['object' for i in labels]
        boxes = predictions.pred_boxes.tensor

        template = "{}:{:.2f}"
        previous_locations = []
        for box, score, label in zip(boxes, scores, new_labels):
            x, y = box[:2]
            s = template.format(label, score).replace("_", " ").replace("(", "").replace(")", "")
            for x_prev, y_prev in previous_locations:
                if abs(x - x_prev) < abs(text_offset) and abs(y - y_prev) < abs(text_offset):
                    y -= text_offset

            cv2.putText(
                image, s, (int(x), int(y) - text_offset_original), cv2.FONT_HERSHEY_SIMPLEX, text_size,
                (self.color, self.color, self.color), text_pixel, cv2.LINE_AA
            )
            previous_locations.append((int(x), int(y)))

        return image


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

        return (oh, ow)

    def __call__(self, image, target=None):
        if isinstance(image, np.ndarray):
            image_size = self.get_size(image.shape[:2])
            image = cv2.resize(image, image_size)
            new_size = image_size
        else:
            image = F.resize(image, self.get_size(image.size))
            new_size = image.size
        if target is not None:
            target = target.resize(new_size)
        return image


def scale_boxes(im_size, orig_im_size, boxes):
    ch, cw = im_size
    sy, sx = orig_im_size[0] / ch, orig_im_size[1] / cw
    boxes.scale(sx, sy)


def create_positive_map_label_to_token_from_positive_map(positive_map, plus=0):
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(positive_map[i], as_tuple=True)[0].tolist()
    return positive_map_label_to_token


def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            try:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
            except Exception as e:
                print("beg:", beg, "end:", end)
                print("token_positive:", tokens_positive)
                # print("beg_pos:", beg_pos, "end_pos:", end_pos)
                raise e
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


def find_noun_phrases(caption: str) -> List[str]:
    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = list()
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            noun_phrases.append(' '.join(t[0] for t in subtree.leaves()))

    return noun_phrases


def remove_punctuation(text: str) -> str:
    punct = ['|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^',
             '\'', '\"', '’', '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
             ]
    for p in punct:
        text = text.replace(p, '')
    return text.strip()
