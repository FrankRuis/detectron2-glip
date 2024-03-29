import random
import yaml
import torch

from torch import nn
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.proposal_generator.build import build_proposal_generator
from detectron2.structures import ImageList
from transformers import AutoTokenizer

from ..language_backbone import build_language_backbone


@META_ARCH_REGISTRY.register()
class GeneralizedVLRCNN(nn.Module):

    def __init__(self, cfg):
        super(GeneralizedVLRCNN, self).__init__()
        self.cfg = cfg

        # visual encoder
        self.backbone = build_backbone(cfg)

        # language encoder
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            # self.tokenizer = build_tokenizer("clip")
            from transformers import CLIPTokenizerFast
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                print("Reuse token 'ðŁĴĳ</w>' (token_id = 49404) for mask token!")
                self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                   from_slow=True, mask_token='ðŁĴĳ</w>')
            else:
                self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                   from_slow=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)
        self.tokenizer_vocab = self.tokenizer.get_vocab()
        self.tokenizer_vocab_ids = [item for key, item in self.tokenizer_vocab.items()]

        self.language_backbone = build_language_backbone(cfg)
        self.rpn = build_proposal_generator(cfg, [self.backbone.output_shape()[e] for e in self.backbone.output_shape()]) # self.cfg.MODEL.RPN.IN_FEATURES]) or ANCHOR_GENERATOR.STRIDES
        self.roi_heads = build_roi_heads(cfg, [self.backbone.output_shape()[e] for e in self.cfg.MODEL.ROI_HEADS.IN_FEATURES]) if not self.cfg.MODEL.RPN_ONLY else None
        self.DEBUG = cfg.MODEL.DEBUG

        self.freeze_backbone = cfg.MODEL.BACKBONE.FREEZE
        self.freeze_fpn = cfg.MODEL.FPN.FREEZE
        self.freeze_rpn = cfg.MODEL.RPN.FREEZE
        self.add_linear_layer = cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER

        self.force_boxes = cfg.MODEL.RPN.FORCE_BOXES

        if cfg.MODEL.LINEAR_PROB:
            assert cfg.MODEL.BACKBONE.FREEZE, "For linear probing, backbone should be frozen!"
            if hasattr(self.backbone, 'fpn'):
                assert cfg.MODEL.FPN.FREEZE, "For linear probing, FPN should be frozen!"
        self.linear_prob = cfg.MODEL.LINEAR_PROB
        self.freeze_cls_logits = cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            # disable cls_logits
            if hasattr(self.rpn.head, 'cls_logits'):
                for p in self.rpn.head.cls_logits.parameters():
                    p.requires_grad = False

        self.freeze_language_backbone = self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE
        if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            for p in self.language_backbone.parameters():
                p.requires_grad = False

        if self.cfg.GLIPKNOW.KNOWLEDGE_FILE:
            with open(self.cfg.GLIPKNOW.KNOWLEDGE_FILE, 'r') as fp:
                self.class_name_to_knowledge = yaml.load(fp, Loader=yaml.CLoader)
            self.class_name_list = sorted([k for k in self.class_name_to_knowledge])

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(GeneralizedVLRCNN, self).train(mode)
        if self.freeze_backbone:
            self.backbone.body.eval()
            for p in self.backbone.body.parameters():
                p.requires_grad = False
        if self.freeze_fpn:
            self.backbone.fpn.eval()
            for p in self.backbone.fpn.parameters():
                p.requires_grad = False
        if self.freeze_rpn:
            if hasattr(self.rpn, 'head'):
                self.rpn.head.eval()
            for p in self.rpn.parameters():
                p.requires_grad = False
        if self.linear_prob:
            if self.rpn is not None:
                for key, value in self.rpn.named_parameters():
                    if not \
                            ('bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                        value.requires_grad = False
            if self.roi_heads is not None:
                for key, value in self.roi_heads.named_parameters():
                    if not (
                            'bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                        value.requires_grad = False
        if self.freeze_cls_logits:
            if hasattr(self.rpn.head, 'cls_logits'):
                self.rpn.head.cls_logits.eval()
                for p in self.rpn.head.cls_logits.parameters():
                    p.requires_grad = False
        if self.add_linear_layer:
            if self.rpn is not None:
                for key, p in self.rpn.named_parameters():
                    if 'tunable_linear' in key:
                        p.requires_grad = True

        if self.freeze_language_backbone:
            self.language_backbone.eval()
            for p in self.language_backbone.parameters():
                p.requires_grad = False

    def forward(self,
                images,
                targets=None,
                captions=None,
                positive_map=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[Instances]): ground-truth boxes and labels present in the image (optional)

            mask_black_list: batch x 256, indicates whether a certain token is maskable or not

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[Instances] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if not isinstance(images, ImageList):
            images = ImageList.from_tensors(images, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        device = images.tensor.device

        # language embedding
        language_dict_features = {}

        if captions is not None:
            tokenized = self.tokenizer.batch_encode_plus(captions,
                                                         max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                         padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                         return_special_tokens_mask=True,
                                                         return_tensors='pt',
                                                         truncation=True).to(device)
            input_ids = tokenized.input_ids

            tokenizer_input = {"input_ids": input_ids,
                               "attention_mask": tokenized.attention_mask}

            if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
                with torch.no_grad():
                    language_dict_features = self.language_backbone(tokenizer_input)
            else:
                language_dict_features = self.language_backbone(tokenizer_input)

            # ONE HOT
            if self.cfg.DATASETS.ONE_HOT:
                new_masks = torch.zeros_like(language_dict_features['masks'],
                                             device=language_dict_features['masks'].device)
                new_masks[:, :self.cfg.MODEL.DYHEAD.NUM_CLASSES+1] = 1
                language_dict_features['masks'] = new_masks

            # MASK ALL SPECIAL TOKENS
            if self.cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL:
                language_dict_features["masks"] = 1 - tokenized.special_tokens_mask

        # visual embedding
        swint_feature_c4 = None
        if 'vl' in self.cfg.MODEL.SWINT.VERSION:
            # the backbone only updates the "hidden" field in language_dict_features
            inputs = {"img": images.tensor, "lang": language_dict_features}
            visual_features, language_dict_features, swint_feature_c4 = self.backbone(inputs)
        else:
            visual_features = self.backbone(images.tensor)

        # rpn force boxes
        if targets:
            targets = [target.to(device) for target in targets if target is not None]

        if self.force_boxes:
            raise NotImplementedError
        else:
            proposals, proposal_losses, fused_visual_features = self.rpn(images, visual_features, targets,
                                                                         language_dict_features, positive_map,
                                                                         captions, swint_feature_c4)
        if self.roi_heads:
            raise NotImplementedError
        else:
            # RPN-only models don't have roi_heads
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
