from collections import OrderedDict
from torch import nn

from glip_detectron2.modeling import registry
from . import bert_model


@registry.LANGUAGE_BACKBONES.register()
class BertBaseUncased(nn.Sequential):
    def __init__(self, cfg):
        body = bert_model.BertEncoder(cfg)
        super().__init__(OrderedDict([("body", body)]))


def build_backbone(cfg):
    # TODO different way of getting the right name
    backbone_name = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE.title().replace('-', '')
    assert backbone_name in registry.LANGUAGE_BACKBONES, \
        "cfg.MODEL.LANGUAGE_BACKBONE.TYPE: {} is not registered in registry".format(
            backbone_name
        )
    return registry.LANGUAGE_BACKBONES.get(backbone_name)(cfg)
