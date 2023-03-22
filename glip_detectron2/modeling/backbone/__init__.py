from collections import OrderedDict

from detectron2.layers import ShapeSpec
from torch import nn

from detectron2.modeling import Backbone
from detectron2.modeling import BACKBONE_REGISTRY, SwinTransformer
from detectron2.modeling.backbone.fpn import FPN, LastLevelP6P7
from glip_detectron2.layers import DropBlock2D, DyHead


class SwinTFPNRetinaNet(Backbone):
    def __init__(self, fpn, dyhead=None):
        super().__init__()
        self.fpn = fpn
        self.dyhead = dyhead

    def forward(self, x):
        x = self.fpn(x)
        if self.dyhead is not None:
            x = self.dyhead(x)

        return x

    def output_shape(self):
        if self.dyhead is not None:
            return self.dyhead.output_shape()
        return self.fpn.output_shape()


@BACKBONE_REGISTRY.register()
def build_retinanet_swint_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
        input_shape: ShapeSpec of the input image
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    if cfg.MODEL.SWINT.VERSION == "v1":
        bottom_up = build_swint_backbone(cfg, input_shape)
        bottom_up.norm0 = nn.Identity()
    else:
        raise NotImplementedError("Only SwinTransformer v1 is supported for now")
    in_features = cfg.MODEL.SWINT.OUT_FEATURES
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
    in_channels_p6p7 = cfg.MODEL.BACKBONE.OUT_CHANNELS
    # TODO GLIP uses DropBlock2D between the top block and the FPN output
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels, in_feature="p5"),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    dyhead = DyHead(cfg, cfg.MODEL.BACKBONE.OUT_CHANNELS) if cfg.MODEL.FPN.USE_DYHEAD else None
    return SwinTFPNRetinaNet(backbone, dyhead=dyhead)


def build_swint_backbone(cfg, input_shape):
    """
    Create a SwinT instance from config.

    Returns:
        VoVNet: a :class:`VoVNet` instance.
    """
    return SwinTransformer(
        patch_size=4,
        in_chans=input_shape.channels,
        embed_dim=cfg.MODEL.SWINT.EMBED_DIM,
        depths=cfg.MODEL.SWINT.DEPTHS,
        num_heads=cfg.MODEL.SWINT.NUM_HEADS,
        window_size=cfg.MODEL.SWINT.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.SWINT.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=cfg.MODEL.SWINT.DROP_PATH_RATE,
        norm_layer=nn.LayerNorm,
        ape=cfg.MODEL.SWINT.APE,
        patch_norm=True,
        frozen_stages=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT,
        out_indices=cfg.MODEL.BACKBONE.OUT_INDICES
    )
