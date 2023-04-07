import math
import torch
import torch.nn.functional as F
from torch import nn

from .loss import make_atss_loss_evaluator
# from .anchor_generator import make_anchor_generator_complex
from detectron2.structures import Instances
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from detectron2.layers import ModulatedDeformConv, ShapeSpec
from .inference import make_atss_postprocessor
# from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

from glip_detectron2.layers import DYReLU, Scale
from glip_detectron2.utils.fuse_helper import BiAttentionBlockForCheckpoint
# from maskrcnn_benchmark.layers import Scale, DYReLU, SELayer, ModulatedDeformConv
# from maskrcnn_benchmark.modeling.backbone.fbnet import *
from ..utils import permute_and_flatten
#
# from maskrcnn_benchmark.utils.fuse_helper import FeatureResizer, func_attention, _make_mlp, _make_conv, _make_coord, \
#     BiAttentionBlock, AttentionT2I, BiAttentionBlockForCheckpoint, BertLMPredictionHead
from transformers.models.bert.modeling_bert import BertConfig, BertPreTrainedModel
from transformers.modeling_utils import apply_chunking_to_forward
import torch.utils.checkpoint as checkpoint
from torch.cuda.amp import custom_fwd


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class BoxCoder(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def encode(self, gt_boxes, anchors):
        TO_REMOVE = 1  # TODO remove
        ex_widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        ex_heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ex_ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ex_ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + TO_REMOVE
        gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
        gt_ctr_y = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2

        wx, wy, ww, wh = (10., 10., 5., 5.)
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

        return targets

    def decode(self, preds, anchors):
        anchors = anchors.to(preds.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        wx, wy, ww, wh = (10., 10., 5., 5.)
        dx = preds[:, 0::4] / wx
        dy = preds[:, 1::4] / wy
        dw = preds[:, 2::4] / ww
        dh = preds[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=math.log(1000. / 16))
        dh = torch.clamp(dh, max=math.log(1000. / 16))

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(preds)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)

        return pred_boxes


class Conv3x3Norm(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 groups=1,
                 deformable=False,
                 bn_type=None):
        super(Conv3x3Norm, self).__init__()

        if deformable:
            self.conv = ModulatedDeformConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                            groups=groups)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups)

        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]

        if bn_type == "bn":
            bn_op = nn.BatchNorm2d(out_channels)
        elif bn_type == "sbn":
            bn_op = nn.SyncBatchNorm(out_channels)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=out_channels)
        elif bn_type in ["nsbn", "af"]:
            raise NotImplementedError
        else:
            bn_op = None

        if bn_type is not None:
            self.bn = bn_op
        else:
            self.bn = None

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, **kwargs):
        x = self.conv(input, **kwargs)
        if self.bn:
            x = self.bn(x)
        return x


class DyConv(torch.nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 conv_func=nn.Conv2d,
                 use_dyfuse=True,
                 use_dyrelu=False,
                 use_deform=False
                 ):
        super(DyConv, self).__init__()

        self.DyConv = nn.ModuleList()
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 2))

        if use_dyfuse:
            self.AttnConv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.ReLU(inplace=True))
            self.h_sigmoid = h_sigmoid()
        else:
            self.AttnConv = None

        if use_dyrelu:
            self.relu = DYReLU(in_channels, out_channels)
        else:
            self.relu = nn.ReLU()

        if use_deform:
            self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)
        else:
            self.offset = None

        self.init_weights()

    def init_weights(self):
        for m in self.DyConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        if self.AttnConv is not None:
            for m in self.AttnConv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight.data, 0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, inputs):
        visual_feats = inputs["visual"]
        language_dict_features = inputs["lang"]

        next_x = []
        for level, feature in enumerate(visual_feats):

            conv_args = dict()
            if self.offset is not None:
                offset_mask = self.offset(feature)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, 18:, :, :].sigmoid()
                conv_args = dict(offset=offset, mask=mask)

            temp_fea = [self.DyConv[1](feature, **conv_args)]

            if level > 0:
                temp_fea.append(self.DyConv[2](visual_feats[level - 1], **conv_args))
            if level < len(visual_feats) - 1:
                # the original GLIP code input shape for the offset did not match the shape of the feature map,
                # resulting in an error in detectron2, but taking a subset seems to give similar results
                fw, fh = visual_feats[level + 1].shape[-2:]
                temp_fea.append(F.interpolate(self.DyConv[0](visual_feats[level + 1], offset=offset[:, :, :fw, :fh], mask=mask[:, :, :fw, :fh]), size=[feature.size(2), feature.size(3)], mode='bilinear', align_corners=True))
            mean_fea = torch.mean(torch.stack(temp_fea), dim=0, keepdim=False)

            if self.AttnConv is not None:
                attn_fea = []
                res_fea = []
                for fea in temp_fea:
                    res_fea.append(fea)
                    attn_fea.append(self.AttnConv(fea))

                res_fea = torch.stack(res_fea)
                spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))

                mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)

            next_x.append(mean_fea)

        next_x = [self.relu(item) for item in next_x]

        features_dict = {"visual": next_x,
                         "lang": language_dict_features}

        return features_dict


class BertEncoderLayer(BertPreTrainedModel):
    def __init__(self, config, clamp_min_for_underflow=False, clamp_max_for_overflow=False):
        super().__init__(config)
        self.config = config

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        # from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput
        from glip_detectron2.modeling.rpn.modeling_bert import BertAttention, BertIntermediate, BertOutput

        self.attention = BertAttention(config, clamp_min_for_underflow, clamp_max_for_overflow)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, inputs):
        language_dict_features = inputs["lang"]
        hidden_states = language_dict_features["hidden"]
        attention_mask = language_dict_features["masks"]

        device = hidden_states.device
        input_shape = hidden_states.size()[:-1]
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        self_attention_outputs = self.attention(
            hidden_states,
            extended_attention_mask,
            None,
            output_attentions=False,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        hidden_states = outputs[0]

        language_dict_features["hidden"] = hidden_states

        features_dict = {"visual": inputs["visual"],
                         "lang": language_dict_features
                         }

        return features_dict

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs


class VLFuse(torch.nn.Module):
    """
    Early Fusion Module
    """

    def __init__(self, cfg):
        super(VLFuse, self).__init__()
        self.init_configs(cfg)
        self.cfg = cfg

        self.use_checkpoint = False
        if hasattr(cfg.MODEL.DYHEAD, 'USE_CHECKPOINT'):
            self.use_checkpoint = cfg.MODEL.DYHEAD.USE_CHECKPOINT
            self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        # early fusion module
        print("EARLY FUSION ON, USING {}".format(cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE))
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "MHA-S":
            raise NotImplementedError
        elif cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "MHA-B":
            # bi-direction (text->image, image->text)
            self.b_attn = BiAttentionBlockForCheckpoint(v_dim=self.joint_embedding_size,
                                                        l_dim=self.lang_dim,
                                                        embed_dim=self.embed_dim,
                                                        num_heads=self.n_head,
                                                        hidden_dim=self.i2t_hidden_dim,
                                                        dropout=0.1,
                                                        drop_path=.0,
                                                        init_values=1.0 / cfg.MODEL.DYHEAD.NUM_CONVS,
                                                        cfg=cfg
                                                        )
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL and self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                raise NotImplementedError
        elif cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "SCAN":
            raise NotImplementedError
        elif cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "FILM":
            raise NotImplementedError
        else:
            print("NO FUSION INVOLVED.")

    def init_configs(self, cfg):
        # common params
        self.lang_model = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        self.joint_embedding_size = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_SIZE
        self.joint_embedding_dropout = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_DROPOUT
        self.joint_mlp_layers = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_MLP_LAYERS

        self.max_query_len = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN
        self.n_layers = cfg.MODEL.LANGUAGE_BACKBONE.N_LAYERS
        self.coord_dim = 8
        self.joint_inp_dim = self.coord_dim + self.joint_embedding_size
        self.joint_out_dim = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_OUT_SIZE

        # mha params
        self.n_head = 8
        self.embed_dim = 2048
        self.t2i_hidden_dim = 1024  # 256 * 4
        self.i2t_hidden_dim = 3072  # 768 * 4

        if self.lang_model in ["bert-base-uncased", "roberta-base", "clip"]:
            self.lang_dim = cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM
        else:
            self.lang_dim = 1024

    def forward(self, x):
        visual_features = x["visual"]
        language_dict_features = x["lang"]
        stages = ['p3', 'p4', 'p5', 'p6', 'p7']  # TODO get from config?

        # batch_size = visual_features[0].shape[0]
        # device = visual_features[0].device

        # fused_visual_features = None
        # fused_language_dict_features = None

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "MHA-S":
            language_feature = language_dict_features['hidden']
            mask = language_dict_features['masks']
            # text -> image
            if self.use_checkpoint:
                q0, q1, q2, q3, q4 = checkpoint.checkpoint(
                    self.t2i_attn,
                    visual_features[stages[0]], visual_features[stages[1]],
                    visual_features[stages[2]], visual_features[stages[3]],
                    visual_features[stages[4]],
                    language_feature, language_feature,
                    mask,
                    self.dummy_tensor
                )
            else:
                q0, q1, q2, q3, q4 = self.t2i_attn(
                    visual_features[stages[0]], visual_features[stages[1]],
                    visual_features[stages[2]], visual_features[stages[3]],
                    visual_features[stages[4]],
                    language_feature, language_feature,
                    attention_mask=mask
                )

            fused_visual_features = [q0, q1, q2, q3, q4]
            fused_language_dict_features = language_dict_features

        elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "MHA-B":
            if self.use_checkpoint:
                if stages[0] in visual_features:
                    q0, q1, q2, q3, q4, l0, l1, l2, l3, l4 = checkpoint.checkpoint(self.b_attn,
                                                                                   visual_features[stages[0]], visual_features[stages[1]],
                                                                                   visual_features[stages[2]], visual_features[stages[3]],
                                                                                   visual_features[stages[4]],
                                                                                   language_dict_features['hidden'],
                                                                                   language_dict_features['masks'],
                                                                                   self.dummy_tensor
                                                                                   )
                else:
                    q0, q1, q2, q3, q4, l0, l1, l2, l3, l4 = checkpoint.checkpoint(self.b_attn,
                                                                                   visual_features[0],
                                                                                   visual_features[1],
                                                                                   visual_features[2],
                                                                                   visual_features[3],
                                                                                   visual_features[4],
                                                                                   language_dict_features['hidden'],
                                                                                   language_dict_features['masks'],
                                                                                   self.dummy_tensor
                                                                                   )
            else:
                q0, q1, q2, q3, q4, l0, l1, l2, l3, l4 = self.b_attn(
                    visual_features[stages[0]], visual_features[stages[1]],
                    visual_features[stages[2]], visual_features[stages[3]],
                    visual_features[stages[4]],
                    language_dict_features['hidden'],
                    language_dict_features['masks'],
                    self.dummy_tensor
                )

            fused_visual_features = [q0, q1, q2, q3, q4]
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL and self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                language_features = self.shrink_lang(torch.cat([l0, l1, l2, l3, l4], dim=-1))
            else:
                language_features = l0

            language_dict_features['hidden'] = language_features
            fused_language_dict_features = language_dict_features

        elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "SCAN":
            raise NotImplementedError
        elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "FILM":
            raise NotImplementedError
        else:
            fused_visual_features = visual_features
            fused_language_dict_features = language_dict_features

        features_dict = {"visual": fused_visual_features,
                         "lang": fused_language_dict_features}

        return features_dict


class VLDyHead(torch.nn.Module):
    def __init__(self, cfg):
        super(VLDyHead, self).__init__()
        self.cfg = cfg
        if cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "bert-base-uncased":
            lang_cfg = BertConfig.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE)
        else:
            raise NotImplementedError

        num_classes = cfg.MODEL.DYHEAD.NUM_CLASSES + 50
        num_tokens = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN
        num_anchors = len(cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS) * cfg.MODEL.ANCHOR_GENERATOR.SCALES_PER_OCTAVE
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        channels = cfg.MODEL.DYHEAD.CHANNELS

        if cfg.MODEL.DYHEAD.USE_GN:
            bn_type = ['gn', cfg.MODEL.GROUP_NORM.NUM_GROUPS]
        else:
            raise NotImplementedError

        use_dyrelu = cfg.MODEL.DYHEAD.USE_DYRELU
        use_dyfuse = cfg.MODEL.DYHEAD.USE_DYFUSE
        use_deform = cfg.MODEL.DYHEAD.USE_DFCONV

        if cfg.MODEL.DYHEAD.CONV_FUNC:
            conv_func = lambda i, o, s: eval(cfg.MODEL.DYHEAD.CONV_FUNC)(i, o, s, bn_type=bn_type)
        else:
            conv_func = lambda i, o, s: Conv3x3Norm(i, o, s, deformable=use_deform, bn_type=bn_type)

        dyhead_tower = []
        for i in range(cfg.MODEL.DYHEAD.NUM_CONVS):
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.EARLY_FUSE_ON:
                # cross-modality fusion
                dyhead_tower.append(
                    VLFuse(cfg)
                )
                # self language path
                if i < cfg.MODEL.DYHEAD.NUM_CONVS - 1 or cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_FUSED_FEATURES_DOT_PRODUCT:
                    if cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "bert-base-uncased":

                        dyhead_tower.append(
                            BertEncoderLayer(
                                lang_cfg,
                                clamp_min_for_underflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW,
                                clamp_max_for_overflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW)
                        )
                    else:
                        raise NotImplementedError

                else:
                    dyhead_tower.append(
                        DummyLayer()
                    )

            # self vision path
            dyhead_tower.append(
                DyConv(
                    in_channels if i == 0 else channels,
                    channels,
                    conv_func=conv_func,
                    use_dyrelu=(use_dyrelu and in_channels == channels) if i == 0 else use_dyrelu,
                    use_dyfuse=(use_dyfuse and in_channels == channels) if i == 0 else use_dyfuse,
                    use_deform=(use_deform and in_channels == channels) if i == 0 else use_deform,
                )
            )

        self.add_module('dyhead_tower', nn.Sequential(*dyhead_tower))

        self.cls_logits = nn.Conv2d(channels, num_anchors * num_classes+1, kernel_size=1)
        self.bbox_pred = nn.Conv2d(channels, num_anchors * 4, kernel_size=1)
        self.centerness = nn.Conv2d(channels, num_anchors * 1, kernel_size=1)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.DYHEAD.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        log_scale = self.cfg.MODEL.DYHEAD.LOG_SCALE

        # soft token head
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            self.token_logits = nn.Conv2d(channels, num_anchors * num_tokens, kernel_size=1)

        # dot product soft token head
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            assert self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS == False
            self.dot_product_projection_image = nn.Identity()
            self.dot_product_projection_text = nn.Linear(self.cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM,
                                                         num_anchors * channels, bias=True)
            self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=True)
            self.bias_lang = nn.Parameter(torch.zeros(self.cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM), requires_grad=True)
            self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)

        # initialization
        for modules in [self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        # if use soft token loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            for modules in [self.token_logits]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)

            torch.nn.init.constant_(self.token_logits.bias, bias_value)

        # if use dot product token loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            for modules in [self.dot_product_projection_image]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, bias_value)

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
            raise NotImplementedError

    def forward(self, x, language_dict_features=None, embedding=None, swint_feature_c4=None):
        logits = []
        bbox_reg = []
        centerness = []

        feat_inputs = {"visual": x, "lang": language_dict_features}
        dyhead_tower = self.dyhead_tower(feat_inputs)

        # soft token
        t_logits = None
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            t_logits = []

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_FUSED_FEATURES_DOT_PRODUCT:
            embedding = dyhead_tower["lang"]["hidden"]

        # dot product soft token
        dot_product_logits = None
        dot_product_proj_tokens = None
        dot_product_proj_tokens_bias = None
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            dot_product_logits = []
            # norm
            embedding = F.normalize(embedding, p=2, dim=-1)
            dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0)
            dot_product_proj_tokens_bias = torch.matmul(embedding, self.bias_lang) + self.bias0
            # torch.save(dot_product_proj_tokens, '/project/tmp/tokens_last.pt')
            # torch.save(dot_product_proj_tokens_bias, '/project/tmp/bias_last.pt')
            # raise ValueError('I want to break freeeeee')

        fused_visual_features = None
        if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
            fused_visual_features = []

        # all_projection_features = []
        # use the feature from FPN
        for l, feature in enumerate(x):
            logits.append(self.cls_logits(dyhead_tower["visual"][l]))  # TODO this is only used for getting the channel shape later

            bbox_pred = self.scales[l](self.bbox_pred(dyhead_tower["visual"][l]))
            bbox_reg.append(bbox_pred)

            centerness.append(self.centerness(dyhead_tower["visual"][l]))

            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
                t_logits.append(self.token_logits(dyhead_tower["visual"][l]))

            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
                x = dyhead_tower["visual"][l]
                if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                    fused_visual_features.append(x)
                B, C, H, W = x.shape

                # add bias (language)
                dot_product_proj_queries = self.dot_product_projection_image(x)
                dot_product_proj_queries = permute_and_flatten(dot_product_proj_queries, B, -1, C, H, W)

                A = dot_product_proj_queries.shape[1]
                bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)
                # all_projection_features.append(dot_product_proj_queries)

                dot_product_logit = (torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2))
                                     / self.log_scale.exp()) + bias
                if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT:
                    dot_product_logit = torch.clamp(dot_product_logit, max=50000)
                    dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
                dot_product_logits.append(dot_product_logit)

        # torch.save(all_projection_features, "/project/tmp/all_projection_features.pt")
        return logits, bbox_reg, centerness, t_logits, dot_product_logits, fused_visual_features


@PROPOSAL_GENERATOR_REGISTRY.register()
class VLDyHeadModule(torch.nn.Module):
    def __init__(self, cfg, input_shape):
        super(VLDyHeadModule, self).__init__()
        self.cfg = cfg
        self.head = VLDyHead(cfg)
        box_coder = BoxCoder(cfg)
        self.loss_evaluator = make_atss_loss_evaluator(cfg, box_coder)
        self.box_selector_test = make_atss_postprocessor(cfg, box_coder, is_train=False)
        self.anchor_generator = build_anchor_generator(cfg, input_shape)

        self.lang_model = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        self.joint_embedding_size = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_SIZE
        self.joint_embedding_dropout = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_DROPOUT
        if self.lang_model in ["bert-base-uncased"]:
            self.lang_dim = cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM
        else:
            self.lang_dim = 1024

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            raise NotImplementedError

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER:
            self.tunable_linear = torch.nn.Linear(self.lang_dim, 1000, bias=False)
            self.tunable_linear.weight.data.fill_(0.0)

    def forward(self, images, features, targets=None,
                language_dict_features=None,
                positive_map=None,
                captions=None,
                swint_feature_c4=None
                ):

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            raise NotImplementedError
        elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            # no resizer needed
            embedding = language_dict_features['embedded']
        else:
            embedding = None

        if "masks" in language_dict_features:
            text_masks = language_dict_features["masks"]
        else:
            text_masks = None

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER:
            embedding = self.tunable_linear.weight[:embedding.size(1), :].unsqueeze(0) + embedding
            language_dict_features['embedded'] = embedding
            language_dict_features['hidden'] = self.tunable_linear.weight[:embedding.size(1), :].unsqueeze(0) + \
                                                   language_dict_features['hidden']

        box_cls, box_regression, centerness, token_logits, dot_product_logits, fused_visual_features = self.head(
            features,
            language_dict_features,
            embedding,
            swint_feature_c4
            )

        # TODO see https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/modeling/proposal_generator/rpn.py for example
        # features = [features[f] for f in self.in_features]
        # anchors = self.anchor_generator(features)

        anchors = self.anchor_generator([features[k] for k in list(sorted(features.keys()))])
        anchors = [[Instances(im_size, boxes=anchor) for im_size in images.image_sizes] for anchor in anchors]

        if self.training:
            return self._forward_train(box_cls, box_regression, centerness, targets, anchors,
                                       captions,
                                       positive_map,
                                       token_logits,
                                       dot_product_logits,
                                       text_masks,
                                       fused_visual_features=fused_visual_features
                                       )
        else:
            return self._forward_test(box_regression, centerness, anchors,
                                      box_cls,
                                      token_logits,
                                      dot_product_logits,
                                      positive_map,
                                      fused_visual_features=fused_visual_features
                                      )

    def _forward_train(self, box_cls, box_regression, centerness, targets, anchors,
                       captions=None,
                       positive_map=None,
                       token_logits=None,
                       dot_product_logits=None,
                       text_masks=None,
                       fused_visual_features=None
                       ):
        loss_box_cls, loss_box_reg, loss_centerness, loss_token, loss_dot_product_token = self.loss_evaluator(
            box_cls, box_regression, centerness, targets, anchors,
            captions,
            positive_map,
            token_logits,
            dot_product_logits,
            text_masks
        )

        losses = {
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS:
            losses["loss_cls"] = loss_box_cls
        else:
            losses["loss_cls"] = 0.0 * loss_box_cls

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            losses["loss_token"] = loss_token * self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_LOSS_WEIGHT
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            losses["loss_dot_product_token"] = loss_dot_product_token * \
                                               self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DOT_PRODUCT_TOKEN_LOSS_WEIGHT

        if self.cfg.MODEL.RPN_ONLY:
            return None, losses, None
        else:
            raise NotImplementedError

    def _forward_test(self, box_regression, centerness, anchors,
                      box_cls=None,
                      token_logits=None,
                      dot_product_logits=None,
                      positive_map=None,
                      fused_visual_features=None
                      ):

        boxes = self.box_selector_test(box_regression, centerness, anchors,
                                       box_cls,
                                       token_logits,
                                       dot_product_logits,
                                       positive_map,
                                       )
        return boxes, {}, fused_visual_features
