import torch
import torch.nn.functional as F

from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit
from detectron2.structures import Boxes, pairwise_iou, BoxMode
from detectron2.utils.comm import get_world_size, dist
from transformers import AutoTokenizer

from ..utils import cat, concat_box_prediction_layers
from glip_detectron2.utils.amp import custom_fwd
from glip_detectron2.utils.comm import reduce_sum

INF = 1e8


class TokenSigmoidFocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(TokenSigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, text_masks=None, version="binary", **kwargs):
        if version == "binary":
            loss_func = token_sigmoid_binary_focal_loss
        elif version == "softmax":
            raise NotImplementedError
        elif version == "binaryv2":
            raise NotImplementedError
        else:
            raise NotImplementedError
        loss = loss_func(logits, targets, self.alpha, self.gamma, text_masks, **kwargs)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr


def token_sigmoid_binary_focal_loss(pred_logits, targets, alpha, gamma, text_mask=None):
    # binary version of focal loss
    # copied from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor with the reduction option applied.
    """
    assert (targets.dim() == 3)
    assert (pred_logits.dim() == 3)  # batch x from x to

    bs, n, _ = pred_logits.shape
    if text_mask is not None:
        assert (text_mask.dim() == 2)
        text_mask = (text_mask > 0).unsqueeze(1)
        # text_mask = text_mask.repeat(1, pred_logits.size(1), 1)  # copy along the image channel dimension
        pred_logits = torch.masked_select(pred_logits, text_mask)
        targets = torch.masked_select(targets, text_mask)

    p = torch.sigmoid(pred_logits)
    ce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss


class ATSSLossComputation(torch.nn.Module):

    def __init__(self, cfg, box_coder):
        super(ATSSLossComputation, self).__init__()

        self.cfg = cfg
        self.fl_gamma = cfg.MODEL.FOCAL.LOSS_GAMMA
        self.fl_alpha = cfg.MODEL.FOCAL.LOSS_ALPHA
        self.cls_loss_func = sigmoid_focal_loss_jit
        self.centerness_loss_func = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.box_coder = box_coder

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS or self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            self.token_loss_func = TokenSigmoidFocalLoss(cfg.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_ALPHA,
                                                         cfg.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_GAMMA)

        self.lang = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE

        # self.tokenizer = AutoTokenizer.from_pretrained(self.lang)
        if self.cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            raise NotImplementedError
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.lang)

        # if use shallow contrastive loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS \
                or self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS:
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS:
                assert self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS == False
                channels = cfg.MODEL.DYHEAD.CHANNELS
                num_anchors = len(cfg.MODEL.RPN.ASPECT_RATIOS) * cfg.MODEL.RPN.SCALES_PER_OCTAVE
                shallow_input_dim = channels * num_anchors
            elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS:
                assert self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS == False
                shallow_input_dim = cfg.MODEL.SWINT.OUT_CHANNELS[-2]

            shallow_log_scale = self.cfg.MODEL.DYHEAD.SHALLOW_LOG_SCALE
            shallow_contrastive_hdim = cfg.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_CONTRASTIVE_HIDDEN_DIM
            # self.shallow_contrastive_projection_image = nn.Conv2d(channels, num_anchors * shallow_contrastive_hdim,
            #                                                       kernel_size=1)
            self.shallow_contrastive_projection_image = nn.Linear(shallow_input_dim, shallow_contrastive_hdim,
                                                                  bias=True)
            self.shallow_contrastive_projection_text = nn.Linear(self.cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM,
                                                                 shallow_contrastive_hdim, bias=True)
            self.shallow_log_scale = nn.Parameter(torch.Tensor([shallow_log_scale]), requires_grad=True)

        # (initialization) if use shallow contrastive loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS:
            for modules in [self.shallow_contrastive_projection_image, self.shallow_contrastive_projection_text]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)
                    if isinstance(l, nn.Linear):
                        torch.nn.init.xavier_uniform_(l.weight)
                        l.bias.data.fill_(0)

    def NllSoftMaxLoss(self, logits, target):
        loss_ce = -target * logits.log_softmax(
            -1)  # basically, only the those positives with positive target_sim will have losses
        return loss_ce

    def ContrastiveAlignLoss(self, logits, positive_map):
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits  # .masked_fill(positive_map, -1000000)

        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        neg_term = negative_logits.logsumexp(2)

        nb_pos = positive_map.sum(2) + 1e-6

        box_to_token_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~boxes_with_pos, 0).sum()

        tokens_with_pos = positive_map.any(1)
        pos_term = positive_logits.sum(1)
        neg_term = negative_logits.logsumexp(1)

        nb_pos = positive_map.sum(1) + 1e-6

        tokens_to_boxes_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~tokens_with_pos, 0).sum()
        tot_loss = (box_to_token_loss + tokens_to_boxes_loss) / 2

        return tot_loss

    def GIoULoss(self, pred, target, anchor, weight=None):
        pred_boxes = self.box_coder.decode(pred.view(-1, 4), anchor.view(-1, 4))
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_y2 = torch.max(pred_y1, pred_y2)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        gt_boxes = self.box_coder.decode(target.view(-1, 4), anchor.view(-1, 4))
        target_x1 = gt_boxes[:, 0]
        target_y1 = gt_boxes[:, 1]
        target_x2 = gt_boxes[:, 2]
        target_y2 = gt_boxes[:, 3]
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        x1_intersect = torch.max(pred_x1, target_x1)
        y1_intersect = torch.max(pred_y1, target_y1)
        x2_intersect = torch.min(pred_x2, target_x2)
        y2_intersect = torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size()).to(pred)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

        x1_enclosing = torch.min(pred_x1, target_x1)
        y1_enclosing = torch.min(pred_y1, target_y1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        y2_enclosing = torch.max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing

        losses = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()

    def prepare_targets(self, targets, anchors, bg_class, positive_map=None):
        cls_labels = []
        reg_targets = []
        token_labels = []

        offset = 0
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            # TODO assert BoxMode.XYXY_ABS?
            # bboxes_per_im = targets_per_im.get_field("boxes")
            bboxes_per_im = targets_per_im.gt_boxes
            labels_per_im = targets_per_im.gt_classes
            num_gt = len(bboxes_per_im)

            if positive_map is not None:
                token_per_im = positive_map[offset:offset + num_gt, :]
                offset += num_gt

            anchors_per_im = Boxes.cat([a[im_i].boxes for a in anchors])

            num_anchors_per_loc = len(self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS) * self.cfg.MODEL.ANCHOR_GENERATOR.SCALES_PER_OCTAVE
            num_anchors_per_level = [len(a[im_i]) for a in anchors]
            # TODO fix the use of Instances and Boxes, especially for more than 1 box per image
            # ious = pairwise_iou(anchors_per_im, Boxes(torch.cat(targets_per_im.boxes, dim=0) if isinstance(targets_per_im.boxes, list) else targets_per_im.boxes))
            ious = pairwise_iou(anchors_per_im, bboxes_per_im)

            gt_cx = (bboxes_per_im.tensor[:, 2] + bboxes_per_im.tensor[:, 0]) / 2.0
            gt_cy = (bboxes_per_im.tensor[:, 3] + bboxes_per_im.tensor[:, 1]) / 2.0
            gt_points = torch.stack((gt_cx, gt_cy), dim=1)

            anchors_cx_per_im = (anchors_per_im.tensor[:, 2] + anchors_per_im.tensor[:, 0]) / 2.0
            anchors_cy_per_im = (anchors_per_im.tensor[:, 3] + anchors_per_im.tensor[:, 1]) / 2.0
            anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

            distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

            # Selecting candidates based on the center distance between anchor box and object
            candidate_idxs = []
            star_idx = 0
            for level, anchors_per_level in enumerate([a[im_i] for a in anchors]):
                end_idx = star_idx + num_anchors_per_level[level]
                distances_per_level = distances[star_idx:end_idx, :]
                topk = min(self.cfg.MODEL.ATSS.TOPK * num_anchors_per_loc, num_anchors_per_level[level])
                _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
                candidate_idxs.append(topk_idxs_per_level + star_idx)
                star_idx = end_idx
            candidate_idxs = torch.cat(candidate_idxs, dim=0)

            # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
            candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
            iou_mean_per_gt = candidate_ious.mean(0)
            iou_std_per_gt = candidate_ious.std(0)
            iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
            is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

            # Limiting the final positive samples’ center to object
            anchor_num = anchors_cx_per_im.shape[0]
            for ng in range(num_gt):
                candidate_idxs[:, ng] += ng * anchor_num
            if num_gt:
                e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                candidate_idxs = candidate_idxs.view(-1)
                l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bboxes_per_im.tensor[:, 0]
                t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bboxes_per_im.tensor[:, 1]
                r = bboxes_per_im.tensor[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
                b = bboxes_per_im.tensor[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
                is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
                is_pos = is_pos & is_in_gts

            if num_gt:
                # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
                ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
                index = candidate_idxs.view(-1)[is_pos.view(-1)]
                ious_inf[index] = ious.t().contiguous().view(-1)[index]
                ious_inf = ious_inf.view(num_gt, -1).t()

                anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
                # get positive anchors index from ATSS
                cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
                cls_labels_per_im[anchors_to_gt_values == -INF] = bg_class

                if positive_map is not None:
                    token_labels_per_im = token_per_im[anchors_to_gt_indexs]
                    unmatched_labels = torch.zeros(token_labels_per_im.shape[1], device=token_labels_per_im.device)
                    # TODO: temporarially disable the [NoObj] token logic, and only restrict to binary loss
                    unmatched_labels[-1] = 1  # token: none object - > 256
                    token_labels_per_im[anchors_to_gt_values == -INF] = unmatched_labels
                    # move from cpu to gpu
                    token_labels_per_im = token_labels_per_im.to(cls_labels_per_im.device)

                matched_gts = bboxes_per_im.tensor[anchors_to_gt_indexs]

                reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.tensor)
            else:
                cls_labels_per_im = labels_per_im.new_full((anchors_per_im.tensor.shape[0],), bg_class)
                reg_targets_per_im = anchors_per_im.tensor.new_zeros((anchors_per_im.tensor.shape[0], 4))
                if positive_map is not None:
                    token_labels_per_im = token_per_im.new_zeros((anchors_per_im.tensor.shape[0], token_per_im.shape[1]))
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)

            if positive_map is not None:
                token_labels.append(token_labels_per_im)

        return cls_labels, reg_targets, token_labels

    def compute_centerness_targets(self, reg_targets, anchors):
        gts = self.box_coder.decode(reg_targets, anchors)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l = anchors_cx - gts[:, 0]
        t = anchors_cy - gts[:, 1]
        r = gts[:, 2] - anchors_cx
        b = gts[:, 3] - anchors_cy
        left_right = torch.stack([l, r], dim=1)
        top_bottom = torch.stack([t, b], dim=1)
        centerness = torch.sqrt((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                                (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    # @custom_fwd(cast_inputs=torch.float32)  # TODO breaks on targets which is of type Instances, but might be needed
    # @torch.amp.autocast('cuda', enabled=False)
    def __call__(self, box_cls, box_regression, centerness, targets, anchors,
                 captions=None,
                 positive_map=None,
                 token_logits=None,
                 dot_product_logits=None,
                 text_masks=None,
                 ):
        labels, reg_targets, token_labels = self.prepare_targets(targets, anchors, bg_class=box_cls[0].shape[1] - 1,
                                                                 positive_map=positive_map)
        N = len(labels)

        box_regression_flatten, box_cls_flatten, token_logits_stacked = concat_box_prediction_layers(
            box_regression,
            box_cls,
            token_logits,
        )
        # dot product soft token logits
        if dot_product_logits is not None:
            dot_product_logits = torch.cat(dot_product_logits, dim=1)

        centerness_flatten = [ct.permute(0, 2, 3, 1).reshape(N, -1, 1) for ct in centerness]
        centerness_flatten = torch.cat(centerness_flatten, dim=1).reshape(-1)

        labels_flatten = torch.cat(labels, dim=0)
        reg_targets_flatten = torch.cat(reg_targets, dim=0)
        anchors_flatten = torch.cat([Boxes.cat([a[i].boxes for a in anchors]).tensor for i in range(len(targets))], dim=0)

        if positive_map is not None:
            token_labels_stacked = torch.stack(token_labels, dim=0)

        pos_inds = torch.nonzero(labels_flatten < (box_cls[0].shape[1] - 1)).squeeze(1)

        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        # TODO check which reduction is needed
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS:
            cls_loss = self.cls_loss_func(box_cls_flatten, F.one_hot(labels_flatten, box_cls_flatten.shape[1]), self.fl_alpha, self.fl_gamma, reduction='sum') / num_pos_avg_per_gpu
        else:
            cls_loss = 0.0

        token_logits_loss = None
        dot_product_token_loss = None

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            token_logits_loss = self.token_loss_func(token_logits_stacked,
                                                     token_labels_stacked, text_masks=text_masks,
                                                     version="binary") / num_pos_avg_per_gpu

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            dot_product_token_loss = self.token_loss_func(dot_product_logits,
                                                          token_labels_stacked, text_masks=text_masks,
                                                          version="binary") / num_pos_avg_per_gpu

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        anchors_flatten = anchors_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten, anchors_flatten)

            sum_centerness_targets_avg_per_gpu = reduce_sum(centerness_targets.sum()).item() / float(num_gpus)
            reg_loss = self.GIoULoss(box_regression_flatten, reg_targets_flatten, anchors_flatten,
                                     weight=centerness_targets) / sum_centerness_targets_avg_per_gpu
            centerness_loss = self.centerness_loss_func(centerness_flatten, centerness_targets) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss * self.cfg.MODEL.ATSS.REG_LOSS_WEIGHT, centerness_loss, \
            token_logits_loss, \
            dot_product_token_loss


def make_atss_loss_evaluator(cfg, box_coder):
    loss_evaluator = ATSSLossComputation(cfg, box_coder)
    return loss_evaluator
