import torch
from ..utils import permute_and_flatten

from detectron2.structures import Boxes, Instances, BoxMode
from detectron2.layers import batched_nms


class ATSSPostProcessor(torch.nn.Module):
    def __init__(
            self,
            pre_nms_thresh,
            pre_nms_top_n,
            nms_thresh,
            fpn_post_nms_top_n,
            min_size,
            num_classes,
            box_coder,
            bbox_aug_enabled=False,
            bbox_aug_vote=False,
            score_agg='MEAN',
            mdetr_style_aggregate_class_num=-1
    ):
        super(ATSSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled
        self.box_coder = box_coder
        self.bbox_aug_vote = bbox_aug_vote
        self.score_agg = score_agg
        self.mdetr_style_aggregate_class_num = mdetr_style_aggregate_class_num

    def forward_for_single_feature_map(self, box_regression, centerness, anchors,
                                       box_cls=None,
                                       token_logits=None,
                                       dot_product_logits=None,
                                       positive_map=None,
                                       ):

        N, _, H, W = box_regression.shape

        A = box_regression.size(1) // 4

        if box_cls is not None:
            C = box_cls.size(1) // A

        if token_logits is not None:
            T = token_logits.size(1) // A

        # put in the same format as anchors
        if box_cls is not None:
            #print('Classification.')
            box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
            box_cls = box_cls.sigmoid()

        # binary focal loss version
        if token_logits is not None:
            #print('Token.')
            token_logits = permute_and_flatten(token_logits, N, A, T, H, W)
            token_logits = token_logits.sigmoid()
            # turn back to original classes
            scores = convert_grounding_to_od_logits(logits=token_logits, box_cls=box_cls, positive_map=positive_map,
                                                    score_agg=self.score_agg)
            box_cls = scores

        # binary dot product focal version
        if dot_product_logits is not None:
            #print('Dot Product.')
            dot_product_logits = dot_product_logits.sigmoid()
            if self.mdetr_style_aggregate_class_num != -1:
                scores = convert_grounding_to_od_logits_v2(
                    logits=dot_product_logits,
                    num_class=self.mdetr_style_aggregate_class_num,
                    positive_map=positive_map,
                    score_agg=self.score_agg,
                    disable_minus_one=False)
            else:
                scores = convert_grounding_to_od_logits(logits=dot_product_logits, box_cls=box_cls,
                                                        positive_map=positive_map,
                                                        score_agg=self.score_agg)
            box_cls = scores

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        centerness = permute_and_flatten(centerness, N, A, 1, H, W)
        centerness = centerness.reshape(N, -1).sigmoid()

        # multiply the classification scores with centerness scores

        box_cls = box_cls * centerness[:, :, None]

        results = []

        # print(anchors.tensor.shape)
        for per_box_cls, per_box_regression, per_pre_nms_top_n, per_candidate_inds, per_anchors \
                in zip(box_cls, box_regression, pre_nms_top_n, candidate_inds, anchors):
            per_box_cls = per_box_cls[per_candidate_inds]

            per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
            per_candidate_nonzeros = per_candidate_inds.nonzero()[top_k_indices, :]

            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            detections = self.box_coder.decode(
                per_box_regression[per_box_loc, :].view(-1, 4),
                per_anchors.boxes.tensor[per_box_loc, :].view(-1, 4)
            )

            boxes = Boxes(detections)  #, per_anchors.size, mode="xyxy")
            boxes.clip((per_anchors.image_size[0] - 1, per_anchors.image_size[1] - 1))
            boxes = remove_small_boxes(boxes, self.min_size)
            instances = Instances(per_anchors.image_size, pred_boxes=boxes, pred_classes=per_class, scores=torch.sqrt(per_box_cls))
            results.append(instances)

        return results

    def forward(self, box_regression, centerness, anchors,
                box_cls=None,
                token_logits=None,
                dot_product_logits=None,
                positive_map=None,
                ):
        sampled_instances = []
        # anchors = list(zip(*anchors))
        for idx, (b, c, a) in enumerate(zip(box_regression, centerness, anchors)):
            o = None
            t = None
            d = None
            if box_cls is not None:
                o = box_cls[idx]
            if token_logits is not None:
                t = token_logits[idx]
            if dot_product_logits is not None:
                d = dot_product_logits[idx]

            sampled_instances.append(
                self.forward_for_single_feature_map(b, c, a, o, t, d, positive_map)
            )

        # TODO check validity
        instancelists = list(zip(*sampled_instances))
        instances = [Instances.cat(instances) for instances in instancelists]
        if not (self.bbox_aug_enabled and not self.bbox_aug_vote):
            instances = self.select_over_all_levels(instances)

        return instances

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, instances):
        num_images = len(instances)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = instances[i][batched_nms(instances[i].pred_boxes.tensor, instances[i].scores, instances[i].pred_classes, self.nms_thresh)]
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get("scores")
                image_thresh, _ = torch.kthvalue(
                    # TODO: confirm with Pengchuan and Xiyang, torch.kthvalue is not implemented for 'Half'
                    # cls_scores.cpu(),
                    cls_scores.cpu().float(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def convert_grounding_to_od_logits(logits, box_cls, positive_map, score_agg=None):
    scores = torch.zeros(logits.shape[0], logits.shape[1], box_cls.shape[2]).to(logits.device)
    # 256 -> 80, average for each class
    if positive_map is not None:
        # score aggregation method
        if score_agg == "MEAN":
            for label_j in positive_map:
                scores[:, :, label_j] = logits[:, :, torch.LongTensor(positive_map[label_j])].mean(-1)
        elif score_agg == "MAX":
            # torch.max() returns (values, indices)
            for label_j in positive_map:
                scores[:, :, label_j] = logits[:, :, torch.LongTensor(positive_map[label_j])].max(-1)[0]
        elif score_agg == "ONEHOT":
            # one hot
            scores = logits[:, :, :len(positive_map)]
        else:
            raise NotImplementedError
    return scores


def convert_grounding_to_od_logits_v2(logits, num_class, positive_map, score_agg=None, disable_minus_one=True):
    scores = torch.zeros(logits.shape[0], logits.shape[1], num_class).to(logits.device)
    # 256 -> 80, average for each class
    if positive_map is not None:
        # score aggregation method
        if score_agg == "MEAN":
            for label_j in positive_map:
                locations_label_j = positive_map[label_j]
                if isinstance(locations_label_j, int):
                    locations_label_j = [locations_label_j]
                scores[:, :, label_j] = logits[:, :,torch.LongTensor(locations_label_j)].mean(-1)
        elif score_agg == "POWER":
            for label_j in positive_map:
                locations_label_j = positive_map[label_j]
                if isinstance(locations_label_j, int):
                    locations_label_j = [locations_label_j]

                probability = torch.prod(logits[:, :, torch.LongTensor(locations_label_j)], dim=-1).squeeze(-1)
                probability = torch.pow(probability, 1 / len(locations_label_j))
                scores[:, :, label_j] = probability
        elif score_agg == "MAX":
            # torch.max() returns (values, indices)
            for label_j in positive_map:
                scores[:, :, label_j] = \
                logits[:, :, torch.LongTensor(positive_map[label_j])].max(-1)[
                    0]
        elif score_agg == "ONEHOT":
            # one hot
            scores = logits[:, :, :len(positive_map)]
        else:
            raise NotImplementedError
    return scores


def make_atss_postprocessor(config, box_coder, is_train=False):
    pre_nms_thresh = config.MODEL.ATSS.INFERENCE_TH
    if is_train:
        pre_nms_thresh = config.MODEL.ATSS.INFERENCE_TH_TRAIN
    pre_nms_top_n = config.MODEL.ATSS.PRE_NMS_TOP_N
    fpn_post_nms_top_n = config.MODEL.ATSS.DETECTIONS_PER_IMG
    if is_train:
        pre_nms_top_n = config.MODEL.ATSS.PRE_NMS_TOP_N_TRAIN
        fpn_post_nms_top_n = config.MODEL.ATSS.POST_NMS_TOP_N_TRAIN
    nms_thresh = config.MODEL.ATSS.NMS_TH
    score_agg = config.MODEL.DYHEAD.SCORE_AGG

    box_selector = ATSSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.ATSS.NUM_CLASSES,
        box_coder=box_coder,
        bbox_aug_enabled=config.TEST.USE_MULTISCALE,
        score_agg=score_agg,
        mdetr_style_aggregate_class_num=config.TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM
    )

    return box_selector


def remove_small_boxes(boxes, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxes (Boxes)
        min_size (int)
    """
    # WORK AROUND: work around unbind using split + squeeze.
    # xywh_boxes = boxes.convert("xywh").bbox
    xywh_boxes = BoxMode.convert(boxes.tensor, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    _, _, ws, hs = xywh_boxes.split(1, dim=1)
    ws = ws.squeeze(1)
    hs = hs.squeeze(1)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
    return boxes[keep]
