from typing import Optional, List, Tuple
import math
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor
from scipy.optimize import linear_sum_assignment
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import numpy as np
from mindspore import Tensor
from mindspore import nn
from mindspore.nn import Cell
from mindspore import dtype as mstype
from mindspore.common.initializer import Normal, XavierUniform
from mindocr.utils.misc import box_cxcywh_to_xyxy,generalized_box_iou



class CtrlPointHungarianMatcher(nn.Cell):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self,
                class_weight: float = 1,
                coord_weight: float = 1, 
                focal_alpha: float = 0.25,
                focal_gamma: float = 2.0):
        """Creates the matcher
        Params:
            class_weight: This is the relative weight of the classification error in the matching cost
            coord_weight: This is the relative weight of the L1 error of the keypoint coordinates in the matching cost
        """
        super().__init__()

        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"
        self.zeros = P.ZerosLike()
        self.sigmoid = nn.Sigmoid()
        self.log = P.Log()
        self.mean = P.ReduceMean(keep_dims=True)

    def construct(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                        objects in the target) containing the class labels
                "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = self.sigmoid(outputs["pred_logits"].reshape(bs * num_queries, -1, 1))
        # [batch_size, n_queries, n_points, 2] --> [batch_size * num_queries, n_points * 2]
        out_pts = outputs["pred_ctrl_points"].reshape((bs * num_queries, -1))

        # Also concat the target labels and boxes
        tgt_pts_list = [v["ctrl_points"][:, :, :2] for v in targets]
        tgt_pts = ops.concat(tgt_pts_list, axis=0).reshape(-1, tgt_pts_list[0].shape[-2]* tgt_pts_list[0].shape[-1])
        tgt_pts_vis_list = [v["ctrl_points"][:, :, 2:] for v in targets]
        tgt_pts_vis = ops.concat(tgt_pts_vis_list, axis=0).reshape(-1, tgt_pts_vis_list[0].shape[-2]* tgt_pts_vis_list[0].shape[-1]) # it seems the contrl points are not used in cost matrix
        neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
            (-(1 - out_prob + 1e-8).log())
        pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
        cost_class = (pos_cost_class[..., 0] - neg_cost_class[..., 0]).mean(-1, keep_dims=True)

        cost_kpts = ops.cdist(out_pts, tgt_pts, p=1.0)
        C = self.class_weight * cost_class + self.coord_weight * cost_kpts
        C = C.view(bs, num_queries, -1)

        sizes = [len(v["ctrl_points"]) for v in targets]
        C_splits = C.split(sizes, -1)
        indices_list = [linear_sum_assignment(C_splits[i][i].numpy()) for i  in range(len(C_splits))]
        return [(Tensor(i, mstype.int64), Tensor(j, mstype.int64)) for i, j in indices_list]



class BoxHungarianMatcher(nn.Cell):
    """
    This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self,
                class_weight=1.0,
                coord_weight=1.0,
                giou_weight=1.0,
                focal_alpha=0.25,
                focal_gamma=2.0):
        """
        Creates the matcher
        Params:
            class_weight: This is the relative weight of the classification error in the matching cost
            coord_weight: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            giou_weight: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.giou_weight = giou_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        
        assert class_weight != 0 or coord_weight != 0 or giou_weight != 0, "all costs cant be 0"
        self.sigmoid = nn.Sigmoid()

    def construct(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                        objects in the target) containing the class labels
                "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = self.sigmoid((outputs["pred_logits"]).reshape(bs * num_queries, -1))
        out_bbox = (outputs["pred_boxes"]).reshape(bs * num_queries, -1) # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = ops.concat([v["labels"] for v in targets])
        tgt_bbox = ops.concat([v["boxes"] for v in targets])

        # Compute the classification cost.
        neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
            (-(1 - out_prob + 1e-8).log())
        pos_cost_class = self.alpha * \
            ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - \
            neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = ops.cdist(out_bbox, tgt_bbox, p=1.0)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                            box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.coord_weight * cost_bbox + self.class_weight * \
            cost_class + self.giou_weight * cost_giou
        C = C.view(bs, num_queries, -1)

        sizes = [len(v["boxes"]) for v in targets]
        C_splits = C.split(sizes, -1)
        indices = [linear_sum_assignment(C_splits[i][i].numpy()) for i in range(len(C_splits))]
        return [(Tensor(i, mstype.int64), Tensor(j, mstype.int64)) for i, j in indices]

def build_matcher(cfg):
    cfg = cfg.MODEL.TRANSFORMER.LOSS
    return BoxHungarianMatcher(class_weight=cfg.BOX_CLASS_WEIGHT,
                               coord_weight=cfg.BOX_COORD_WEIGHT,
                               giou_weight=cfg.BOX_GIOU_WEIGHT,
                               focal_alpha=cfg.FOCAL_ALPHA,
                               focal_gamma=cfg.FOCAL_GAMMA), \
        CtrlPointHungarianMatcher(class_weight=cfg.POINT_CLASS_WEIGHT,
                                 coord_weight=cfg.POINT_COORD_WEIGHT,
                                 focal_alpha=cfg.FOCAL_ALPHA,
                                 focal_gamma=cfg.FOCAL_GAMMA)