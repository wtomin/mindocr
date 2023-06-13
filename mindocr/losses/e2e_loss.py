from typing import Optional, List, Tuple
import math
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor
#from mindspore.communication._comm_helper import GlobalComm
#from mindspore.communication import get_group_size
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import numpy as np
from mindspore import Tensor
from mindspore import nn
from mindspore.nn import Cell
from mindspore import dtype as mstype
from mindspore.nn.loss.loss import LossBase

from mindocr.utils.misc import box_cxcywh_to_xyxy,box_xyxy_to_cxcywh,generalized_box_iou, accuracy
import mindspore.numpy as mnp
from .matcher import CtrlPointHungarianMatcher, BoxHungarianMatcher

class SigmoidFocalLoss(nn.Cell):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        num_inst: Number of instances
    Returns:
        Loss tensor
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def construct(self, inputs, targets, num_inst=1):
        prob = ops.sigmoid(inputs)
        ce_loss = self.ce_loss(inputs, targets)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if loss.ndim == 4:
            # loss shape (N, num_queries, num_points, num_classes)
            return loss.mean(1).mean(1).sum() / num_inst
        elif loss.ndim == 3:
            # loss shape (N, num_queries, num_classes)
            return loss.mean(1).sum() / num_inst
        else:
            raise NotImplementedError(f"Unsupported dim {loss.ndim}")


class TESTRLoss(LossBase):
    """
    Compute the loss for TESTR.
    The process happens in two steps:
    1) we compute hungarian assignment between ground truth boxes and the outputs of the model
    2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, 
                 num_classes, 
                 num_ctrl_points, 
                 box_class_weight,
                 point_class_weight,
                 box_coord_weight,
                 point_coord_weight,
                 point_text_weight,
                 box_giou_weight,  
                 transformer_dec_layers,
                 aux_loss = True,
                 enc_losses= ['labels', 'boxes'], 
                 dec_losses = ['labels', 'ctrl_points', 'texts'], 
                 focal_alpha=0.25, 
                 focal_gamma=2.0,
                 use_polygon=True):
        """
        Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        weight_dict = TESTRLoss.get_weight_dict(point_class_weight, point_coord_weight, point_text_weight,
                                                box_coord_weight, box_giou_weight, box_class_weight,
                                                transformer_dec_layers, aux_loss)
        self.num_classes = num_classes
        self.enc_matcher = TESTRLoss.get_encoder_matcher(box_class_weight, box_coord_weight, box_giou_weight, focal_alpha, focal_gamma)
        self.dec_matcher = TESTRLoss.get_decoder_matcher(point_class_weight, point_coord_weight, focal_alpha, focal_gamma)
        self.weight_dict = weight_dict
        self.enc_losses = enc_losses 
        self.dec_losses = dec_losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_ctrl_points = num_ctrl_points
        self.use_polygon = use_polygon
        self.cross_entropy_with_logits = ops.SoftmaxCrossEntropyWithLogits()
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.sigmoid_focal_loss = SigmoidFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    @classmethod
    def get_weight_dict(cls, point_class_weight, point_coord_weight, point_text_weight,
                       box_coord_weight, box_giou_weight, box_class_weight,
                       transformer_dec_layers, aux_loss):
        weight_dict = {'loss_ce': point_class_weight, 'loss_ctrl_points': point_coord_weight, 'loss_texts': point_text_weight}
        enc_weight_dict = {'loss_bbox': box_coord_weight, 'loss_giou': box_giou_weight, 'loss_ce': box_class_weight}
        if aux_loss:
            aux_weight_dict = {}
            # decoder aux loss
            for i in range(transformer_dec_layers - 1):
                aux_weight_dict.update(
                    {k + f'_{i}': v for k, v in weight_dict.items()})
            # encoder aux loss
            aux_weight_dict.update(
                {k + f'_enc': v for k, v in enc_weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        return weight_dict
    
    @classmethod
    def get_encoder_matcher(cls, box_class_weight, box_coord_weight, box_giou_weight, focal_alpha, focal_gamma):
        return BoxHungarianMatcher(class_weight=box_class_weight,
                               coord_weight=box_coord_weight,
                               giou_weight=box_giou_weight,
                               focal_alpha=focal_alpha,
                               focal_gamma=focal_gamma)
    @classmethod
    def get_decoder_matcher(cls, point_class_weight, point_coord_weight, focal_alpha, focal_gamma):
        return CtrlPointHungarianMatcher(class_weight=point_class_weight,
                                 coord_weight=point_coord_weight,
                               focal_alpha=focal_alpha,
                               focal_gamma=focal_gamma)
    def loss_labels(self, outputs, targets, indices, num_inst, log=False):
        """
        Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits'] #(bs, num_queries, n_points, num_classes-1)

        idx = self.get_src_permutation_idx(indices)
        target_classes = mnp.full(src_logits.shape[:-1], self.num_classes, dtype=mstype.int32)
        target_classes_o = ops.concat([t["labels"][J]  for t, (_, J) in zip(targets, indices)])
        if len(target_classes_o.shape) < len(target_classes[idx].shape):
            target_classes_o = target_classes_o[..., None]
        target_classes[idx] = target_classes_o

        shape = list(src_logits.shape)
        shape[-1] += 1 # (Bs, N_proposals, n_points, N_classes + 1)
        target_classes_onehot = ops.zeros(shape, dtype=mstype.int32)
        rep_shapes =  [1] * (len(shape)-1) + [self.num_classes + 1]
        target_classes_onehot = ops.scatter(target_classes_onehot, -1, 
                                            mnp.tile(target_classes.unsqueeze(-1), rep_shapes), 
                                            ops.ones(shape, mstype.int32))
        target_classes_onehot = target_classes_onehot[..., :-1]
        # whether to use mask to remove unvisible points?
        loss_ce = self.sigmoid_focal_loss(src_logits, target_classes_onehot.float(), num_inst) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - \
                accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    
    # def loss_cardinality(self, outputs, targets, indices, num_inst):
    #     """
    #     Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
    #     This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
    #     """
    #     pred_logits = outputs['pred_logits']
    #     device = pred_logits.device
    #     tgt_lengths = ops.tensor.Tensor(np.as_tensor(
    #         [len(v["labels"]) for v in targets], device=device))
    #     card_pred = (pred_logits.mean(-2).argmax(-1) == 0).sum(1)
    #     card_err = self.l1_loss((Tensor(ops.asnumpy(card_pred), dtype=ops.float32)), (Tensor(ops.asnumpy(tgt_lengths), dtype=ops.float32)))
    #     losses = {'cardinality_error': card_err}
    #     return losses

    def loss_boxes(self, outputs, targets, indices, num_inst):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        
        idx = self.get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = ops.concat([t['boxes'][i]  for t, (_, i) in zip(targets, indices)], axis=0)

        loss_bbox = self.l1_loss(src_boxes, target_boxes) # reduction=sum

        losses = {}
        losses['loss_bbox'] = loss_bbox / num_inst
        giou = generalized_box_iou( box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        loss_giou = 1 - ops.diagonal(giou)
        losses['loss_giou'] = loss_giou.sum() / num_inst
        return losses

    def loss_texts(self, outputs, targets, indices, num_inst):
        assert 'pred_texts' in outputs
        idx = self.get_src_permutation_idx(indices)
        src_texts = outputs['pred_texts'][idx] 
        target_texts = mnp.concatenate([t['texts'][i] for t, (_, i) in zip(targets, indices)], axis=0) #(num_instances, max_text_len)
        shape = src_texts.shape #(num_instances, max_text_len, dict_size)
        num_classes = shape[-1]
        target_texts_onehot = ops.zeros(shape, dtype=src_texts.dtype)
        rep_shapes =  [1] * (len(shape)-1) + [num_classes]
        target_texts_onehot = ops.scatter(target_texts_onehot, -1, 
                                          mnp.tile(target_texts.unsqueeze(-1), rep_shapes), 
                                          ops.ones(shape, src_texts.dtype))

        loss_texts, _ = self.cross_entropy_with_logits(src_texts.reshape(-1, num_classes), target_texts_onehot.reshape(-1, num_classes))
        return {'loss_texts': loss_texts.mean()}

    def loss_ctrl_points(self, outputs, targets, indices, num_inst):
        """Compute the losses related to the keypoint coordinates, the L1 regression loss
           Use the visible mask to remove unvisible points
        """
        assert 'pred_ctrl_points' in outputs
        idx = self.get_src_permutation_idx(indices)
        src_ctrl_points = outputs['pred_ctrl_points'][idx]
        target_ctrl_points = mnp.concatenate([t['ctrl_points'][i][:,:,:2] for t, (_, i) in zip(targets, indices)], axis=0)
        target_ctrl_points_vis = mnp.concatenate([t['ctrl_points'][i][:,:,2:] for t, (_, i) in zip(targets, indices)], axis=0)
        mask = ops.concat([target_ctrl_points_vis, target_ctrl_points_vis], axis=-1) == 1
        loss_ctrl_points = self.l1_loss(src_ctrl_points[mask], target_ctrl_points[mask]) # reduction = sum?

        losses = {'loss_ctrl_points': loss_ctrl_points / num_inst}
        return losses

    @staticmethod
    def get_src_permutation_idx(indices):
        # permute predictions following indices
        len_indices = len(indices)
        batch_idx = mnp.concatenate([mnp.full_like(indices[i][0], i) for i in range(len_indices)])
        src_idx = mnp.concatenate([src for (src, _) in indices])
        return batch_idx, src_idx


    # def prepare_targets(self, targets):
    #     """
    #     prepare the 
    #     """
    #     new_targets = {'enc': [], 'dec': []}
    #     image_sizes = targets['image_size'][:, ::-1] # h,w order
    #     raw_ctrl_points = targets['polys'] if self.use_polygon else targets['beziers']
    #     gt_classes = targets['gt_classes'] # 0 indicates the text instances, 1 indicates padded instances
    #     ignore_tags = targets['ignore_tags']
    #     gt_text_ids = targets['rec_ids']
    #     import pdb; pdb.set_trace()
    #     for i in range(len(image_sizes)):
    #         # prepare encoder labels and gt_boxes
    #         non_pad_mask = gt_classes[i] == 0 
    #         gt_class = gt_classes[i][non_pad_mask] # mask cause dynamic shape
    #         gt_boxes_per_image = targets['boxes'][i][non_pad_mask] / image_sizes[i] # (N, 4, 2)
    #         gt_boxes_per_image = box_xyxy_to_cxcywh(gt_boxes_per_image) # (x_center, y_center, width, height)
    #         new_targets['enc'].append({'labels': gt_class, 'boxes': gt_boxes_per_image})
    #         # prepare decoder labels, gt_ctrl_points and text_ids
    #         non_ignore_tags = ~ignore_tags[i]
    #         gt_ctrl_points = raw_ctrl_points[i][non_ignore_tags& non_pad_mask] # (N, num_ctrl_points, 3), the last dimension indicates the visibility
    #         gt_ctrl_points[:, :, :2] = gt_ctrl_points[:,:, :2] / image_sizes[i] # normalize the coordinates
    #         gt_class = gt_classes[i][non_ignore_tags& non_pad_mask]
    #         gt_text_id = gt_text_ids[i][non_ignore_tags& non_pad_mask]
    #         new_targets['dec'].append({'ctrl_points': gt_ctrl_points, 'labels': gt_class, 'texts': gt_text_id})

    #     return new_targets
    def prepare_targets(self, targets):
        """
        prepare the 
        """
        new_targets = {'enc': [], 'dec': []}
        image_sizes = targets['image_size'][:, ::-1] # h,w order
        raw_ctrl_points = targets['polys'] if self.use_polygon else targets['beziers']
        gt_classes = targets['gt_classes'] # 0 indicates the text instances, 1 indicates padded instances
        ignore_tags = targets['ignore_tags']
        gt_text_ids = targets['rec_ids']
        import pdb; pdb.set_trace()
        for i in range(len(image_sizes)):
            # prepare encoder labels and gt_boxes
            # non_pad_mask = gt_classes[i] == 0 
            gt_class = gt_classes[i] # mask cause dynamic shape
            gt_boxes_per_image = targets['boxes'][i] / image_sizes[i] # (N, 4, 2)
            gt_boxes_per_image = box_xyxy_to_cxcywh(gt_boxes_per_image) # (x_center, y_center, width, height) zeros won't change
            new_targets['enc'].append({'labels': gt_class, 'boxes': gt_boxes_per_image})
            # prepare decoder labels, gt_ctrl_points and text_ids
            non_ignore_tags = ~ignore_tags[i]
            gt_ctrl_points = raw_ctrl_points[i] # (N, num_ctrl_points, 3), the last dimension indicates the visibility
            gt_ctrl_points[:, :, :2] = gt_ctrl_points[:,:, :2] / image_sizes[i] # normalize the coordinates
            gt_text_id = gt_text_ids[i]
            new_targets['dec'].append({'ctrl_points': gt_ctrl_points, 'labels': gt_class, 'texts': gt_text_id, 'ignore_tags': ignore_tags[i]})

        return new_targets

    def get_loss(self, loss_name, outputs, targets, indices, num_inst, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            #'cardinality': self.loss_cardinality,
            'ctrl_points': self.loss_ctrl_points,
            'boxes': self.loss_boxes,
            'texts': self.loss_texts,
        }
        if loss_name not in loss_map:
            raise ValueError(f"Do you really want to compute {loss_name} loss?")
        return loss_map[loss_name](outputs, targets, indices, num_inst, **kwargs)

    def construct(self, outputs, image_size, polys, boxes, rec_ids, ignore_tags, gt_classes):
        #outputs is a dictionary containing the outputs of the model.
        # change the targets
        targets = {'image_size': image_size, 'polys': polys, 'boxes': boxes, 'rec_ids': rec_ids, 'ignore_tags': ignore_tags, 'gt_classes': gt_classes}
        targets = self.prepare_targets(targets)
        # Remove auxiliary outputs from the outputs
        outputs_without_aux = dict([(k, v) for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'])
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = ops.stop_gradient(self.dec_matcher(outputs_without_aux, targets['dec']))

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_inst = sum(len(t['ctrl_points']) for t in targets['dec'])
        num_inst = Tensor([num_inst], dtype=mstype.float32)
        # if GlobalComm.INITED: # distributed training
        #     P.AllReduce()(num_inst)
        # num_inst = num_inst / (1 if not GlobalComm.INITED else get_group_size())
        # num_inst = ops.clip_by_value(num_inst, 1, num_inst).asnumpy()[0]

        # Compute all the requested losses
        losses = {}
        for loss in self.dec_losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets['dec'], indices, num_inst, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                aux_outputs = outputs['aux_outputs'][i]
                indices = ops.stop_gradient(self.dec_matcher(aux_outputs, targets['dec']))
                for loss in self.dec_losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets['dec'], indices, num_inst, **kwargs)
                    l_dict = dict([(k + f'_{i}', v) for k, v in l_dict.items()])
                    losses.update(l_dict)

        # In case of encoder losses, we compute the loss for labels and boxes
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices = self.enc_matcher(enc_outputs, targets['enc'])
            for loss in self.enc_losses:
                kwargs = {}
                if loss == 'labels':
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, targets['enc'], indices, num_inst, **kwargs)
                l_dict = dict([(k + f'_enc', v) for k, v in l_dict.items()])
                losses.update(l_dict)
        
        weight_dict = self.weight_dict
        for k in losses.keys():
            if k in weight_dict:
                losses[k] *= weight_dict[k]
        return sum(losses.values()).item() # return the weighted sum of all losses
        