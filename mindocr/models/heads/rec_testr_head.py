from mindspore import nn
from mindspore.common import dtype as mstype
from mindocr.models.layers import MultiScaleDeformableAttention, DeformableTransformerEncoderLayer, DeformableTransformerEncoder,\
                                  DeformableCompositeTransformerDecoderLayer, DeformableCompositeTransformerDecoder,\
                                  PositionalEncoding1D, PositionalEncoding2D
                                  
from mindspore.nn.transformer.op_parallel_config import default_dpmp_config
import copy
from typing import Optional, List, Tuple
import math
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor

from mindspore.ops import operations as P
from mindspore.ops import functional as F
import numpy as np
from mindspore import Tensor
from mindspore import nn
from mindspore.nn import Cell
from mindspore import dtype as mstype
from mindspore.common.initializer import initializer, Normal, XavierUniform
from mindocr.utils.misc import NestedTensor,  MLP, inverse_sigmoid_offset, sigmoid_offset, _get_clones

class TESTRHead(nn.Cell):
    def __init__(self, hidden_size:int,
                 in_channels: int,
                 num_classes: int,
                 num_pred: int, # num of control points
                 use_polygon: bool,
                 aux_loss: bool,
                 voc_size: int,
                 param_init_type: mstype.number = mstype.float32,
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.sigmoid_offset = not use_polygon
        self.aux_loss = aux_loss
        self.voc_size = voc_size
        self.ctrl_point_class = nn.Dense(self.hidden_size, self.num_classes)
        self.ctrl_point_coord = MLP(self.hidden_size, self.hidden_size, output_dim = 2, num_layers = 3)
        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        
        self.ctrl_point_class.bias.set_data(initializer('ones', [num_classes]) * bias_value, param_init_type)
        self.ctrl_point_coord.layers[-1].weight.set_data(initializer('zeros', self.ctrl_point_coord.layers[-1].weight.shape))
        self.ctrl_point_coord.layers[-1].bias.set_data(initializer('zeros', self.ctrl_point_coord.layers[-1].bias.shape))

        self.ctrl_point_class = _get_clones(self.ctrl_point_class, num_pred)
        self.ctrl_point_coord = _get_clones(self.ctrl_point_coord, num_pred)
        self.text_class = nn.Dense(self.hidden_size, self.voc_size + 1)

    def construct(self, hs, hs_text, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact):
        # output
        outputs_classes = []
        outputs_coords = []
        outputs_texts = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)
            outputs_class = self.ctrl_point_class[lvl](hs[lvl])
            tmp = self.ctrl_point_coord[lvl](hs[lvl])
            if reference.shape[-1] == 2:
                tmp += reference[:, :, None, :]
            else:
                assert reference.shape[-1] == 4
                tmp += reference[:, :, None, :2]
            outputs_texts.append(self.text_class(hs_text[lvl]))
            outputs_coord = sigmoid_offset(tmp, offset=self.sigmoid_offset)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = ops.stack(outputs_classes)
        outputs_coord = ops.stack(outputs_coords)
        outputs_text = ops.stack(outputs_texts)

        out = {'pred_logits': outputs_class[-1],
               'pred_ctrl_points': outputs_coord[-1],
               'pred_texts': outputs_text[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_text)

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {
            'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out
    
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_text):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_ctrl_points': b, 'pred_texts': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_text[:-1])]
    
    
    # def inference(self, ctrl_point_cls, ctrl_point_coord, text_pred, image_sizes):
    #     """
        
    #     """
    #     assert len(ctrl_point_cls) == len(image_sizes)
    #     results = []

    #     text_pred = ops.softmax(text_pred, -1)
    #     prob = ctrl_point_cls.mean(-2).sigmoid()
    #     scores, labels = prob.max(-1)

    #     for scores_per_image, labels_per_image, ctrl_point_per_image, text_per_image, image_size in zip(
    #         scores, labels, ctrl_point_coord, text_pred, image_sizes
    #     ):
    #         selector = scores_per_image >= self.test_score_threshold
    #         scores_per_image = scores_per_image[selector]
    #         labels_per_image = labels_per_image[selector]
    #         ctrl_point_per_image = ctrl_point_per_image[selector]
    #         text_per_image = text_per_image[selector]
    #         result = Instances(image_size) # the Instance is imported from detectron2, needs to be replaced.                 The :class:`Instances` object has the following keys:
    #             "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
    #         result.scores = scores_per_image
    #         result.pred_classes = labels_per_image
    #         result.rec_scores = text_per_image
    #         ctrl_point_per_image[..., 0] *= image_size[1]
    #         ctrl_point_per_image[..., 1] *= image_size[0]
    #         if self.use_polygon:
    #             result.polygons = ctrl_point_per_image.flatten(1)
    #         else:
    #             result.beziers = ctrl_point_per_image.flatten(1)
    #         _, topi = text_per_image.topk(1)
    #         result.recs = topi.squeeze(-1)
    #         results.append(result)
    #     return results
    # def get_processed_results(self, output_from_heads, image_sizes):
    #     ctrl_point_cls = output_from_heads["pred_logits"]
    #     ctrl_point_coord = output_from_heads["pred_ctrl_points"]
    #     text_pred = output_from_heads["pred_texts"]
    #     results = self.inference(ctrl_point_cls, ctrl_point_coord, text_pred, image_sizes)
    #     processed_results = []
    #     for results_per_image, input_per_image, image_size in zip(results, batched_inputs, image_sizes): # see if we can delete batched inputs here
    #         # height = input_per_image.get("height", image_size[0])
    #         # width = input_per_image.get("width", image_size[1])
    #         r = detector_postprocess(results_per_image, height, width) # part of post process
    #         processed_results.append({"instances": r})
    #     return processed_results