from typing import Optional, List
from mindspore import nn, ops, Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
import mindspore.ops.operations as P
import copy
import numpy as np
import yaml
import os
from addict import Dict
import inspect
def load_yaml_with_base(yaml_file_path):
    assert os.path.exists(yaml_file_path), "file {} not found".format(yaml_file_path)
    with open(yaml_file_path) as fp:
        config = yaml.safe_load(fp)
        config = Dict(config)
        if '_base_' not in config:
            return config
        else:
            assert isinstance(config._base_ , str), "expected a string, but got {}".format(type(config._base_))
            base_config_fp = os.path.join(os.path.dirname(yaml_file_path), config._base_)
            assert os.path.exists(base_config_fp), "file {} not found".format(config._base_)
            base_config = load_yaml_with_base(base_config_fp)
            config.pop('_base_')
            base_config.update(config)
            config = base_config
            return config


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.size() == 0:
        return [Tensor(0)]
    if target.ndim == 2:
        assert output.ndim == 3
        output = P.ReduceMean(keep_dims=False)(output, 1)
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = P.TopK(sorted=True)(output, maxk)
    pred = pred.transpose()

    target = ops.broadcast_to(target.reshape((1, -1)),pred.shape())
    correct = ops.Equal()(pred, target)

    res = []
    for k in topk:
        correct_k = correct[:, :k].reshape((-1,)) # correct[:, :k] or # correct[:k]
        correct_k = correct_k.astype(np.float32).reduce_sum()
        res.append(correct_k * 100.0 / batch_size)
    return res

# class NestedTensor(object):
    
#     def __init__(self, tensor, mask: Optional[Tensor]):
#         self.tensor = tensor
#         self.mask = mask
#         pixel_indices = np.where(mask==0)
#         min_x, min_y = pixel_indices[0].min(), pixel_indices[1].min()
#         max_x, max_y = pixel_indices[0].max(), pixel_indices[1].max()
#         size = [max_x - min_x+1, max_y - min_y+1]
        
#         self.image_sizes = size
#     def decompose(self):
#         return self.tensor, self.mask

#     def __repr__(self):
#         return str(self.tensor)

# def nested_tensor_from_batch(image, image_mask, polys, rec_ids, ignore_tags, gt_classes, BatchInfo):
#     nested_tensor = NestedTensor(image, image_mask)
#     return nested_tensor, polys, rec_ids, ignore_tags, gt_classes, BatchInfo

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        tensor = ops.zeros(batch_shape, dtype=dtype)
        mask = ops.zeros((b, h, w), dtype=mstype.bool_)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

class MLP(nn.Cell):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()

        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.CellList([nn.Dense(n, k) for n, k in zip([input_dim] + h, h + [output_dim])])

    def construct(self, x):
        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x)) if i < self.num_layers - 1 else self.layers[i](x)
        return x
def _get_clones(module, N):
    return nn.CellList([copy.deepcopy(module) for i in range(N)])


def sigmoid_offset(x, offset=True):
    # modified sigmoid for range [-0.5, 1.5]
    if offset:
        return x.sigmoid() * 2 - 0.5
    else:
        return x.sigmoid()

def inverse_sigmoid(x, eps=1e-5):
    x = ops.clip_by_value(x, 0, 1)
    x1 = ops.clip_by_value(x, eps, x.max())
    x2 = ops.clip_by_value((1 - x), eps, (1 - x).max())
    return ops.log(x1/x2)
def inverse_sigmoid_offset(x, eps=1e-5, offset=True):
    if offset:
        x = (x + 0.5) / 2.0
    return inverse_sigmoid(x, eps)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = ops.unbind(x, -1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return ops.stack(b, -1)


def box_xyxy_to_cxcywh(x):
    x_min, x_max = x[:,:, 0].min(-1), x[:, :, 0].max(-1)
    y_min, y_max = x[:, :, 1].min(-1), x[:,:, 1].max(-1)
    b = [(x_min + x_max) / 2, (y_min + y_max) / 2, (x_max - x_min), (y_max - y_min)]
    assert (b[2]>0).all() and (b[3] > 0).all()
    return ops.stack(b, -1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

class ImageTensorWithMask:
    def __init__(self, image, mask: Optional[Tensor]) -> None:
       self.image = image
       self.mask = mask

    def fetch_image_mask(self,):
        return self.image, self.mask
    def __repr__(self) -> str:
        return "Image:"+str(self.image) +"\nMask"+str(self.mask)

# resize image according the largest image in the same batch
class BatchPadding:
    def __init__(self, input_columns, pad_image = True, pad_polys = True):
        self.input_columns = input_columns
        self.output_columns = input_columns+['mask', 'valid_length']
        self.pad_image = pad_image
        self.pad_polys = pad_polys

    
    def pad_DetData(self, image, polys, boxes, texts, rec_ids, ignore_tags, gt_classes, batchinfo):
        images_list = image 
        if images_list[0].ndim == 3 and self.pad_image:
            max_shape = [img.shape for img in images_list]
            max_shape = np.array(max_shape).max(axis=0)
            batch_shape = (len(images_list), max_shape[0], max_shape[1], max_shape[2])
            b, c, h, w = batch_shape
            new_tensors = np.zeros(batch_shape)
            masks = np.ones((b, h, w), dtype=np.bool)
            for img, paded_img, mask in zip(images_list, new_tensors, masks):
                paded_img[:, :img.shape[-2], :img.shape[-1]] = img
                mask[:img.shape[-2], :img.shape[-1]] = False
        else:
            raise ValueError("images_list should be a list of 3D tensors")

        if self.pad_polys:
            #pad polys, texts and ignore_tags, ensure they have the same length
            max_len = max([len(poly) for poly in polys])
            new_polys, new_boxes, new_texts, new_rec_ids, new_ignore_tags, new_gt_classes = [], [], [], [], [], []
            valid_label_lengths = []
            for poly, box, text, rec_id, ignore_tage, gt in zip(polys, boxes, texts, rec_ids, ignore_tags, gt_classes):
                new_polys.append([x for x in poly] + [poly[0]] * (max_len - len(poly)))
                new_boxes.append([x for x in box] + [box[0]] * (max_len - len(box)))
                new_texts.append([x for x in text] + ['###'] * (max_len - len(text)))
                new_rec_ids.append([x for x in rec_id] + [96] * (max_len - len(rec_id)))
                new_ignore_tags.append([x for x in ignore_tage] + [True] * (max_len - len(ignore_tage)))
                new_gt_classes.append([x for x in gt] + [1] * (max_len - len(gt)))
                valid_label_lengths.append(len(poly))
        
        return_list= [new_tensors, new_polys, new_boxes, new_texts, new_rec_ids, new_ignore_tags, new_gt_classes, mask, valid_label_lengths]
        return return_list
