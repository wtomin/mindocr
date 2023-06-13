from typing import Optional, List
from mindspore import nn, ops, Tensor
import mindspore.common.dtype as mstype
import mindspore.numpy as mnp
from mindspore.ops import functional as F
import mindspore.ops.operations as P
import copy
import numpy as np
import mindspore as ms

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



def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


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
        return ops.sigmoid(x) * 2 - 0.5
    else:
        return ops.sigmoid(x)

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
    assert (b[2]>=0).all() and (b[3] >= 0).all()
    return ops.stack(b, -1)

def box_area(boxes):
    """
    Computes the area of a set of bounding boxes (xmin, ymin, xmax, ymax)
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = mnp.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = mnp.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union+1e-6)
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

    lt = mnp.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = mnp.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0) # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1] # what if area contains zeros

    return iou - (area - union) / (area+1e-6)

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = Tensor(0.0, dtype=ms.float32)
        self.avg = Tensor(0.0, dtype=ms.float32)
        self.sum = Tensor(0.0, dtype=ms.float32)
        self.count = Tensor(0.0, dtype=ms.float32)

    def update(self, val: Tensor, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count