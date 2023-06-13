import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, Tensor
from mindspore import dtype as mstype
from mindspore.common.initializer import Normal
import math
class PositionalEncoding1D(nn.Cell):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.channels = num_pos_feats
        dim_t = ops.range(Tensor(0, mstype.int32), Tensor(self.channels, mstype.int32), Tensor(2, mstype.int32)).astype(mstype.float32)
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.normalize = normalize
        inv_freq = 1. / (temperature ** (dim_t / self.channels))
        self.inv_freq = ms.Parameter(Tensor(inv_freq, mstype.float32), name="inv_freq") 

    def construct(self, tensor):
        if tensor.ndim != 2:
            raise RuntimeError("The input tensor has to be 2D!")
        x, orig_ch = tensor.shape
        pos_x = ops.range(
            Tensor(1, mstype.int32), Tensor(x + 1, mstype.int32), Tensor(1, mstype.int32)).astype(tensor.dtype)

        if self.normalize:
            eps = 1e-6
            pos_x = pos_x / (pos_x[-1:] + eps) * self.scale

        sin_inp_x = ops.matmul(pos_x[:, None], self.inv_freq[None])
        emb_x = ops.concat([ops.sin(sin_inp_x), ops.cos(sin_inp_x)], axis=-1)
        emb = ops.zeros((x, self.channels), tensor.dtype)
        emb[:, :self.channels] = emb_x

        return emb[:, :orig_ch]

class PositionalEncoding2D(nn.Cell):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    def construct(self, image_masks):
        mask = image_masks
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=mstype.float32)
        x_embed = not_mask.cumsum(2, dtype=mstype.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = ops.range(Tensor(0, mstype.int32), Tensor(self.num_pos_feats, mstype.int32), Tensor(1, mstype.int32))
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = ops.unsqueeze(x_embed, dim=3) / dim_t 
        pos_y = ops.unsqueeze(y_embed, dim=3) / dim_t
        pos_x = ops.stack([ops.sin(pos_x[:, :, :, 0::2]), 
                           ops.cos(pos_x[:, :, :, 1::2])], axis=4)
        
        s1, s2, s3, _, _ = pos_x.shape
        pos_x = ops.reshape(pos_x, (s1, s2, s3, -1))

        pos_y = ops.stack([ops.sin(pos_y[:, :, :, 0::2]), 
                                 ops.cos(pos_y[:, :, :, 1::2])], axis=4)
        s1, s2, s3, _, _ = pos_y.shape
        pos_y = ops.reshape(pos_y, (s1, s2, s3, -1))
        pos = ops.transpose(ops.concat((pos_y, pos_x), axis=3), (0, 3, 1, 2))
        return pos