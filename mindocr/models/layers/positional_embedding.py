import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, Tensor
from mindspore import dtype as mstype

from mindspore.common.initializer import Normal

class PositionalEncoding1D(nn.Cell):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.channels = num_pos_feats
        dim_t = ops.arange(0, self.channels, 2).astype(np.float32)
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * np.pi
        self.scale = scale
        self.normalize = normalize
        inv_freq = 1. / (temperature ** (dim_t / self.channels))
        self.inv_freq = ms.Parameter(Tensor(inv_freq, mstype.float32), name="inv_freq") 

    def construct(self, tensor):
        if tensor.ndim != 2:
            raise RuntimeError("The input tensor has to be 2D!")
        x, orig_ch = tensor.shape
        pos_x = ops.arange(
            1, x + 1, dtype=tensor.dtype).astype(self.inv_freq.dtype)

        if self.normalize:
            eps = 1e-6
            pos_x = pos_x / (pos_x[-1:] + eps) * self.scale

        sin_inp_x = ops.Einsum()("i,j->ij", pos_x, self.inv_freq)
        emb_x = ops.Concat()(ops.Sin()(sin_inp_x), ops.Cos()(sin_inp_x), 1)
        emb = ops.ZerosLike()(tensor)
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
            scale = 2 * np.pi
        self.scale = scale
    def construct(self, tensors):
        x = tensors
        mask = tensors.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=np.float32)
        x_embed = not_mask.cumsum(2, dtype=np.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = ops.arange(self.num_pos_feats, dtype=np.float32)
        dim_t = self.temperature ** (2 * ops.FloorDiv()(dim_t, 2) / self.num_pos_feats)

        pos_x = ops.unsqueeze(x_embed, axis=3) / ops.unsqueeze(dim_t, axis=0)
        pos_y = ops.unsqueeze(y_embed, axis=3) / ops.unsqueeze(dim_t, axis=0)
        pos_x = ops.Concat()(ops.Sin()(pos_x[:, :, :, 0::2]), ops.Cos()(pos_x[:, :, :, 1::2]), 3)
        pos_y = ops.Concat()(ops.Sin()(pos_y[:, :, :, 0::2]), ops.Cos()(pos_y[:, :, :, 1::2]), 3)
        pos = ops.Concat()(pos_y, pos_x, 1)
        pos = ops.Permute()(pos, (0, 3, 1, 2))
        return pos