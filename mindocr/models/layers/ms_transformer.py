# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Transformer Cells module, include MultiheadAttention
"""
from mindspore.ops.function.nn_func import pad
import math
from typing import Union, Optional
import mindspore
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer, XavierNormal, XavierUniform, \
    HeUniform, Uniform, _calculate_fan_in_and_fan_out
from mindspore.nn.cell import Cell
from mindspore.nn import Dense, Dropout, ReLU, GELU, LayerNorm, CellList
from mindspore.ops.primitive import constexpr
from mindspore.ops._primitive_cache import _get_cache_prim
import mindspore.common.dtype as mstype
__all__ = ['MultiheadAttention']
def linear(x, w, b):
    """inner linear"""
    out = ops.matmul(x, w.swapaxes(-1, -2))
    if b is not None:
        out = out + b
    return out
def _inner_dropout(x, p, training):
    """inner dropout"""
    _dropout = _get_cache_prim(P.Dropout)(1 - p)
    if p > 0. and training:
        return _dropout(x)[0]
    return x
def _in_projection(q, k, v, w_q, w_k, w_v, b_q=None, b_k=None, b_v=None):
    """in projection function"""
    Eq, Ek, Ev = q.shape[-1], k.shape[-1], v.shape[-1]
    w_q_shape, w_k_shape, w_v_shape = w_q.shape, w_k.shape, w_v.shape

    if w_q_shape != (Eq, Eq):
        raise ValueError(f"Expecting query weights shape of {(Eq, Eq)}, but got {w_q_shape}")
    if w_k_shape != (Eq, Ek):
        raise ValueError(f"Expecting key weights shape of {(Eq, Ek)}, but got {w_k_shape}")
    if w_v_shape != (Eq, Ev):
        raise ValueError(f"Expecting value weights shape of {(Eq, Ev)}, but got {w_v_shape}")
    if b_q is not None:
        b_q_shape = b_q.shape
        if b_q_shape != (Eq,):
            raise ValueError(f"Expecting query bias shape of {(Eq,)}, but got {b_q_shape}")
    if b_k is not None:
        b_k_shape = b_k.shape
        if b_k_shape != (Eq,):
            raise ValueError(f"Expecting key bias shape of {(Eq,)}, but got {b_k_shape}")
    if b_v is not None:
        b_v_shape = b_v.shape
        if b_v_shape != (Eq,):
            raise ValueError(f"Expecting value bias shape of {(Eq,)}, but got {b_v_shape}")
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection_packed(q, k, v, w, b, k_is_v, q_is_k):
    """in projecktion packed function"""
    E = q.shape[-1]
    if k_is_v:
        if q_is_k:
            # self-attention
            return linear(q, w, b).tensor_split(3, axis=-1)
        # encoder-decoder attention
        w_q, w_kv = w.split([E, E * 2])
        if b is None:
            b_q = b_kv = None
        else:
            b_q, b_kv = b.split([E, E * 2])
        return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).tensor_split(2, axis=-1)
    w_q, w_k, w_v = w.tensor_split(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.tensor_split(3)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, is_training):
    """scaled dot product attention"""
    embed_size = query.shape[-1]
    scaling_factor = Tensor(embed_size, mstype.float32).sqrt().sqrt()
    query = query / scaling_factor

    if is_causal:
        L = query.shape[-2]
        S = key.shape[-2]
        attn_mask = ops.ones((L, S), mstype.bool_).tril()

    attn = ops.matmul(query, key.swapaxes(-2, -1) / scaling_factor)
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = ops.softmax(attn, -1)
    attn = _inner_dropout(attn, dropout_p, is_training)
    output = ops.matmul(attn, value)

    return (output, attn)


@constexpr
def _check_qkv_shape(query_ndim, key_ndim, value_ndim):
    """Check the expected shape for `query, `key`, `value` and returns whether the input is batched."""
    # Shape check.
    if query_ndim == 3:
        # Batched Inputs
        is_batched = True
        if key_ndim != 3 or value_ndim != 3:
            raise ValueError(f"For batched `query`, the `key` and `value` must be 3D tensor, "
                             f"but got `key` with {key_ndim}D and `value` with {value_ndim}D.")
    elif query_ndim == 2:
        # Unbatched Inputs
        is_batched = False
        if key_ndim != 2 or value_ndim != 2:
            raise ValueError(f"For unbatched `query`, the `key` and `value` must be 2D tensor, "
                             f"but got `key` with {key_ndim}D and `value` with {value_ndim}D.")
    else:
        raise ValueError(f"The `query` should be unbatched 2D or batched 3D tensor, "
                         f"but got `query` with {query_ndim}D.")

    return is_batched


@constexpr
def _check_kpm_shape(query_ndim, kmp_ndim):
    """check key_padding_mask shape"""
    if query_ndim == 3:
        if kmp_ndim != 2:
            raise ValueError(f"For batched `query`, the `key_padding_mask` must be `None` or 2D, "
                             f"but got `key_padding_mask` with {kmp_ndim}D.")

    elif query_ndim == 2:
        if kmp_ndim != 1:
            raise ValueError(f"For unbatched `query`, the `key_padding_mask` must be `None` or 1D, "
                             f"but got `key_padding_mask` with {kmp_ndim}D.")


@constexpr
def _check_attn_mask_shape(query_ndim, query_shape, key_shape, attn_mask_ndim,
                           attn_mask_shape, num_heads):
    """
    Check the expected shape for `attn_mask`.
    """
    # Shape check.
    if query_ndim == 3:
        if attn_mask_ndim not in (2, 3):
            raise ValueError(f"For batched `query`, the `attn_mask` must be `None`, 2-D or 3-D, "
                             f"but got `attn_mask` with{attn_mask_ndim}D.")
    elif query_ndim == 2:
        if attn_mask_ndim not in (2, 3):
            raise ValueError(f"For unbatched `query`, the `attn_mask` must be `None`, 2-D or 3-D, "
                             f"but got `attn_mask` with{attn_mask_ndim}D.")

        if attn_mask_ndim == 3:
            expected_shape = (num_heads, query_shape[0], key_shape[0])
            if attn_mask_shape != expected_shape:
                raise ValueError(f"The shape of `attn_mask` must to be {expected_shape}, "
                                 f"but got {attn_mask_shape}.")
def _inner_pad(x, padding, value=None):
    """inner pad function for bool type."""
    x_dtype = x.dtype
    if x_dtype == mstype.bool_:
        x = x.astype(mstype.int32)
    x = pad(x, padding, value=value)
    x = x.astype(x_dtype)
    return x

def multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, in_proj_weight,
                                 in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
                                 out_proj_bias, training=True, key_padding_mask=None, attn_mask=None,
                                 use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None,
                                 v_proj_weight=None, static_k=None, static_v=None, average_attn_weights=True,
                                 is_causal=False, k_is_v=False, q_is_k=False):
    """multi head attetion forward function"""
    is_batched = _check_qkv_shape(query.ndim, key.ndim, value.ndim)
    if key_padding_mask is not None:
        _check_kpm_shape(query.ndim, key_padding_mask.ndim)
    if attn_mask is not None:
        _check_attn_mask_shape(query.ndim, query.shape, key.shape, attn_mask.ndim,
                               attn_mask.shape, num_heads)

    if not is_batched:
        query = query.expand_dims(1)
        key = key.expand_dims(1)
        value = value.expand_dims(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.expand_dims(0)

    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    if key_padding_mask is not None:
        _kpm_dtype = key_padding_mask.dtype
        if _kpm_dtype != mstype.bool_ and not ops.is_floating_point(key_padding_mask):
            raise ValueError("The `key_padding_mask` only supports bool and floating dtypes.")
    if embed_dim != embed_dim_to_check:
        raise ValueError(f"The `embed_dim` should be {embed_dim_to_check}, but got {embed_dim}.")

    head_dim = embed_dim // num_heads
    if head_dim * num_heads != embed_dim:
        raise ValueError(f"The `embed_dim` {embed_dim} can not be divisible by `num_heads` {num_heads}.")
    if use_separate_proj_weight:
        # allow MHA to have different embedding dims when separate projection weights are used
        if key.shape[:2] != value.shape[:2]:
            raise ValueError(f"The sequence length and batch dims of `key`: {key.shape[:2]} do not match "
                             f"`value`: {value.shape[:2]}.")
    else:
        if key.shape != value.shape:
            raise ValueError(f"The shape of `key` {key.shape} does not match `value` {value.shape}.")

    # compute in-projection
    if not use_separate_proj_weight:
        if in_proj_weight is None:
            raise ValueError("`use_separate_proj_weight` is ``False`` but `in_proj_weight` got ``None``.")
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias, k_is_v, q_is_k)
    else:
        if q_proj_weight is None:
            raise ValueError("`use_separate_proj_weight` is ``True`` but `q_proj_weight` got ``None``.")
        if k_proj_weight is None:
            raise ValueError("`use_separate_proj_weight` is ``True`` but `k_proj_weight` got ``None``.")
        if v_proj_weight is None:
            raise ValueError("`use_separate_proj_weight` is ``True`` but `v_proj_weight` got ``None``.")
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.tensor_split(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == mstype.uint8:
            attn_mask = attn_mask.astype(mstype.bool_)
        else:
            if not ops.is_floating_point(attn_mask) and attn_mask.dtype != mstype.bool_:
                raise ValueError(f"`attn_mask` only support float, byte, and bool types, "
                                 f"but got not {attn_mask.dtype}.")
        # ensure attn_mask's ndim is 3
        if attn_mask.ndim == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise ValueError(f"The shape of the `attn_mask` should be {correct_2d_size}, "
                                 f"but got {attn_mask.shape}.")
            attn_mask = attn_mask.expand_dims(0)
        elif attn_mask.ndim == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise ValueError(f"The shape of the `attn_mask` should be {correct_3d_size}, "
                                 f"but got {attn_mask.shape}.")
        else:
            raise ValueError(f"The ndim of `attn_mask` only support 2 or 3, "
                             f"but got {attn_mask.ndim}.")

    if bias_k is not None and bias_v is not None:
        if static_k is not None:
            raise ValueError("The bias_k cannot be added to static_k.")
        if static_v is not None:
            raise ValueError("The bias_v cannot be added to static_v.")
        k = ops.cat([k, bias_k.tile((1, bsz, 1))])
        v = ops.cat([v, bias_v.tile((1, bsz, 1))])
        if attn_mask is not None:
            attn_mask = _inner_pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = _inner_pad(key_padding_mask, (0, 1))
    else:
        if bias_k is not None or bias_v is not None:
            raise ValueError("The bias_k and bias_v should be ``None``"
                             "at the same time.")

    q = q.view((tgt_len, bsz * num_heads, head_dim)).swapaxes(0, 1)
    if static_k is None:
        k = k.view((k.shape[0], bsz * num_heads, head_dim)).swapaxes(0, 1)
    else:
        if static_k.shape[0] != bsz * num_heads:
            raise ValueError(f"The shape[0] of `static_k` should be {bsz * num_heads}, "
                             f"but got {static_k.shape[0]}")
        if static_k.shape[2] != head_dim:
            raise ValueError(f"The shape[2] of `static_k` should be {head_dim}, "
                             f"but got {static_k.shape[2]}")
        k = static_k
    if static_v is None:
        v = v.view((v.shape[0], bsz * num_heads, head_dim)).swapaxes(0, 1)
    else:
        if static_v.shape[0] != bsz * num_heads:
            raise ValueError(f"The shape[0] of `static_v` should be {bsz * num_heads}, "
                             f"but got {static_v.shape[0]}")
        if static_v.shape[2] != head_dim:
            raise ValueError(f"The shape[2] of `static_v` should be {head_dim}, "
                             f"but got {static_v.shape[2]}")
        v = static_v

    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = ops.cat([k, ops.zeros(zero_attn_shape, dtype=k.dtype)], axis=1)
        v = ops.cat([v, ops.zeros(zero_attn_shape, dtype=v.dtype)], axis=1)
        if attn_mask is not None:
            attn_mask = _inner_pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = _inner_pad(key_padding_mask, (0, 1))

    src_len = k.shape[1]

    if key_padding_mask is not None:
        if key_padding_mask.shape != (bsz, src_len):
            raise ValueError(f"The shape of `key_padding_mask` should be {(bsz, src_len)}, "
                             f"but got {key_padding_mask.shape}.")

        key_padding_mask = key_padding_mask.view((bsz, 1, 1, src_len)). \
            tile((1, num_heads, 1, 1)).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == mstype.bool_:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask + key_padding_mask

    if attn_mask is not None and attn_mask.dtype == mstype.bool_:
        new_attn_mask = ops.zeros_like(attn_mask, dtype=q.dtype)
        attn_mask = new_attn_mask.masked_fill(attn_mask, float("-inf"))

    if attn_mask is not None:
        if attn_mask.shape[0] == 1:
            attn_mask = attn_mask.expand_dims(0)
        else:
            attn_mask = attn_mask.view((bsz, num_heads, -1, src_len))

    q = q.view((bsz, num_heads, tgt_len, head_dim))
    k = k.view((bsz, num_heads, src_len, head_dim))
    v = v.view((bsz, num_heads, src_len, head_dim))

    attn_output, attn_output_weights = _scaled_dot_product_attention(
        q, k, v, attn_mask, dropout_p, is_causal, training)
    attn_output = attn_output.transpose(2, 0, 1, 3).view((bsz * tgt_len, embed_dim))

    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view((tgt_len, bsz, attn_output.shape[1]))

    attn_output_weights = attn_output_weights.view((bsz, num_heads, tgt_len, src_len))
    if average_attn_weights:
        attn_output_weights = attn_output_weights.sum(axis=1) / num_heads

    if not is_batched:
        attn_output = attn_output.squeeze(1)
        attn_output_weights = attn_output_weights.squeeze(0)
    return attn_output, attn_output_weights


class _Linear(Dense):
    def __init__(self, in_channels, out_channels, has_bias=True):
        fan_in, _ = _calculate_fan_in_and_fan_out((out_channels, in_channels))
        bound = 1 / math.sqrt(fan_in)
        super().__init__(in_channels, out_channels, weight_init=HeUniform(math.sqrt(5)),
                         bias_init=Uniform(bound), has_bias=has_bias, activation=None)


class MultiheadAttention(Cell):
    r"""
    This is an implementation of multihead attention in the paper `Attention is all you need
    <https://arxiv.org/pdf/1706.03762v5.pdf>`_. Given the query vector with source length, and the
    key and value vector with target length, the attention will be performed as the following

    .. math::
        MultiHeadAttention(query, key, vector) = Concat(head_1, \dots, head_h)W^O

    where :math:`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`. The default is with a bias.

    if query, key and value tensor is same, then it will be self attention.

    Args:
        embed_dim (int): Total dimension of MultiheadAttention.
        num_heads (int): Number of attention heads. Note that `embed_dim` will be split
            across `num_heads` (i.e. each head will have dimension `embed_dim // num_heads`).
        dropout (float): Dropout probability of `attn_output_weights`. Default: ``0.0``.
        has_bias (bool): Whether adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv (bool): Whether adds bias to the key and value sequences at axis=0. Default: ``False``.
        add_zero_attn (bool): Whether adds a new batch of zeros to the key and value sequences at axis=1.
            Default: ``False``.
        kdim (int): Total number of features for keys. Default: ``None`` (`kdim=embed_dim`).
        vdim (int): Total number of features for values. Default: ``None`` (`vdim=embed_dim`).
        batch_first (bool): If ``True``, then the input and output shape are :math:`(batch, seq, feature)` ,
            else :math:`(seq, batch, feature)` . Default: ``False``.

    Inputs:
        - **query** (Tensor): The query embeddings. If `query` is unbatched, the shape is :math:`(L, E_q)`,
          otherwise the shape is :math:`(L, N, E_q)` when `batch_first=False` or :math:`(N, L, E_q)` when
          `batch_first=True`, where :math:`L`is the target sequence length, :math:`N` is the batch size,
          and :math:`E_q` is the query embedding dimension `embed_dim`. Queries are compared against
          key-value pairs to produce the output. See "Attention Is All You Need" for more details.
        - **key** (Tensor): The key embeddings. If `key` is unbatched, the shape is :math:`(S, E_k)`, otherwise
          the shape is :math:`(S, N, E_k)` when `batch_first=False` or :math:`(N, S, E_k)` when
          `batch_first=True`, where :math:`S` is the source sequence length, :math:`N` is the batch size,
          and :math:`E_k` is the key embedding dimension `kdim`. See "Attention Is All You Need" for more details.
        - **value** (Tensor): The value embeddings. If `value` is unbatched, the shape is :math:`(S, E_v)`,
          otherwise the shape is :math:`(S, N, E_v)` when `batch_first=False` or :math:`(N, S, E_v)` when
          `batch_first=True`, where :math:`S` is the source sequence length, :math:`N` is the batch size,
          and :math:`E_v` is the value embedding dimension `vdim`. See "Attention Is All You Need" for more details.
        - **key_padding_mask** (Tensor, optional): If specified, a mask of shape :math:`(N, S)` indicating which
          elements within `key` to ignore for the purpose of attention (i.e. treat as "padding").
          For unbatched `query`, shape should be :math:`(S)`. Binary and byte masks are supported.
          For a binary mask, a ``True`` value indicates that the corresponding `key` value will be ignored for
          the purpose of attention. For a float mask, it will be directly added to the corresponding `key` value.
        - **need_weights** (bool): Whether returns `attn_output_weights` in addition to `attn_outputs`.
          Default: ``True``.
        - **attn_mask** (Tensor, optional): If specified, a 2D or 3D mask preventing attention to certain positions.
          Must be of shape :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the
          batch size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length.
          A 2D mask will be broadcasted across the batch while a 3D mask allows for a different mask for each entry
          in the batch. Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates
          that the corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that
          the corresponding position is not allowed to attend. For a float mask, the mask values will be added to
          the attention weight.
        - **average_attn_weights** (bool): If true, indicates that the returned `attn_weights` should be averaged
          across heads. Otherwise, `attn_weights` are provided separately per head. Note that this flag only
          has an effect when `need_weights=True`. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        Tuple, a tuple contains(`attn_output`, `attn_output_weights`)

        - **attn_output** - Attention outputs. If input is unbatched, the output shape is :math:`(L, E)`, otherwise
          the output shape is :math:`(L, N, E)` when `batch_first=False` or :math:`(N, L, E)` when
          `batch_first=True`, where :math:`L` is the target sequence length, :math:`N` is the batch size,
          and :math:`E` is the embedding dimension `embed_dim`.
        - **attn_output_weights** - Only returned when `need_weights=True`. If `average_attn_weights=True`,
          returns attention weights averaged across heads with shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)` when input is batched, where :math:`N` is the batch size, :math:`L` is
          the target sequence length, and :math:`S` is the source sequence length.
          If `average_attn_weights=False`, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or
          :math:`(N, \text{num\_heads}, L, S)` when input is batched.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> embed_dim, num_heads = 128, 8
        >>> seq_length, batch_size = 10, 8
        >>> query = Tensor(np.random.randn(seq_length, batch_size, embed_dim), mindspore.float32)
        >>> key = Tensor(np.random.randn(seq_length, batch_size, embed_dim), mindspore.float32)
        >>> value = Tensor(np.random.randn(seq_length, batch_size, embed_dim), mindspore.float32)
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
        >>> print(attn_output.shape)
        (10, 8, 128)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., has_bias=True, add_bias_kv=False,
                 add_zero_attn=False, kdim=None, vdim=None, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("The init argument 'embed_dim' must be divisible by 'num_heads'.")

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, embed_dim)), 'q_proj_weight')
            self.k_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, self.kdim)), 'k_proj_weight')
            self.v_proj_weight = Parameter(initializer(XavierUniform(), (embed_dim, self.vdim)), 'v_proj_weight')
            self.in_proj_weight = None
        else:
            self.in_proj_weight = Parameter(initializer(XavierUniform(), (3 * embed_dim, embed_dim)), 'in_proj_weight')
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None

        if has_bias:
            self.in_proj_bias = Parameter(initializer('zeros', (3 * embed_dim)), 'in_proj_bias')
        else:
            self.in_proj_bias = None
        self.out_proj = _Linear(embed_dim, embed_dim, has_bias=has_bias)

        if add_bias_kv:
            self.bias_k = Parameter(initializer(XavierNormal(), (1, 1, embed_dim)), 'bias_k')
            self.bias_v = Parameter(initializer(XavierNormal(), (1, 1, embed_dim)), 'bias_v')
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.k_is_v = False
        self.q_is_k = False

    def __call__(self, *args, **kwargs):
        query = kwargs.get('query', args[0])
        key = kwargs.get('key', args[1])
        value = kwargs.get('value', args[2])
        self.k_is_v = key is value
        self.q_is_k = query is key
        return super().__call__(*args, **kwargs)

    def construct(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                  need_weights: bool = True, attn_mask: Optional[Tensor] = None, average_attn_weights: bool = True):
        is_batched = query.ndim == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != mindspore.bool_ and not ops.is_floating_point(key_padding_mask):
                raise ValueError(
                    "only bool and floating types of key_padding_mask are supported")

        if self.batch_first and is_batched:
            # k_is_v and q_is_k preprocess in __call__ since Graph mode do not support `is`
            if self.k_is_v:
                if self.q_is_k:
                    query = key = value = query.swapaxes(1, 0)
                else:
                    query, key = [x.swapaxes(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.swapaxes(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights,
                k_is_v=self.k_is_v, q_is_k=self.q_is_k)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights,
                k_is_v=self.k_is_v, q_is_k=self.q_is_k)

        if self.batch_first and is_batched:
            attn_output = attn_output.swapaxes(1, 0)
        if need_weights:
            return attn_output, attn_output_weights
        return (attn_output,)




