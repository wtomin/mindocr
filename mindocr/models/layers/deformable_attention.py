import math
import warnings
from typing import Optional, List, Tuple
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore import nn, ops, Tensor
import mindspore.common.dtype as mstype
import mindspore.common.initializer as init
from mindspore.common.initializer import XavierUniform, Uniform
from mindspore._checkparam import Validator
from mindspore import log as logger
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.context import ParallelMode
from mindspore.log import _LogActionOnce
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.transformer.layers import _LayerNorm, _Linear, \
    _args_type_validator_check, _valid_type_checks, _valid_value_checks, \
    _check_past_none_input_none, _check_input_dtype
from mindspore.nn.transformer.op_parallel_config import default_dpmp_config, _PipeLineConfig, OpParallelConfig, \
    _Config, _check_config, MoEParallelConfig
import mindspore.numpy as ms_np
from mindspore import ms_function

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MultiScaleDeformableAttention(nn.Cell):
    """
    Multi-Scale Deformable Attention Module used in Deformable-DETR

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        hidden_size (int): The embedding dimension of Attention. Default: 256.
        num_heads (int): The number of attention heads. Default: 8.
        num_levels (int): The number of feature map used in Attention. Default: 4.
        num_points (int): The number of sampling points for each query
            in each head. Default: 4.
        dropout (float): Dropout layer used in output. Default: 0.1.
    """
    @_LogActionOnce(logger=logger, key='MultiScaleDeformableAttention',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(hidden_size=Validator.check_positive_int,
                                num_heads=Validator.check_positive_int,
                                #src_seq_length=Validator.check_positive_int,
                                #tgt_seq_length=Validator.check_positive_int,
                                attention_dropout_rate=Validator.check_non_negative_float,
                                hidden_dropout_rate=Validator.check_non_negative_float,
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.float16],
                                                                  "MultiScaleDeformableAttention"),
                                softmax_compute_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                         "MultiScaleDeformableAttention"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                    "MultiScaleDeformableAttention"),
                                parallel_config=_valid_type_checks([OpParallelConfig],
                                                                   "MultiScaleDeformableAttention"))
    def __init__(
        self, batch_size: int,
        hidden_size: int,
        num_heads: int,
        src_seq_length: Optional[int] = None,
        tgt_seq_length: Optional[int] = None,
        num_levels: int = 4,
        num_points: int = 4,
        hidden_dropout_rate=0.0,
        attention_dropout_rate=0.0,
        compute_dtype=mstype.float32,
        softmax_compute_type=mstype.float32,
        param_init_type=mstype.float32,
        parallel_config=default_dpmp_config) -> None:
        super(MultiScaleDeformableAttention, self).__init__()
        self._is_ascend = context.get_context('device_target') in ["Ascend"]
        self.dp = parallel_config.data_parallel
        self.is_parallel_mode = _get_parallel_mode() in (
            ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        if batch_size:
            Validator.check_positive_int(batch_size)
        #if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
        _check_config(parallel_config)
        self.src_seq_length = src_seq_length
        self.tgt_seq_length = tgt_seq_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        if hidden_dropout_rate < 0 or hidden_dropout_rate >= 1:
            raise ValueError("For 'MultiScaleDeformableAttention', the class variable 'hidden_dropout_rate' must be "
                                "in range [0, 1.0), but got the value : {}.".format(hidden_dropout_rate))
        if attention_dropout_rate < 0 or attention_dropout_rate >= 1:
            raise ValueError("For 'MultiScaleDeformableAttention', the class variable 'attention_dropout_rate' must be "
                                "in range [0, 1.0), but got the value : {}.".format(attention_dropout_rate))
        if hidden_size % num_heads != 0:
            raise ValueError("For 'MultiScaleDeformableAttention', the class variable 'hidden_size' must be a multiple "
                                "of 'num_heads', but got the hidden_size is {} and the num_heads is {}."
                                .format(hidden_size, num_heads))
        if num_heads % parallel_config.model_parallel != 0:
            raise ValueError("For 'MultiScaleDeformableAttention', the class variable 'num_heads' must be a multiple of "
                                "'parallel_config.model_parallel', but got the num_heads is {} "
                                "and the parallel_config.model_parallel  is {}."
                                .format(num_heads, parallel_config.model_parallel))

        self.size_per_head = hidden_size // num_heads
        self.concat_k = P.Concat(axis=3)
        self.concat_v = P.Concat(axis=2)
        self.multiply_data = Tensor([
            -10000.0,
        ], dtype=softmax_compute_type)
        self.dropout = nn.Dropout(keep_prob=1 - hidden_dropout_rate)
        self.prob_dropout = nn.Dropout(keep_prob= 1 - attention_dropout_rate)

        self.cos = P.Cos()
        self.sin = P.Sin()
        if not _is_power_of_2(self.size_per_head ):
            warnings.warn(
                """
                You'd better set d_model in MSDeformAttn to make sure that
                each dim of the attention head a power of 2, which is more efficient.
                """
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.dtype = compute_dtype
        self.param_init_type = param_init_type

        self.sampling_offsets = nn.Dense(hidden_size, num_heads * num_levels * num_points *2)
        self.attention_weights = nn.Dense(hidden_size, num_heads * num_levels * num_points)
        self.value_proj = nn.Dense(hidden_size, hidden_size)
        self.output_proj = nn.Dense(hidden_size, hidden_size)    
    
    def _reset_parameters(self):
        
        pshape, dtype = self.sampling_offsets.weight.shape, self.sampling_offsets.weight.dtype
        self.sampling_offsets.weight.set_data(init.initializer('zeros', pshape, dtype))

        thetas = ops.arange(self.num_heads, dtype=self.param_init_type) * (2.0 * math.pi / self.num_heads)
        grid_init = ops.stack([self.cos(thetas), self.sin(thetas)], axis=-1)
        grid_init = grid_init  / ms_np.amax(ops.abs(grid_init), axis=-1, keepdims=True)
        grid_init = grid_init.reshape((self.num_heads, 1, 1, 2))
        grid_init = ops.repeat_elements(grid_init, self.num_levels, axis=1)
        grid_init = ops.repeat_elements(grid_init, self.num_points, axis=2)
        
        for i in range(self.num_points):
            grid_init[:, :, i, : ] *= i+1
        
        self.sampling_offsets.bias = ms.Parameter(grid_init.view(-1))
        self.sampling_offsets.bias = ops.stop_gradient(self.sampling_offsets.bias)

        self.attention_weights.weight.set_data(init.initializer('zeros', self.attention_weights.weight.shape, self.param_init_type))
        self.attention_weights.bias.set_data(init.initializer('zeros', self.attention_weights.bias.shape, self.param_init_type))
        self.value_proj.weight.set_data(init.initializer(init.XavierUniform(), self.value_proj.weight.shape, self.param_init_type))
        self.value_proj.bias.set_data(init.initializer('zeros', self.value_proj.bias.shape, self.param_init_type))

        self.output_proj.weight.set_data(init.initializer(init.XavierUniform(), self.output_proj.weight.shape, self.param_init_type))
        self.output_proj.bias.set_data(init.initializer('zeros', self.output_proj.bias.shape, self.param_init_type))

    def construct(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,  # not used in deformable-attn, for good layout
        value: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        reference_points: Optional[Tensor] = None,
        spatial_shapes: Optional[List[Tuple[int, int]]] = None,
        level_start_index: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:

        """
        Defines the computation to be performed.

        Args:
            query (Tensor): Query embeddings with shape
                `(num_query, bs, hidden_size)`
            key (Tensor): Key embeddings with shape
                `(num_key, bs, hidden_size)`
            value (Tensor): Value embeddings with shape
                `(num_key, bs, hidden_size)`
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default: None. If None, `query` will be
                used.
            query_pos (Tensor): The position embedding for `query`. Default: None.
            key_padding_mask (Tensor): ByteTensor for `query`, with shape `(bs, num_key)`,
                indicating which elements within `key` to be ignored in attention.
            reference_points (Tensor): The normalized reference points
                with shape `(bs, num_query, num_levels, 2)`,
                all elements is range in [0, 1], top-left (0, 0),
                bottom-right (1, 1), including padding are.
                or `(N, Length_{query}, num_levels, 4)`, add additional
                two dimensions `(h, w)` to form reference boxes.
            spatial_shapes (List[Tuple[int, int]]): Spatial shape of features in different levels.
                With shape `(num_levels, 2)`, last dimension represents `(h, w)`.
            level_start_index (Tensor): The start index of each level. A tensor with
                shape `(num_levels, )` which can be represented as
                `[0, h_0 * w_0, h_0 * w_0 + h_1 * w_1, ...]`.

        Returns:
            Tensor: forward results with shape `(num_query, bs, hidden_size)`
        """
        # self._check_inputs()
        ori_dtype = F.dtype(query)
        query = F.cast(query, self.dtype)

        if value is None:
            value = query
        value = F.cast(value, self.dtype)
        if query_pos is not None:
            query_pos =  F.cast(query_pos, self.dtype)
            query = query + query_pos
        
        bs = self.batch_size
        _, num_query, _ = query.shape
        _, num_value, _ = value.shape
        # assert query.shape[0] == bs and value.shape[0] == bs
        # assert  (spatial_shapes.asnumpy()[:, 0] * spatial_shapes.asnumpy()[:, 1]).sum() == num_value
        # assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)  # (bs, sum(hw), num_head, head_dim)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        # softmax to aggregate features of different level
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = ops.softmax(attention_weights, -1)  # (bs, sum(hw), num_head, num_level*num_point)
        attention_weights = self.prob_dropout(attention_weights)
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            spatial_shapes_array = ms_np.array(spatial_shapes)
            offset_normalizer = ops.stack([spatial_shapes_array[:, 1], spatial_shapes_array[:, 0]] ,axis=-1)  # (num_level, 2)
            normalized_offsets = sampling_offsets / offset_normalizer.view(1, 1, 1, -1, 1, 2)
            sampling_locations = reference_points.view(bs, num_query, 1, -1, 1, 2) + normalized_offsets
              # (bs, sum(hw), num_heads, num_levels, num_points, 2)
        elif reference_points.shape[-1] == 4:
            # modulate xy offset by hw
            sampling_locations = reference_points.view(bs, num_query, 1, -1, 1, 4)[...,:2] + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1])
            )
        # if False and ms.get_context('device_target') in {'GPU', 'Ascend'}:
        #     # TODO apply cuda version deform-attn
        #     output = MultiScaleDeformableAttnFunction.apply(
        #         value,
        #         spatial_shapes,
        #         level_start_index,
        #         sampling_locations,
        #         attention_weights,
        #         self.im2col_step,
        #     )
        # else:
        output = multi_scale_deformable_attn(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights
        )

        output = self.output_proj(output)
        output = self.dropout(output) 
        output = F.cast(output, ori_dtype)
        return output


def multi_scale_deformable_attn(
    value: Tensor,  # (bs, sum(hw), num_head, head_hidden_sizes)  head_hidden_sizes=hidden_size//num_head
    value_spatial_shapes: Tensor,  # (num_level, 2)
    level_start_index: Tensor,
    sampling_locations: Tensor,  # (bs, num_query, num_head, num_level, num_points, 2), normalized
    attention_weights: Tensor,
) -> Tensor:
   
    bs, _, num_heads, head_hidden_sizes = value.shape  # hidden_size is the one for head
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    # indices_or_sections = ops.cumsum(value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1], axis=0)[:-1]
    # split_sections = (value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]).astype(ms.int32).asnumpy().tolist()
    # value_list = ops.split(value, split_sections, axis=1)
    level_start_index = F.cast(level_start_index, mstype.int64)
    value_list = [value[:, level_start_index[i]: level_start_index[i+1], ...] for i in range(len(level_start_index)-1)]
    
    value_list.append(value[:, level_start_index[-1]:, ...])
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    # value_spatial_shapes_list = value_spatial_shapes.asnumpy().tolist()
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, head_hidden_sizes ->
        # bs, H_*W_, num_heads*head_hidden_sizes ->
        # bs, num_heads*head_hidden_sizes, H_*W_ ->
        # bs*num_heads, head_hidden_sizes, H_, W_
        value_l_ = (
            value_list[level].reshape(bs, H_ * W_, -1).transpose((0, 2, 1)).reshape(bs * num_heads, head_hidden_sizes, H_, W_)
        )
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose((0, 2, 1, 3, 4)).reshape(
            bs * num_heads, num_queries, num_points, 2)
        # bs*num_heads, head_hidden_sizes, num_queries, num_points
        sampling_value_l_ = ops.grid_sample(
            value_l_, sampling_grid_l_, interpolation_mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose((0, 2, 1, 3, 4)).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )

    # (bs*num_heads, head_hidden_sizes, num_queries, num_levels, num_points) ->
    # (bs*num_heads, head_hidden_sizes, num_queries, num_levels*num_points) ->
    # (bs*num_heads, head_hidden_sizes, num_queries, num_levels*num_points) ->
    # (bs*num_heads, head_hidden_sizes, num_queries) -> [aggregate among level and pts axis]
    # (bs, num_heads*head_hidden_sizes, num_queries)
    output = (
        (ops.stack(sampling_value_list, axis=-2).reshape(bs * num_heads, head_hidden_sizes, num_queries, -1) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * head_hidden_sizes, num_queries)
    )
    # (bs, num_queries, hidden_sizes)  hidden_sizes = num_heads*head_hidden_sizes
    return output.transpose((0, 2, 1))