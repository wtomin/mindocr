from .deformable_attention import MultiScaleDeformableAttention
from typing import Optional, List, Tuple
import math
import mindspore as ms
from mindspore import nn, ops, Tensor
import mindspore.common.dtype as mstype

from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import initializer 
from mindspore._checkparam import Validator
from mindspore import log as logger
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.context import ParallelMode
from mindspore.log import _LogActionOnce
from mindspore.nn.transformer.layers import _LayerNorm, _Linear, \
    _args_type_validator_check, _valid_type_checks, _valid_value_checks, \
    _check_past_none_input_none, _check_input_dtype
from mindspore.nn.transformer.op_parallel_config import default_dpmp_config, _PipeLineConfig, OpParallelConfig, \
    _Config, _check_config, MoEParallelConfig

from mindocr.utils.misc import inverse_sigmoid, _get_clones

class DeformableTransformerEncoderLayer(nn.Cell):
    """
    This ia an implementation of Deformable Transformer Encoder Layer. It is composed of a MultiScale Deformable Attention, 
    a LayerNorm, and a feedforward network.
    Args:
        batch_size (int): The batch size of the input tensor.
        hidden_size (int): number of features in the input.
        ffn_hidden_size (int): hidden size of the feedforward network (ffn) layers.
        num_heads (int): number of attention heads in the deformable attention.
        seq_length (int): The input sequence length.
        num_levels (int, default=4): number of levels of multi-scale features maps.
        num_points (int, default=4): number of sampling points per attention head per level.
        attention_dropout_rate (float, default=0.1): the dropout rate of the attention scores.
        hidden_dropout_rate (float, default=0.1): the dropout rate of the final output of this layer.
        activation (str, default="relu"): the non-linear activation function in the ffn. Supports 'relu', 'gelu', 'elu', 'sigmoid', 
            and so on.
        layernorm_compute_type (dtype.Number, default=dtype.float32): the computation type of the layernorm. Should be dtype.float32 
            or dtype.float16.
        softmax_compute_type (dtype.Number, default=dtype.float32): the computation type of the softmax. Should be dtype.float32 
            or dtype.float16.
        param_init_type (dtype.Number, default=dtype.float32): the paramter initialization type of this module. Should be dtype.float32 
            or dtype.float16.
        parallel_config(TransformerOpParallelConfig): The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.
    """
    @_LogActionOnce(logger=logger, key='DeformableTransformerEncoderLayer',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(hidden_size=Validator.check_positive_int,
                                batch_size=Validator.check_positive_int,
                                num_heads=Validator.check_positive_int,
                                ffn_hidden_size=Validator.check_positive_int,
                                #seq_length=Validator.check_positive_int,
                                attention_dropout_rate=Validator.check_non_negative_float,
                                hidden_dropout_rate=Validator.check_non_negative_float,
                                layernorm_compute_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                           "DeformableTransformerEncoderLayer"),
                                softmax_compute_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                         "DeformableTransformerEncoderLayer"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                    "DeformableTransformerEncoderLayer"),
                                parallel_config=_valid_type_checks([OpParallelConfig, MoEParallelConfig],
                                                                   "DeformableTransformerEncoderLayer"))
    def __init__(self, batch_size: int,
                 hidden_size: int, 
                 ffn_hidden_size: int,
                 num_heads: int,
                 seq_length: Optional[int] = None,
                 num_levels: int = 4,
                 num_points: int = 4,
                 attention_dropout_rate: float = 0.0,
                 hidden_dropout_rate: float = 0.0, 
                 ffn_dropout_rate: float = 0.0,
                 activation: str = "relu", 
                 layernorm_compute_type: mstype.number = mstype.float32,
                 softmax_compute_type: mstype.number = mstype.float32,
                 param_init_type: mstype.number = mstype.float32,
                 compute_dtype: mstype.number = mstype.float32,
                 parallel_config = default_dpmp_config) -> None:
        super(DeformableTransformerEncoderLayer, self).__init__()
        if batch_size:
            Validator.check_positive_int(batch_size)
        self.batch_size = batch_size
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            _check_config(parallel_config)
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'num_heads' must be divisibled by the "
                    "'parallel_config.model_parallel', but got the num_heads is {} and "
                    "parallel_config.model_parallel is {}.".format(num_heads, parallel_config.model_parallel))
            if hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'hidden_size' must be divisibled by "
                    "the 'parallel_config.model_parallel', but got the hidden_size is {} and parallel_config."
                    " model_parallel is {}.".format(hidden_size, parallel_config.model_parallel))
            if ffn_hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'ffn_hidden_size' must be divisibled "
                    "by the 'parallel_config.model_parallel', but got the ffn_hidden_size is {} "
                    "and parallel_config. model_parallel is {}."
                    .format(ffn_hidden_size, parallel_config.model_parallel))
            
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.layernorm1 = _LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
        self.layernorm2 = _LayerNorm((hidden_size,)).to_float(layernorm_compute_type)

        attention_parallel_config = parallel_config
        self.self_attn = MultiScaleDeformableAttention(batch_size=batch_size,
                                                        src_seq_length=seq_length,
                                                        tgt_seq_length=seq_length,
                                                        hidden_size=hidden_size,
                                                        num_heads=num_heads,
                                                        num_levels = num_levels,
                                                        num_points = num_points,
                                                        hidden_dropout_rate=hidden_dropout_rate,
                                                        attention_dropout_rate=attention_dropout_rate,
                                                        compute_dtype=compute_dtype,
                                                        softmax_compute_type=softmax_compute_type,
                                                        param_init_type=param_init_type,
                                                        parallel_config=attention_parallel_config)
        
        self.dtype = compute_dtype
        self.dropout1 = nn.Dropout(1- ffn_dropout_rate)
        self.linear1 = nn.Dense(hidden_size, ffn_hidden_size, weight_init=initializer.Uniform(1/hidden_size), bias_init = 'zeros')
        self.activation = nn.get_activation(activation)
        self.dropout2 = nn.Dropout(1- ffn_dropout_rate)
        self.linear2 = nn.Dense(ffn_hidden_size, hidden_size, weight_init=initializer.Uniform(1/ffn_hidden_size), bias_init = 'zeros')
        self.dropout3 = nn.Dropout(1- ffn_dropout_rate)


    @staticmethod
    def with_pos_embed(tensor: Tensor, 
                       pos: Tensor) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src: Tensor) -> Tensor:
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.layernorm2(src)
        return src

    def construct(self, src: Tensor, 
                  pos: Tensor, 
                  reference_points: Tensor, 
                  spatial_shapes: List[Tuple[int, int]], 
                  level_start_index: List, 
                  padding_mask: List = None) -> Tensor:
        """
        Args: 
            src (Tensor): input tensor of shape (batch_size, seq_length, hidden_size).
            pos (Tensor): positional encoding tensor of shape (batch_size, seq_length, hidden_size).
            reference_points (Tensor): tensor of shape (batch_size, num_levels, num_points, 2) or (batch_size, num_levels, num_points, 4), 
                representing the reference points locations.
            spatial_shapes (list of tuples): a list of tuples, where each tuple represents the spatial shape of the image 
                at a certain level in the deformable attention.
            level_start_index (list of integers): a list of integers representing the starting index of each level in the 
                deformable attention.
            padding_mask (Tensor, optional): tensor representing the padding mask to be applied to the input tensor src. 
                If not provided, no mask will be applied.
        """
        # self.check_input
        src2 = self.self_attn(src, query_pos = pos, key_padding_mask = padding_mask, reference_points=reference_points, value=src, 
                              spatial_shapes=spatial_shapes, 
                              level_start_index=level_start_index)
        src = src + self.dropout1(src2)
        src = self.layernorm1(src)

        src = self.forward_ffn(src)
        return src

class DeformableTransformerEncoder(nn.Cell):
    """
    Deformable Transformer Encoder composed of multiple Deformable Transformer Encoder Layers

    Args:
        encoder_layer (DeformableTransformerEncoderLayer): An instance of DeformableTransformerEncoderLayer
            to be used as the encoder layer for building the Deformable Transformer Encoder.
        num_layers (int): The number of layers to be used for the Deformable Transformer Encoder.
    """

    def __init__(self, 
                 encoder_layer: DeformableTransformerEncoderLayer, 
                 num_layers: int) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes: List[Tuple[int, int]], 
                             valid_ratios: Tensor) -> Tensor:
        """
        Computes reference points for each spatial location.

        Args:
            spatial_shapes (List[Tuple[int, int]]): The shapes of the features maps for all the levels.
            valid_ratios (Tensor): The valid ratios of each feature levels computed from the masks.
        
        Returns:
            reference_points (Tensor): The computed reference points.
        """
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # Create a grid of y and x coordinates for each position in the feature map
            ref_y, ref_x = ops.meshgrid( (ops.linspace(ms.Tensor(0.5), ms.Tensor(H_ - 0.5), H_),
                ops.linspace(ms.Tensor(0.5), ms.Tensor(W_ - 0.5), W_))
            )
            # Normalize the coordinates
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = ops.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = ops.concat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def construct(self, 
                  src: Tensor, 
                  spatial_shapes: List[Tuple[int, int]], 
                  level_start_index: List[int],
                  valid_ratios: Tensor, 
                  pos: Optional[Tensor] = None, 
                  padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            src (Tensor): The input tensor.
            spatial_shapes (List[Tuple[int, int]]): The shapes of the features maps for all the levels.
            level_start_index (List[int]): The start indices for all the levels in a flattened feature map.
            valid_ratios (Tensor): The valid ratios of each feature levels computed from the masks.
            pos (Optional[Tensor]): The positional encoding tensor for the input tensor.
            padding_mask (Optional[Tensor]): A tensor representing padding masks,
                with 1 indicating padding positions and 0 indicating non-padding positions.
        
        Returns:
            Tensor: The encoded output tensor.
        """
        
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios)
        #reference_points = P.cast(reference_points, self.dtype)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output

def run_encoder(batch_size, hidden_size, num_levels, num_channels, spatial_shapes, level_start_index):
    """
    for debugging only
    """
    import numpy as np
    def get_valid_ratio(mask):
        _, H, W = mask.shape
        valid_H = np.sum(~mask[:, :, 0], axis=1)
        valid_W = np.sum(~mask[:, 0, :], axis=1)
        valid_ratio_h = valid_H.astype(np.float32) / H
        valid_ratio_w = valid_W.astype(np.float32) / W
        valid_ratio = np.stack([valid_ratio_w, valid_ratio_h], axis=-1)
        return valid_ratio
    srcs = []
    masks = []
    pos_embeds = []
    
    for i_level in range(num_levels):
        src = np.random.random((batch_size, num_channels[i_level], *spatial_shapes[i_level]))
        mask = np.zeros((batch_size, *spatial_shapes[i_level])).astype('bool')
        pos_embed = np.random.random((batch_size, num_channels[i_level], *spatial_shapes[i_level]))
        srcs.append(src)
        pos_embeds.append(pos_embed)
        masks.append(mask)
    level_embed = np.random.random((num_levels, hidden_size))
    # prepare data for encoder layer
    src_flatten = []
    mask_flatten = []
    lvl_pos_embed_flatten = []

    for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
        bs, c, h, w = src.shape
        # src = src.flatten(2).transpose(1, 2)
        # mask = mask.flatten(1)
        # pos_embed = pos_embed.flatten(2).transpose(1, 2)
        src = np.transpose(src.reshape((src.shape[0], src.shape[1], -1)), (0, 2, 1))
        mask = mask.reshape(mask.shape[0], -1)
        pos_embed = np.transpose(pos_embed.reshape((pos_embed.shape[0], pos_embed.shape[1], -1)), (0, 2, 1))
        lvl_pos_embed = pos_embed + level_embed[lvl].reshape((1, 1, -1))
        lvl_pos_embed_flatten.append(lvl_pos_embed)
        src_flatten.append(src)
        mask_flatten.append(mask)
    src_flatten = ms.Tensor(np.concatenate(src_flatten, 1), mstype.float32)
    mask_flatten = ms.Tensor(np.concatenate(mask_flatten, 1), mstype.bool_)
    lvl_pos_embed_flatten = ms.Tensor(np.concatenate(lvl_pos_embed_flatten, 1), mstype.float32)
    level_start_index = ms.Tensor(np.concatenate((np.zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])), mstype.int32)
    spatial_shapes = spatial_shapes.tolist()
    valid_ratios = ms.Tensor(np.stack([get_valid_ratio(m) for m in masks], 1), mstype.float32)

    encoder_layer = DeformableTransformerEncoderLayer(batch_size, hidden_size, ffn_hidden_size=1024,
                                                      seq_length=src_flatten.shape[1], num_heads = 4, num_levels=3, num_points=4)
    encoder = DeformableTransformerEncoder(encoder_layer, num_layers = 3)
    output = encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
    
    return output, [src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten]

if __name__ == "__main__":
    import numpy as np
    import mindspore.context as context
    # context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    batch_size = 2
    hidden_size = 1024
    num_levels = 3
    num_channels = np.array([1024, 1024, 1024])
    spatial_shapes = np.array([[28, 28], [14, 14], [7, 7]])
    level_start_index = np.array([0, 28*28, 28*28+14*14])

    run_encoder(batch_size, hidden_size, num_levels, num_channels, spatial_shapes, level_start_index)
    # On CPU, no error in PYNATIVE and GRAPH mode!
    # On GPU graph mode, it has no error. On GPU PyNative mode, it has the following error:
    # [ERROR] DEVICE(22276,7fd12cffd700,python):2023-03-21-17:45:06.989.961 [mindspore/ccsrc/runtime/pynative/async/async_queue.cc:75] WorkerLoop] Run task failed, error msg:For 'MatMul', encountered an exception: cuBLAS Error: cublasGemmEx failed. Possible reasons: the GPU is occupied by other processes. | Error Number: 8 CUBLAS_STATUS_ARCH_MISMATCH: The function requires a feature absent from the device architecture; usually caused by compute capability lower than 5.0.

    
