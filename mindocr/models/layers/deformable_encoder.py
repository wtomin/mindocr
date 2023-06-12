from .deformable_attention import MultiScaleDeformableAttention
from typing import Optional, List, Tuple
import math
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor
import mindspore.common.dtype as mstype

from mindspore.common import initializer 
try:
    from mindspore._checkparam import Validator as validator #<=2.0.0a1
except ImportError:
    from mindspore import _checkparam as validator
from mindspore import log as logger
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.context import ParallelMode
from mindspore.log import _LogActionOnce
try:
    from mindspore.parallel._transformer.layers import _args_type_validator_check, _valid_type_checks, _valid_value_checks
    from mindspore.parallel._transformer.op_parallel_config import default_dpmp_config, MoEParallelConfig, OpParallelConfig, _check_config
except ImportError:
    from mindspore.nn.transformer.layers import _args_type_validator_check, _valid_type_checks, _valid_value_checks
    from mindspore.nn.transformer.op_parallel_config import default_dpmp_config, MoEParallelConfig, OpParallelConfig, _check_config

from mindocr.utils.misc import inverse_sigmoid, _get_clones

class DeformableTransformerEncoderLayer(nn.Cell):
    """
    This ia an implementation of Deformable Transformer Encoder Layer. It is composed of a MultiScale Deformable Attention, 
    a LayerNorm, and a feedforward network.
    Args:
        hidden_size (int): number of features in the input.
        ffn_hidden_size (int): hidden size of the feedforward network (ffn) layers.
        num_heads (int): number of attention heads in the deformable attention.
        num_levels (int, default=4): number of levels of multi-scale features maps.
        num_points (int, default=4): number of sampling points per attention head per level.
        attention_dropout_rate (float, default=0.1): the dropout rate of the attention scores.
        activation (str, default="relu"): the non-linear activation function in the ffn. Supports 'relu', 'gelu', 'elu', 'sigmoid', 
            and so on.
        softmax_compute_type (dtype.Number, default=dtype.float32): the computation type of the softmax. Should be dtype.float32 
            or dtype.float16.
        param_init_type (dtype.Number, default=dtype.float32): the paramter initialization type of this module. Should be dtype.float32 
            or dtype.float16.
        parallel_config(TransformerOpParallelConfig): The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.
    """
    @_LogActionOnce(logger=logger, key='DeformableTransformerEncoderLayer',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(hidden_size=validator.check_positive_int,
                                num_heads=validator.check_positive_int,
                                ffn_hidden_size=validator.check_positive_int,
                                attention_dropout_rate=validator.check_non_negative_float,
                                softmax_compute_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                         "DeformableTransformerEncoderLayer"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                    "DeformableTransformerEncoderLayer"),
                                parallel_config=_valid_type_checks([OpParallelConfig, MoEParallelConfig],
                                                                   "DeformableTransformerEncoderLayer"))
    def __init__(self, 
                 hidden_size: int, 
                 ffn_hidden_size: int,
                 num_heads: int,
                 num_levels: int = 4,
                 num_points: int = 4,
                 attention_dropout_rate: float = 0.0,
                 ffn_dropout_rate: float = 0.0,
                 activation: str = "relu", 
                 softmax_compute_type: mstype.number = mstype.float32,
                 param_init_type: mstype.number = mstype.float32,
                 parallel_config = default_dpmp_config) -> None:
        super(DeformableTransformerEncoderLayer, self).__init__()

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
    

        self.hidden_size = hidden_size
        self.layernorm1 = nn.LayerNorm((hidden_size,))
        self.layernorm2 = nn.LayerNorm((hidden_size,))

        attention_parallel_config = parallel_config
        self.self_attn = MultiScaleDeformableAttention( hidden_size=hidden_size,
                                                        num_heads=num_heads,
                                                        num_levels = num_levels,
                                                        num_points = num_points,
                                                        attention_dropout_rate=attention_dropout_rate,
                                                        softmax_compute_type=softmax_compute_type,
                                                        param_init_type=param_init_type,
                                                        parallel_config=attention_parallel_config)
        
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
        src2 = self.self_attn(self.with_pos_embed(src, pos), key = None, value=src, 
                              #ValueError: For 'Add', x.shape and y.shape need to broadcast. 
                              #The value of x.shape[-2] or y.shape[-2] must be 1 or -1 when they are not the same,
                              #but got x.shape = [2, 8500, 256] and y.shape = [2, 7268, 256].
                              value_padding_mask = padding_mask, 
                              reference_points=reference_points, 
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
        lvl = 0
        for shapes in spatial_shapes:
            H, W = shapes
            # Create a grid of y and x coordinates for each position in the feature map as the reference points
            yy = ops.linspace(Tensor(0.5, mstype.float32), ops.cast(H - 0.5, mstype.float32), H)
            xx = ops.linspace(Tensor(0.5, mstype.float32), ops.cast(W - 0.5, mstype.float32), W)
            ref_y, ref_x = mnp.meshgrid(ops.cast(yy, mstype.int32), ops.cast(xx, mstype.int32)) 
            # Normalize the coordinates
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H) #valid_ratios: [bs, w, h]
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = ops.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
            lvl += 1
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
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output
