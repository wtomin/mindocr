from .deformable_attention import MultiScaleDeformableAttention

import copy
from typing import Optional, List, Tuple
import math
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor
import mindspore.common.dtype as mstype 
try:
    from mindspore import _checkparam as validator
except ImportError:
    from mindspore._checkparam import Validator as validator
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.context import ParallelMode
try:
    from mindspore.parallel._transformer.layers import _args_type_validator_check, _valid_type_checks, _valid_value_checks
    from mindspore.parallel._transformer.op_parallel_config import default_dpmp_config, MoEParallelConfig, OpParallelConfig, _check_config
except ImportError:
    from mindspore.nn.transformer.layers import _args_type_validator_check, _valid_type_checks, _valid_value_checks
    from mindspore.nn.transformer.op_parallel_config import default_dpmp_config, MoEParallelConfig, OpParallelConfig, _check_config

from mindocr.utils.misc import inverse_sigmoid, _get_clones
from mindspore.nn.layer.transformer import MultiheadAttention # >=mindspore2.0.0rc1
class DeformableCompositeTransformerDecoderLayer(nn.Cell):
    @_args_type_validator_check(hidden_size=validator.check_positive_int,
                                num_heads=validator.check_positive_int,
                                num_levels=validator.check_positive_int,
                                num_points=validator.check_positive_int,
                                ffn_hidden_size=validator.check_positive_int,
                                attention_dropout_rate=validator.check_non_negative_float,
                                softmax_compute_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                         "DeformableCompositeTransformerDecoderLayer"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                    "DeformableCompositeTransformerDecoderLayer"),
                                parallel_config=_valid_type_checks([OpParallelConfig, MoEParallelConfig],
                                                                   "DeformableCompositeTransformerDecoderLayer"))
    def __init__(self, 
                 hidden_size: int, 
                 ffn_hidden_size: int,
                 num_levels:int=4, 
                 num_heads:int=8, 
                 num_points:int=4,
                 dropout_rate: float = 0.0,
                 attention_dropout_rate: float = 0.0,
                 activation: str = "relu", 
                 softmax_compute_type: mstype.number = mstype.float32,
                 param_init_type: mstype.number = mstype.float32,
                 parallel_config = default_dpmp_config) -> None:
        """
        Args:
        hidden_size (int): hidden size of the model, defaults to 256 
        ffn_hidden_size (int): hidden size of feedforward layer, defaults to 1024
        dropout (float): dropout rate, defaults to 0.1
        activation (str): activation function, defaults to "relu"
        num_levels (int): the number of levels in decoder, defaults to 4 
        num_heads (int): number of attention heads, defaults to 8
        num_points (int): number of sampling points of Deformalbe ATtention, defaults to 4
        """
        super().__init__()
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
        self.ffn_hidden_size = ffn_hidden_size
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        # cross attention
        self.attn_cross = MultiScaleDeformableAttention( hidden_size=hidden_size,
                                               num_heads=num_heads, num_points=num_points, num_levels=num_levels,
                                               attention_dropout_rate=attention_dropout_rate, softmax_compute_type=softmax_compute_type,
                                               param_init_type=param_init_type, parallel_config=parallel_config)
        
        self.dropout_cross = nn.Dropout(keep_prob=1-dropout_rate)
        self.norm_cross = nn.LayerNorm((hidden_size,), epsilon=1e-5)
        
        # self attention (intra)
        
        self.attn_intra = MultiheadAttention(hidden_size, num_heads, dropout=attention_dropout_rate, batch_first=True)
        
        self.dropout_intra = nn.Dropout(keep_prob=1-dropout_rate)
        self.norm_intra = nn.LayerNorm((hidden_size,), epsilon=1e-5)

        # self attention (inter)
        self.attn_inter = MultiheadAttention(hidden_size, num_heads,  dropout=attention_dropout_rate, batch_first=True)
        
        self.dropout_inter = nn.Dropout(keep_prob=1-dropout_rate)
        self.norm_inter = nn.LayerNorm((hidden_size,), epsilon=1e-5)

        # ffn
        self.linear1 = nn.Dense(hidden_size, ffn_hidden_size, activation=activation)
        self.dropout3 = nn.Dropout(keep_prob=1-dropout_rate)
        self.linear2 = nn.Dense(ffn_hidden_size, hidden_size)
        self.dropout4 = nn.Dropout(keep_prob=1-dropout_rate)
        self.norm3 = nn.LayerNorm((hidden_size,), epsilon=1e-5)

        ## (factorized) attn for text branch
        ## TODO: different embedding dim for text/loc?
        # attention between text embeddings belonging to the same object query
        self.attn_intra_text = MultiheadAttention(hidden_size, num_heads,  dropout=attention_dropout_rate, batch_first=True)
        self.dropout_intra_text = nn.Dropout(keep_prob=1-dropout_rate)
        self.norm_intra_text = nn.LayerNorm((hidden_size,), epsilon=1e-5)

        # attention between text embeddings on the same spatial position of different objects
        self.attn_inter_text = MultiheadAttention(hidden_size, num_heads,  dropout=attention_dropout_rate, batch_first=True)
        self.dropout_inter_text = nn.Dropout(keep_prob=1-dropout_rate)
        self.norm_inter_text = nn.LayerNorm((hidden_size,), epsilon=1e-5)

        # cross attention for text
        self.attn_cross_text = MultiScaleDeformableAttention( hidden_size=hidden_size,
                                               num_heads=num_heads, num_points=num_points, num_levels=num_levels,
                                               attention_dropout_rate=attention_dropout_rate, softmax_compute_type=softmax_compute_type,
                                               param_init_type=param_init_type, parallel_config=parallel_config)
        self.dropout_cross_text = nn.Dropout(keep_prob=1-dropout_rate)
        self.norm_cross_text = nn.LayerNorm((hidden_size,), epsilon=1e-5)

        # ffn
        self.linear1_text =nn.Dense(hidden_size, ffn_hidden_size, activation=activation)
        self.dropout3_text = nn.Dropout(keep_prob=1-dropout_rate)
        self.linear2_text = nn.Dense(ffn_hidden_size, hidden_size)
        self.dropout4_text = nn.Dropout(keep_prob=1-dropout_rate)
        self.norm3_text = nn.LayerNorm((hidden_size,), epsilon=1e-5)


    @staticmethod
    def with_pos_embed(tensor: Tensor, 
                       pos: Tensor):
        return tensor if pos is None else tensor + pos

    def ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.linear1(tgt)))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def ffn_text(self, tgt):
        tgt2 = self.linear2_text(self.dropout3_text(self.linear1_text(tgt)))
        tgt = tgt + self.dropout4_text(tgt2)
        tgt = self.norm3_text(tgt)
        return tgt

    def construct(self, tgt: Tensor, 
                query_pos: Tensor, 
                tgt_text: Tensor,
                query_pos_text: Tensor,
                reference_points: Tensor, 
                src: Tensor, 
                src_spatial_shapes: List[Tuple[int, int]], 
                level_start_index: Tensor, 
                src_padding_mask: Optional[Tensor] = None,
                text_padding_mask:  Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the TransformerDecoderLayer.

        Args:
            tgt (Tensor): The target sequence. Shape [batch x n_objects x num_points x embed_dim]
            query_pos (Tensor): The positional encoding for tgt. Shape [batch x n_objects x num_points x embed_dim]
            reference_points (Tensor): Reference points for each feature map in src. Shape [batch x num_levels x embed_dim]
            tgt_text (Tensor): The target word sequence. Shape [batch x n_objects x n_words x embed_dim]
            query_pos_text (Tensor): The positional encoding for tgt_text. Shape [batch x n_objects x n_words x embed_dim]

            src (Tensor): The source sequence. Shape [batch x c x h x w]
            src_spatial_shapes (Tensor): The spatial shapes of each feature map in src. Shape [num_levels x 2]
            level_start_index (Tensor): The starting index in reference_points of features from each level of src. Shape [num_levels]
            src_padding_mask (Optional[Tensor]): Padding mask for src. Shape [batch x src_h x src_w]. Default: None.

        Returns:
            Tensor: The output sequence. Shape [seqlen x batch x embed_dim]

        """
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        bs, n_objects, num_points, embed_dim = q.shape
        tgt2 = self.attn_intra(q.reshape((bs * n_objects, num_points, embed_dim)).transpose((1, 0, 2)), 
                               k.reshape((bs * n_objects, num_points, embed_dim)).transpose((1, 0, 2)),  
                               tgt.reshape((bs * n_objects, num_points, embed_dim)).transpose((1, 0, 2)))[0]
        tgt2 = tgt2.transpose((1, 0, 2)).reshape((bs, n_objects, num_points, embed_dim))
        tgt = tgt + self.dropout_intra(tgt2)
        tgt = self.norm_intra(tgt)

        q_inter = k_inter = tgt_inter = ops.swapdims(tgt, 1, 2) # swap n_objects and num_points -> (bs, num_points, n_objects, embed_dim)
        tgt2_inter = self.attn_inter(q_inter.reshape((bs *num_points,  n_objects, embed_dim)).transpose((1, 0, 2)), 
                               k_inter.reshape((bs *num_points,  n_objects, embed_dim)).transpose((1, 0, 2)),  
                               tgt_inter.reshape((bs *num_points,  n_objects, embed_dim)).transpose((1, 0, 2)))[0]
        tgt2_inter = tgt2_inter.transpose((1, 0, 2)).reshape((bs, num_points, n_objects, embed_dim))
        tgt_inter = tgt_inter + self.dropout_inter(tgt2_inter)
        tgt_inter = ops.swapdims(self.norm_inter(tgt_inter), 1, 2) # -> (bs, n_objects, num_points, embed_dim)

        # cross attention
        reference_points_loc = ops.repeat_elements(reference_points.unsqueeze(2), tgt_inter.shape[2], axis=2)
        tgt2 = self.attn_cross(self.with_pos_embed(tgt_inter, query_pos).reshape((bs, n_objects * num_points, embed_dim)),
                               key = None, value = src,
                               value_padding_mask = src_padding_mask, 
                               reference_points= reference_points_loc.reshape((bs, -1, reference_points_loc.shape[3], reference_points_loc.shape[4])),
                               spatial_shapes = src_spatial_shapes, 
                               level_start_index = level_start_index).reshape(tgt_inter.shape)
        tgt_inter = tgt_inter + self.dropout_cross(tgt2)
        tgt = self.norm_cross(tgt_inter)

        # text branch - intra self attn (word-wise)
        q_text = k_text = self.with_pos_embed(tgt_text, query_pos_text)
        bs, n_objects, num_chars, embed_dim = q_text.shape
        tgt2_text = self.attn_intra_text(q_text.reshape((bs * n_objects, num_chars, embed_dim)).transpose((1, 0, 2)), 
                            k_text.reshape((bs * n_objects, num_chars, embed_dim)).transpose((1, 0, 2)),  
                            tgt_text.reshape((bs * n_objects, num_chars, embed_dim)).transpose((1, 0, 2)),
                            text_padding_mask.reshape((bs * n_objects, num_chars, embed_dim)) if text_padding_mask is not None else None
                            )[0]
        tgt2_text = tgt2_text.transpose((1, 0, 2)).reshape((bs, n_objects, num_chars, embed_dim))
        tgt_text = tgt_text + self.dropout_intra_text(tgt2_text)
        tgt_text = self.norm_intra_text(tgt_text)
         
        # text branch - intra self attn (object-wise)
        q_text_inter = k_text_inter = tgt_text_inter = ops.swapdims(tgt_text, 1, 2) # swap n_objects and num_chars -> (bs, num_chars, n_objects, embed_dim)
        tgt2_text_inter = self.attn_inter_text(q_text_inter.reshape((bs * num_chars, n_objects, embed_dim)).transpose((1, 0, 2)), 
                            k_text_inter.reshape((bs * num_chars, n_objects, embed_dim)).transpose((1, 0, 2)),  
                            tgt_text_inter.reshape((bs * num_chars, n_objects, embed_dim)).transpose((1, 0, 2)),
                            text_padding_mask.reshape((bs * num_chars, n_objects, embed_dim)) if text_padding_mask is not None else None
                            )[0]
        tgt2_text_inter = tgt2_text_inter.transpose((1, 0, 2)).reshape(q_text_inter.shape) # -> (bs, num_chars, n_objects, embed_dim)
        tgt_text_inter = tgt_text_inter + self.dropout_inter_text(tgt2_text_inter)
        tgt_text_inter = ops.swapdims(self.norm_inter_text(tgt_text_inter), 1, 2)

        # text branch - cross attention
        reference_points_loc = ops.repeat_elements(reference_points.unsqueeze(2), tgt_text_inter.shape[2], axis=2)
        tgt2_text_cm = self.attn_cross_text(self.with_pos_embed(tgt_text_inter, query_pos_text).reshape((bs, n_objects * num_chars, embed_dim)),
                            key = None, value = src,
                            value_padding_mask = src_padding_mask,
                            reference_points = reference_points_loc.reshape((bs, -1, reference_points_loc.shape[3], reference_points_loc.shape[4])),
                            spatial_shapes = src_spatial_shapes, 
                            level_start_index = level_start_index).reshape(tgt_text_inter.shape)
        tgt_text_inter = tgt_text_inter + self.dropout_cross_text(tgt2_text_cm)
        tgt_text = self.norm_cross_text(tgt_text_inter)

        # ffn
        tgt = self.ffn(tgt)
        tgt_text = self.ffn_text(tgt_text)

        return tgt, tgt_text

class DeformableCompositeTransformerDecoder(nn.Cell):
    def __init__(self, decoder_layer: nn.Cell, 
                 num_layers: int, 
                 return_intermediate: bool = False):
        """
        Deformable Composite Transformer Decoder module. Reference paper: https://arxiv.org/abs/2204.01918

        Args:
            decoder_layer: Instance of the DeformableCompositeTransformerDecoderLayer class
            num_layers (int): Number of decoder layers.
            return_intermediate (bool): If True, returns the intermediate outputs of each layer.

        Returns:
            Tuple of:
                output: Tensor with shape (batch_size, sequence_length, model_dim)
                output_text: Tensor with shape (batch_size, num_queries, input_dim)
                reference_points: Tensor with shape (batch_size, num_queries, 4) or (batch_size, num_queries, 2)

        Inputs:
            tgt (Tensor): Target sequence. Tensor with shape (batch_size, sequence_length, model_dim).
            tgt_text (Tensor): Target text. Tensor with shape (batch_size, num_queries, input_dim).
            reference_points (Tensor): Reference points. Tensor with shape (batch_size, num_queries, 4) or (batch_size, num_queries, 2).
            src (Tensor): Source sequence. Tensor with shape (batch_size, sequence_length, model_dim).
            src_spatial_shapes (List[Tensor]): List of tensors storing the spatial shapes of the feature level. The spatial shapes should be ordered from finest resolution to coarsest resolution. Each tensor must have shape (2,) representing (height, width) of the feature map.
            src_level_start_index (List[int]): List of integers storing the number to start place the source sequence in the feature map.
            src_valid_ratios (Tensor): Ratios of the valid feature regions at each level in the feature pyramid.
            query_pos (Tensor): Query positions for sequence. Tensor with shape (batch_size, num_queries, model_dim).
            query_pos_text (Tensor): Query positions for text. Tensor with shape (batch_size, num_queries, input_dim).
            src_padding_mask (Tensor): Boolean tensor indicating which indices are valid within the source sequence. Tensor with shape (batch_size, sequence_length).
            text_padding_mask (Tensor): Boolean tensor indicating which indices are valid within the text sequence. Tensor with shape (batch_size, num_queries).

        """
        super().__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def construct(self, tgt: Tensor, 
                  tgt_text: Tensor, 
                  reference_points: Tensor, 
                  src: Tensor,
                  src_spatial_shapes: List[Tuple[int, int]], 
                  src_level_start_index: List[int], 
                  src_valid_ratios: Tensor,
                  query_pos: Tensor = None, 
                  query_pos_text: Tensor = None, 
                  src_padding_mask: Tensor = None,
                  text_padding_mask: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward computation of the Deformable Composite Transformer Decoder.

        Args:
            See __init__ docstring for input arguments.

        Returns:
            Tuple of:
                output: Tensor with shape (batch_size, sequence_length, model_dim)
                output_text: Tensor with shape (batch_size, num_queries, input_dim)
                reference_points: Tensor with shape (batch_size, num_queries, 4) or (batch_size, num_queries, 2)

        """
        output, output_text = tgt, tgt_text

        intermediate = []
        intermediate_text = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * ops.concat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            # passing through decoder layer
            output, output_text = layer(output, query_pos, output_text, query_pos_text, reference_points_input, src,
                                         src_spatial_shapes, src_level_start_index, src_padding_mask,
                                         text_padding_mask)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_text.append(output_text)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return ops.stack(intermediate, axis=0), ops.stack(intermediate_text, axis=0), ops.stack(intermediate_reference_points, axis=0)

        return output, output_text, reference_points

class DeformableTransformerDecoderLayer(nn.Cell):
    """
    Deformable Transformer Decoder Layer.

    Args:
        hidden_size (int): The dimension of model.
        ffn_hidden_size (int): The dimension of feed forward network.
        dropout (float): The dropout probability.
        activation (str): The activation function used in feed forward network.
        num_levels (int): The number of deformable levels.
        num_heads (int): The number of heads in multihead attention.
        num_points (int): The number of sampling points in deformable attention.

    Inputs:
        - tgt (Tensor) - The target sequence. Tensor of shape [batch_size, target_length, hidden_size].
        - query_pos (Tensor) - The positional encoding for target sequence. Tensor of shape [batch_size, target_length, hidden_size].
        - reference_points (Tensor) - The reference points for deformable attention. Tensor of shape [batch_size, num_levels, N, 2].
        - src (Tuple) - The source sequence and source masks. Tuple of two tensors:
            - src_tensor (Tensor) - X. Tensor of shape [batch_size, f_channels, src_height, src_width].
            - src_mask (Tensor) - M. Tensor of shape [batch_size, 1, src_height, src_width].
        - src_spatial_shapes (List[Tuple[int]]): The spatial shapes of sources.
        - level_start_index (List[int]): The start index of each level in reference points.
        - src_padding_mask (Tensor): The padding mask for source sequence. Tensor of shape [batch_size, 1, src_height, src_width].

    Outputs:
        Tensor of shape [batch_size, target_length, hidden_size].
    """
    @_args_type_validator_check(hidden_size=validator.check_positive_int,
                                num_heads=validator.check_positive_int,
                                num_levels=validator.check_positive_int,
                                num_points=validator.check_positive_int,
                                ffn_hidden_size=validator.check_positive_int,
                                attention_dropout_rate=validator.check_non_negative_float,
                                softmax_compute_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                         "DeformableTransformerDecoderLayer"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                    "DeformableTransformerDecoderLayer"),
                                parallel_config=_valid_type_checks([OpParallelConfig, MoEParallelConfig],
                                                                   "DeformableTransformerDecoderLayer"))
    def __init__(self, 
                 hidden_size: int, 
                 ffn_hidden_size: int,
                 num_levels:int=4, 
                 num_heads:int=8, 
                 num_points:int=4,
                 dropout_rate: float = 0.0,
                 attention_dropout_rate: float = 0.0,
                 activation: str = "relu", 
                 softmax_compute_type: mstype.number = mstype.float32,
                 param_init_type: mstype.number = mstype.float32,
                 parallel_config = default_dpmp_config) -> None:
        super().__init__()

        # cross attention
        self.cross_attn = MultiScaleDeformableAttention(hidden_size=hidden_size,
                                               num_heads=num_heads, num_points=num_points, num_levels=num_levels,
                                               attention_dropout_rate=attention_dropout_rate, softmax_compute_type=softmax_compute_type,
                                               param_init_type=param_init_type, parallel_config=parallel_config)
        self.dropout1 = nn.Dropout(1 - dropout_rate)
        self.norm1 = nn.LayerNorm((hidden_size,), epsilon=1e-5)

        # self attention
        self.self_attn =  MultiheadAttention(hidden_size, num_heads,  dropout=attention_dropout_rate, batch_first=True)
        self.dropout2 = nn.Dropout(1 - dropout_rate)
        self.norm2 = nn.LayerNorm((hidden_size,), epsilon=1e-5)

        # feed-forward network
        self.linear1 = nn.Dense(hidden_size, ffn_hidden_size, activation = activation)
        self.dropout3 = nn.Dropout(1 - dropout_rate)
        self.linear2 = nn.Dense(ffn_hidden_size, hidden_size)
        self.dropout4 = nn.Dropout(1 - dropout_rate)
        self.norm3 = nn.LayerNorm((hidden_size,), epsilon=1e-5)

    def construct(self, tgt: Tensor, 
                query_pos: Tensor, 
                reference_points: Tensor, 
                src: Tensor, 
                src_spatial_shapes: List[Tuple[int, int]], 
                level_start_index: Tensor, 
                src_padding_mask: Optional[Tensor] = None) -> Tensor:
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        bs, n_objects, num_points, embed_dim = tgt.shape
        tgt2 = self.self_attn(q.reshape((bs * n_objects, num_points, embed_dim)).transpose((1, 0, 2)), 
                               k.reshape((bs * n_objects, num_points, embed_dim)).transpose((1, 0, 2)),  
                               tgt.reshape((bs * n_objects, num_points, embed_dim)).transpose((1, 0, 2)))[0]
        tgt2 = tgt2.transpose((1, 0, 2)).reshape((bs, n_objects, num_points, embed_dim))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), 
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # feed forward network
        tgt2 = self.linear2(self.dropout3(self.linear1(tgt)))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional encoding to the input tensor."""
        return tensor if pos is None else tensor + pos.unsqueeze(-2).expand_as(tensor)

class DeformableTransformerDecoder(nn.Cell):
    """
    Deformable Transformer decoder.

    Args:
        decoder_layer (nn.Cell): Single decoder layer.
        num_layers (int): Number of decoder layers.
        return_intermediate (bool, optional): Whether or not to return intermediate outputs of each decoder layer.

    Attributes:
        layers (nn.layer.CellList): List of decoder layers.
        num_layers (int): Number of decoder layers.
        return_intermediate (bool): Whether or not to return intermediate outputs of each decoder layer.
        bbox_embed (None or nn.Cell): Neural network layer for iterative bounding box refinement.
        class_embed (None or nn.Cell): Neural network layer for class embedding.

    Returns:
        Tensor, Tensor: Intermediate outputs of each decoder layer and intermediate reference points.

    """
    def __init__(self, 
                 decoder_layer: nn.Cell, 
                 num_layers: int, 
                 return_intermediate: bool = False):
        super(DeformableTransformerDecoder, self).__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def construct(self, tgt: Tensor, 
                  reference_points: Tensor, 
                  src: Tensor, 
                  src_spatial_shapes: Tensor,
                  src_level_start_index: Tensor, 
                  src_valid_ratios: Tensor, 
                  query_pos: Tensor = None,
                  src_padding_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the Deformable Transformer decoder.

        Args:
            tgt (Tensor): Target tensor.
            reference_points (Tensor): Tensor of reference points.
            src (Tensor): Source tensor.
            src_spatial_shapes (Tensor): Spatial shapes of the source tensor.
            src_level_start_index (Tensor): Start index of each pyramidal level in the flattened source tensor.
            src_valid_ratios (Tensor): Ratios of the valid regions in the source tensor for each level.
            query_pos (Tensor, optional): Query position tensor. Defaults to None.
            src_padding_mask (Tensor, optional): Padding mask for the source tensor. Defaults to None.

        Returns:
            Tensor, Tensor: Intermediate outputs of each decoder layer and intermediate reference points.

        """
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * ops.concat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = ops.sigmoid(new_reference_points)
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = ops.sigmoid(new_reference_points)
                reference_points = ops.stop_gradient(new_reference_points)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return ops.stack(intermediate, axis=0), ops.stack(intermediate_reference_points, axis=0)

# def run_decoder(decoderlayer_cell, decoder_cell, hidden_size, num_levels, num_channels, spatial_shapes, level_start_index):
#     from deformable_encoder import run_encoder
#     # prepare data for decoder
#     def gen_encoder_output_proposals(memory:Tensor, memory_padding_mask:Tensor, spatial_shapes=None):
#         N, S, C = memory.shape
#         dtype = memory.dtype
#         base_scale = 4.0
#         proposals = []
#         cur = 0
#         for lvl, (H, W) in enumerate(spatial_shapes):
#             mask_flatten = memory_padding_mask[:, cur:(cur + H * W)].reshape(N, H, W, 1)
#             valid_H = (~mask_flatten[:, :, 0, 0]).sum(1)
#             valid_W = (~mask_flatten[:, 0, :, 0]).sum(1)

#             grid_y, grid_x = ops.meshgrid((ops.linspace(Tensor(0, dtype=mstype.float32), Tensor(H - 1, dtype=mstype.float32), H),
#                                         ops.linspace(Tensor(0, dtype=mstype.float32), Tensor(W - 1, dtype=mstype.float32), W)))
#             grid = ops.concat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)],-1 )

#             scale = ops.concat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).reshape(N, 1, 1, 2)
#             grid = (ops.repeat_elements(grid.unsqueeze(0), N, axis=0) + 0.5) / scale
#             wh = ops.ones_like(grid) * 0.05 * (2.0 ** lvl)
#             proposal = ops.concat([grid, wh], -1).view(N, -1, 4)
#             proposals.append(proposal)
#             cur += (H * W)
#         output_proposals = ops.concat( proposals, 1)
#         output_proposals_valid = ops.tile(ops.logical_and(output_proposals > 0.01, output_proposals < 0.99).all(-1).unsqueeze(-1), 
#                                                     (1, 1,output_proposals.shape[-1]))
            
#         output_proposals = ops.log(output_proposals / (1 - output_proposals))
#         output_proposals = ops.MaskedFill()(output_proposals, memory_padding_mask.unsqueeze(-1), float('inf'))
#         output_proposals = ops.MaskedFill()(output_proposals, ~output_proposals_valid, float('inf'))

#         output_memory = memory
#         output_memory = ops.MaskedFill()(output_memory, memory_padding_mask.unsqueeze(-1), float(0))
#         # ignore because they won't change the shapes
#         # output_memory = ops.where(~output_proposals_valid, float(0), output_memory)
#         # output_memory = self.enc_output_norm(self.enc_output(output_memory))
#         return output_memory, output_proposals
#     def get_proposal_pos_embed( proposals):
#         num_pos_feats = 64
#         temperature = 10000
#         scale = 2 * math.pi
#         dim_t = mnp.arange(num_pos_feats, dtype=proposals.dtype)
#         dim_t = temperature ** (2 * ops.FloorDiv()(dim_t, Tensor(2, proposals.dtype)) / num_pos_feats)
#         proposals = ops.sigmoid(proposals) * scale
#         pos = proposals[:,:,:,:, None] / dim_t # size incorrect?
#         sin_pos = mnp.sin(pos[:, :, :, :, 0::2])
#         cos_pos = mnp.cos(pos[:, :, :, :, 1::2])
#         pos = ops.stack((sin_pos, cos_pos), axis=4).reshape((pos.shape[0], pos.shape[1],-1))
#         return pos

#     memory, [src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten] = \
#         run_encoder(hidden_size, num_levels, num_channels, spatial_shapes, level_start_index)
#     bs, _, c = memory.shape
#     output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
#     enc_outputs_class = nn.Dense(hidden_size, 2)(output_memory)
#     enc_outputs_coord_unact = nn.Dense(hidden_size, 4)(output_memory) + output_proposals

#     topk = 300
#     _, topk_proposals = ops.TopK(sorted=True)(enc_outputs_class[..., 0], topk)
#     topk_proposals = ops.repeat_elements(mnp.expand_dims(topk_proposals, axis=-1), 4, axis=2)
#     topk_coords_unact = ops.Gather()(enc_outputs_coord_unact, topk_proposals,1 )
#     topk_coords_unact =  ops.stop_gradient(topk_coords_unact)
#     reference_points = ops.sigmoid(topk_coords_unact)
#     init_reference_out = reference_points
#     #query_pos = nn.LayerNorm((hidden_size,))(nn.Dense(hidden_size, hidden_size)(get_proposal_pos_embed(topk_coords_unact)))
    
#     query_embed = mnp.expand_dims(query_embed, axis=0).repeat(bs, 1, 1, 1)
#     query_pos_shape = query_pos.shape
#     query_pos = mnp.expand_dims(query_pos, axis=2).repeat(1, 1, query_embed.shape[2], 1)
#     query_pos = mnp.reshape(query_pos, (query_pos_shape[0] * query_pos_shape[1], query_pos_shape[2], query_pos_shape[3]))
#     text_embed = mnp.expand_dims(text_embed, axis=0).repeat(bs, 1, 1, 1)

#     # dummy 
#     text_pos_embed = text_embed
#     text_mask = None
#     num_decoder_layers = 3
#     return_intermediate_dec = False
#     decoder_layer = decoderlayer_cell(hidden_size, hidden_size, ffn_hidden_size=1024,
#                                                      
#                                                       num_heads = 4, num_levels=3, num_points=4)
#     decoder = decoder_cell(decoder_layer, num_decoder_layers, return_intermediate_dec)

#     hs, hs_text, inter_references = decoder(
#                 query_embed, text_embed, reference_points, memory, spatial_shapes, 
#                 level_start_index, valid_ratios, query_pos, text_pos_embed, mask_flatten, text_mask
#             )

# if __name__ == "__main__":
#     import numpy as np
#     import mindspore.context as context
#     # context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
#     context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
#     batch_size = 2
#     hidden_size = 1024
#     num_levels = 3
#     num_channels = np.array([1024, 1024, 1024])
#     spatial_shapes = np.array([[28, 28], [14, 14], [7, 7]])
#     level_start_index = np.array([0, 28*28, 28*28+14*14])
    
#     run_decoder(DeformableCompositeTransformerDecoderLayer, DeformableCompositeTransformerDecoder,
#         hidden_size, num_levels, num_channels, spatial_shapes, level_start_index)

#     run_decoder(DeformableTransformerDecoderLayer, DeformableTransformerDecoder,
#         hidden_size, num_levels, num_channels, spatial_shapes, level_start_index)