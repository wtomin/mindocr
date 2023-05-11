from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.common import initializer 
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
import mindspore as ms
from mindspore.nn import Cell
from mindspore import dtype as mstype
from mindspore.common.initializer import initializer, Normal, XavierUniform
from mindocr.utils.misc import NestedTensor, nested_tensor_from_tensor_list, MLP

class DeformableTransformer(nn.Cell):
    def __init__(self, batch_size: int,
                 hidden_size: int, 
                 ffn_hidden_size: int,
                 src_seq_length:Optional[int] = None,
                 tgt_seq_length:Optional[int] = None,
                 num_levels:int=4, 
                 num_heads:int=8, 
                 return_intermediate_dec=False,
                 dec_num_points=4,  enc_num_points=4, 
                 num_encoder_layers=6, num_decoder_layers=6, 
                 num_proposals=300,
                 dropout_rate: float = 0.0,
                 attention_dropout_rate: float = 0.0,
                 hidden_dropout_rate: float = 0.0, 
                 activation: str = "relu", 
                 layernorm_compute_type: mstype.number = mstype.float32,
                 softmax_compute_type: mstype.number = mstype.float32,
                 param_init_type: mstype.number = mstype.float32,
                 compute_dtype: mstype.number = mstype.float32,
                 parallel_config = default_dpmp_config):
        
        super().__init__()
        self.num_proposals = num_proposals
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        encoder_layer = DeformableTransformerEncoderLayer(batch_size, hidden_size, ffn_hidden_size, num_heads, src_seq_length, num_levels, 
                                                          enc_num_points, attention_dropout_rate, hidden_dropout_rate, ffn_dropout_rate=dropout_rate,
                                                          activation = activation, layernorm_compute_type=layernorm_compute_type,
                                                          softmax_compute_type=softmax_compute_type,
                                                          param_init_type=param_init_type,
                                                          compute_dtype=compute_dtype,
                                                          parallel_config=parallel_config)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        

        decoder_layer = DeformableCompositeTransformerDecoderLayer(batch_size, hidden_size, ffn_hidden_size,
                                                                   src_seq_length, tgt_seq_length, num_levels, num_heads, dec_num_points,
                                                                   dropout_rate, attention_dropout_rate, hidden_dropout_rate, 
                                                                   activation, layernorm_compute_type, softmax_compute_type, 
                                                                   param_init_type, compute_dtype,
                                                                   parallel_config)
        self.decoder = DeformableCompositeTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        
        self.level_embed = ms.Parameter(initializer(XavierUniform(gain=1), [num_levels, hidden_size], dtype=param_init_type))

        self.bbox_class_embed = None
        self.bbox_embed = None
        self.enc_output = nn.Dense(hidden_size, hidden_size)
        self.enc_output_norm = nn.LayerNorm((hidden_size,), epsilon=1e-5)
        self.pos_trans = nn.Dense(hidden_size, hidden_size)
        self.pos_trans_norm = nn.LayerNorm((hidden_size,), epsilon=1e-5)
        
        self.param_init_type = param_init_type
        self._reset_parameters()


    def _reset_parameters(self):
        xavier_uniform_init = XavierUniform(gain=1)
        
        for p in self.get_parameters():
            if p.dim() > 1:
                p.set_data(initializer(xavier_uniform_init, p.shape,  dtype=self.param_init_type))
                
        for name, cell in self.cells_and_names():
            if isinstance(cell, MultiScaleDeformableAttention):
                cell._reset_parameters()

    def gen_encoder_output_proposals(self, memory:Tensor, memory_padding_mask:Tensor, spatial_shapes=None):
        N, S, C = memory.shape
        base_scale = 4.0
        proposals = []
        cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten = memory_padding_mask[:, cur:(cur + H * W)].reshape(N, H, W, 1)
            valid_H = (~mask_flatten[:, :, 0, 0]).sum(1)
            valid_W = (~mask_flatten[:, 0, :, 0]).sum(1)

            grid_y, grid_x = ops.meshgrid((ops.linspace(Tensor(0, dtype=mstype.float32), Tensor(H - 1, dtype=mstype.float32), H),
                                        ops.linspace(Tensor(0, dtype=mstype.float32), Tensor(W - 1, dtype=mstype.float32), W)))
            grid = ops.concat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)],-1 )

            scale = ops.concat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).reshape(N, 1, 1, 2)
            grid = (ops.repeat_elements(grid.unsqueeze(0), N, axis=0) + 0.5) / scale
            wh = ops.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = ops.concat([grid, wh], -1).view(N, -1, 4)
            proposals.append(proposal)
            cur += (H * W)
        output_proposals = ops.concat( proposals, 1)
        output_proposals_valid = ops.tile(ops.logical_and(output_proposals > 0.01, output_proposals < 0.99).all(-1).unsqueeze(-1), 
                                                    (1, 1,output_proposals.shape[-1]))
            
        output_proposals = ops.log(output_proposals / (1 - output_proposals))
        output_proposals = ops.MaskedFill()(output_proposals, memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = ops.MaskedFill()(output_proposals, ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = ops.MaskedFill()(output_memory, memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = ops.MaskedFill()(output_memory, ~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals
    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 64
        temperature = 10000
        scale = 2 * math.pi
        dim_t = mnp.arange(num_pos_feats, dtype=proposals.dtype)
        dim_t = temperature ** (2 * ops.FloorDiv()(dim_t, Tensor(2, proposals.dtype)) / num_pos_feats)
        proposals = ops.sigmoid(proposals) * scale
        pos = proposals[:,:,:,:, None] / dim_t # size incorrect?
        sin_pos = mnp.sin(pos[:, :, :, :, 0::2])
        cos_pos = mnp.cos(pos[:, :, :, :, 1::2])
        pos = ops.stack((sin_pos, cos_pos), axis=4).reshape((pos.shape[0], pos.shape[1],-1))
        return pos

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = ops.sum(~mask[:, :, 0], axis=1)
        valid_W = ops.sum(~mask[:, 0, :], axis=1)
        valid_ratio_h = valid_H / H
        valid_ratio_w = valid_W / W
        valid_ratio = ops.stack([valid_ratio_w, valid_ratio_h], axis=-1)
        return valid_ratio
    def construct(self, srcs, masks, pos_embeds, query_embed, text_embed, text_pos_embed, text_mask=None):
        # prepare data for encoder layer
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []

        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            src = ops.transpose(src.reshape((src.shape[0], src.shape[1], -1)), (0, 2, 1))
            mask = mask.reshape(mask.shape[0], -1)
            pos_embed = ops.transpose(pos_embed.reshape((pos_embed.shape[0], pos_embed.shape[1], -1)), (0, 2, 1))
            lvl_pos_embed = pos_embed + self.level_embed[lvl].reshape((1, 1, -1))
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = Tensor(ops.concat(src_flatten, 1), mstype.float32)
        mask_flatten = Tensor(ops.concat(mask_flatten, 1), mstype.bool_)
        lvl_pos_embed_flatten = Tensor(ops.concat(lvl_pos_embed_flatten, 1), mstype.float32)
        level_start_index = Tensor(ops.concat((np.zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])), mstype.int32)
        spatial_shapes = spatial_shapes.asnumpy().tolist()
        valid_ratios = Tensor(ops.stack([self.get_valid_ratio(m) for m in masks], 1), mstype.float32)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        enc_outputs_class = self.bbox_class_embed(output_memory)
        enc_outputs_coord_unact = self.bbox_embed(output_memory) + output_proposals

        topk = 300
        _, topk_proposals = ops.TopK(sorted=True)(enc_outputs_class[..., 0], topk)
        topk_proposals = ops.repeat_elements(mnp.expand_dims(topk_proposals, axis=-1), 4, axis=2)
        topk_coords_unact = ops.Gather()(enc_outputs_coord_unact, topk_proposals,1 )
        topk_coords_unact =  ops.stop_gradient(topk_coords_unact)
        reference_points = ops.sigmoid(topk_coords_unact)
        init_reference_out = reference_points
        query_pos = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
        
        query_embed = mnp.expand_dims(query_embed, axis=0).repeat(bs, 1, 1, 1)
        query_pos_shape = query_pos.shape
        query_pos = mnp.expand_dims(query_pos, axis=2).repeat(1, 1, query_embed.shape[2], 1)
        query_pos = mnp.reshape(query_pos, (query_pos_shape[0] * query_pos_shape[1], query_pos_shape[2], query_pos_shape[3]))
        text_embed = mnp.expand_dims(text_embed, axis=0).repeat(bs, 1, 1, 1)

        # decoder
        hs, hs_text, inter_references = self.decoder(
            query_embed, text_embed, reference_points, memory, spatial_shapes, 
            level_start_index, valid_ratios, query_pos, text_pos_embed, mask_flatten, text_mask
        )

        inter_references_out = inter_references
        return hs, hs_text, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact



class TESTRDeformableTransformer(nn.Cell):
    def __init__(self, batch_size: int,
                 hidden_size: int, 
                 ffn_hidden_size: int,
                 pos_embed_scale: float, 
                 return_intermediate_dec:bool,
                 dec_num_points:int,  
                 enc_num_points:int, 
                 num_encoder_layers:int, 
                 num_decoder_layers:int, 
                 num_proposals: int,
                 src_seq_length: Optional[int] = None,
                 tgt_seq_length: Optional[int] = None,
                 in_channels: List[int] = [256, 512, 1024, 2048],
                 num_levels:int=4, 
                 num_heads:int=8, 
                 num_classes: int = 1,
                 voc_size: int = 96,
                 num_ctrl_points: int = 16,
                 max_text_len: int = 25,
                 dropout_rate: float = 0.0,
                 attention_dropout_rate: float = 0.0,
                 hidden_dropout_rate: float = 0.0, 
                 activation: str = "relu", 
                 layernorm_compute_type: mstype.number = mstype.float32,
                 softmax_compute_type: mstype.number = mstype.float32,
                 param_init_type: mstype.number = mstype.float32,
                 compute_dtype: mstype.number = mstype.float32,
                 parallel_config = default_dpmp_config):
        super().__init__()

        self.pos_embed_scale = pos_embed_scale
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.voc_size = voc_size
        self.num_ctrl_points = num_ctrl_points
        self.max_text_len = max_text_len
        self.num_levels = num_levels

        self.in_channels = in_channels
        assert len(self.in_channels)>0

        self.text_pos_embed   = PositionalEncoding1D(self.hidden_size, normalize=True, scale=self.pos_embed_scale)
        self.transformer = DeformableTransformer(batch_size, hidden_size, ffn_hidden_size,
                                                 src_seq_length, tgt_seq_length, num_levels,
                                                 num_heads, return_intermediate_dec, dec_num_points,
                                                 enc_num_points, num_encoder_layers, num_decoder_layers,
                                                 num_proposals, dropout_rate, attention_dropout_rate, hidden_dropout_rate,
                                                 activation, layernorm_compute_type, softmax_compute_type,
                                                 param_init_type, compute_dtype,
                                                 parallel_config)
        
        self.bbox_coord = MLP(self.hidden_size, self.hidden_size, output_dim = 4, num_layers = 3) 
        self.bbox_class = nn.Dense(self.hidden_size, self.num_classes)

        self.ctrl_point_embed = nn.Embedding(self.num_ctrl_points, self.hidden_size)
        self.text_embed = nn.Embedding(self.max_text_len, self.hidden_size)


        if num_levels >1:
            # use three levels by default
            strides = [8, 16, 32]
            num_channels = [512, 1024, 2048]
            num_backbone_outs = len(strides)
            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channel = num_channels[i]
                input_proj_list.append(nn.SequentialCell(
                    [nn.Conv2d(in_channel, hidden_size, kernel_size=1, pad_mode='valid', has_bias=True),
                    nn.GroupNorm(32, hidden_size)]))
                    
            for j in range(num_levels - num_backbone_outs):
                input_proj_list.append(nn.SequentialCell(
                    [nn.Conv2d(in_channel, hidden_size, kernel_size=3, stride=2, padding=1, pad_mode='pad', has_bias=True),
                    nn.GroupNorm(32, hidden_size)]))
                in_channel = hidden_size

            self.input_proj = nn.CellList(input_proj_list)
        else:
            # use single level
            strides = [32]
            num_channels = [2048]
            self.input_proj = nn.SequentialCell(
                [nn.Conv2d(num_channels[0], hidden_size, kernel_size=1, pad_mode='valid', has_bias=True),
                nn.GroupNorm(32, hidden_size)])
        
        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.bbox_class.bias.set_data(initializer('ones', [num_classes]) * bias_value, param_init_type)
        self.bbox_class.weight.set_data(initializer('zeros', self.bbox_class.weight.shape, param_init_type))

        self.transformer.decoder.bbox_embed = None
        for proj in self.input_proj:
            proj[0].weight.set_data(initializer(XavierUniform(1), proj[0].weight.shape, param_init_type))
            proj[0].bias.set_data(initializer('zeros', proj[0].bias.shape, param_init_type))

        self.bbox_coord.layers[-1].bias.init_data()[2:] = Tensor(initializer('zeros', self.bbox_coord.layers[-1].bias[2:].shape, param_init_type))
        self.transformer.bbox_class_embed = self.bbox_class
        self.transformer.bbox_embed = self.bbox_coord

        self.out_channels = self.hidden_size # it is for argument setting, not used by the TESTRHead
    
    def forward(self, samples):
        """ samples is NestedTensor
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        # features, pos = self.backbone(samples)

        if self.num_levels == 1:
            features = [features[-1]]
            pos = [pos[-1]]

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        assert self.num_levels == len(srcs), "length does not match!"
        # if self.num_levels > len(srcs):
        #     _len_srcs = len(srcs)
        #     for l in range(_len_srcs, self.num_levels):
        #         if l == _len_srcs:
        #             src = self.input_proj[l](features[-1].tensors)
        #         else:
        #             src = self.input_proj[l](srcs[-1])
        #         m = masks[0]
        #         mask = P.cast(F.interpolate( m[None].float(), size=src.shape[-2:]), dtype=mstype.bool_)[0]
        #         pos_l = P.cast(self.backbone[1](NestedTensor(src, mask)), dtype=src.dtype)
        #         srcs.append(src)
        #         masks.append(mask)
        #         pos.append(pos_l)

        # n_points, embed_dim --> n_objects, n_points, embed_dim
        ctrl_point_embed = self.ctrl_point_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)
        text_pos_embed = self.text_pos_embed(self.text_embed.weight)[None, ...].repeat(self.num_proposals, 1, 1)
        text_embed = self.text_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)

        hs, hs_text, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
            srcs, masks, pos, ctrl_point_embed, text_embed, text_pos_embed, text_mask=None)
        
        return hs, hs_text, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact
