from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.common import initializer 
from mindocr.models.layers import MultiScaleDeformableAttention, DeformableTransformerEncoderLayer, DeformableTransformerEncoder,\
                                  DeformableCompositeTransformerDecoderLayer, DeformableCompositeTransformerDecoder,\
                                  PositionalEncoding1D, PositionalEncoding2D
                                  
from mindspore.parallel._transformer.op_parallel_config import default_dpmp_config
import copy
from typing import Optional, List, Tuple
import math
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor
import numpy as np
import mindspore as ms
from mindspore import dtype as mstype
from mindspore.common.initializer import initializer,  XavierUniform
from mindocr.utils.misc import MLP

class DeformableTransformer(nn.Cell):
    def __init__(self, 
                 hidden_size: int, 
                 ffn_hidden_size: int,
                 num_levels:int=4, 
                 num_heads:int=8, 
                 return_intermediate_dec=False,
                 dec_num_points=4,  enc_num_points=4, 
                 num_encoder_layers=6, num_decoder_layers=6, 
                 num_proposals=300,
                 dropout_rate: float = 0.0,
                 attention_dropout_rate: float = 0.0,
                 activation: str = "relu", 
                 softmax_compute_type: mstype.number = mstype.float32,
                 param_init_type: mstype.number = mstype.float32,
                 parallel_config = default_dpmp_config):
        
        super().__init__()
        self.num_proposals = num_proposals
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        encoder_layer = DeformableTransformerEncoderLayer(hidden_size, ffn_hidden_size, num_heads, num_levels, 
                                                          enc_num_points, attention_dropout_rate, ffn_dropout_rate=dropout_rate,
                                                          activation = activation, 
                                                          softmax_compute_type=softmax_compute_type,
                                                          param_init_type=param_init_type,
                                                          parallel_config=parallel_config)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        

        decoder_layer = DeformableCompositeTransformerDecoderLayer(hidden_size, ffn_hidden_size, num_levels, num_heads, dec_num_points,
                                                                   dropout_rate, attention_dropout_rate, 
                                                                   activation, softmax_compute_type, 
                                                                   param_init_type, 
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
        spatial_shapes_list = spatial_shapes.asnumpy().tolist()
        for lvl, (H, W) in enumerate(spatial_shapes_list):
            mask_flatten = memory_padding_mask[:, cur:(cur + H * W)].reshape(N, H, W, 1)
            valid_H = (~mask_flatten[:, :, 0, 0]).sum(1)
            valid_W = (~mask_flatten[:, 0, :, 0]).sum(1)

            grid_y, grid_x = ops.meshgrid(ops.linspace(Tensor(0, dtype=mstype.float32), Tensor(H - 1, dtype=mstype.float32), H),
                                        ops.linspace(Tensor(0, dtype=mstype.float32), Tensor(W - 1, dtype=mstype.float32), W))
            grid = ops.concat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)],-1 )

            scale = ops.concat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).reshape(N, 1, 1, 2)
            grid = (ops.repeat_elements(grid.unsqueeze(0), N, axis=0) + 0.5) / scale
            wh = ops.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = ops.concat([grid, wh], -1).view(N, -1, 4)
            proposals.append(proposal)
            cur += (H * W)
        output_proposals = ops.concat( proposals, 1)
        output_proposals_valid = ops.logical_and(output_proposals > 0.01, output_proposals < 0.99).all(-1, keep_dims=True)
            
        output_proposals = ops.log(output_proposals / (1 - output_proposals))
        output_proposals = ops.masked_fill(output_proposals, memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = ops.masked_fill(output_proposals, ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = ops.masked_fill(output_memory, memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = ops.masked_fill(output_memory, ~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals
    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 64
        temperature = 10000
        scale = 2 * math.pi
        dim_t = ops.range(Tensor(0, mstype.int32), Tensor(num_pos_feats, mstype.int32), Tensor(1, mstype.int32))
        dim_t = temperature ** (2 * ops.FloorDiv()(dim_t, Tensor(2, proposals.dtype)) / num_pos_feats)
        proposals = ops.sigmoid(proposals) * scale
        # N, L, 4
        pos = proposals[:,:,:, None] / dim_t 
        # N, L, 4, 128
        sin_pos = ops.sin(pos[:, :, :, 0::2])
        cos_pos = ops.cos(pos[:, :, :, 1::2])
        # N, L, 4, 64, 2
        pos = ops.stack((sin_pos, cos_pos), axis=4).reshape((pos.shape[0], pos.shape[1], -1))
        # N, L, 4*128
        return pos

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = (~mask[:, :, 0]).sum(1)
        valid_W = (~mask[:, 0, :]).sum(1)
        valid_ratio_h = valid_H.float() / float(H)
        valid_ratio_w = valid_W.float()  / float(W) 
        valid_ratio = ops.stack([valid_ratio_w, valid_ratio_h], axis=-1)
        return valid_ratio
    def construct(self, srcs, masks, pos_embeds, query_embed, text_embed, text_pos_embed, text_mask=None):
        # prepare data for encoder layer
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src = ops.transpose(src.reshape((bs, c, -1)), (0, 2, 1))
            mask = mask.reshape((bs, -1))
            pos_embed = ops.transpose(pos_embed.reshape((pos_embed.shape[0], pos_embed.shape[1], -1)), (0, 2, 1))
            lvl_pos_embed = pos_embed + self.level_embed[lvl].reshape((1, 1, -1))
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = ops.concat(src_flatten, 1)
        mask_flatten = ops.concat(mask_flatten, 1)
        lvl_pos_embed_flatten = ops.concat(lvl_pos_embed_flatten, 1)
        spatial_shapes = Tensor(mnp.array(spatial_shapes), mstype.float16)
        level_start_index = ops.concat((ops.zeros((1,), mstype.int32), ops.prod(spatial_shapes, 1).cumsum(0).astype(mstype.int32)[:-1]))
        spatial_shapes = spatial_shapes.astype(mstype.int64)
        valid_ratios = ops.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        enc_outputs_class = self.bbox_class_embed(output_memory)
        enc_outputs_coord_unact = self.bbox_embed(output_memory) + output_proposals
        
        topk = self.num_proposals
        _, topk_proposals = ops.TopK(sorted=True)(enc_outputs_class[..., 0], topk)
        topk_proposals = ops.repeat_elements(topk_proposals.unsqueeze(-1), 4, axis=-1)
        topk_coords_unact = ops.GatherD()(enc_outputs_coord_unact, 1, topk_proposals) 
        topk_coords_unact =  ops.stop_gradient(topk_coords_unact)
        reference_points = ops.sigmoid(topk_coords_unact)
        init_reference_out = reference_points
        query_pos = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
        query_embed = ops.repeat_elements(query_embed.unsqueeze(0), bs, axis=0)
        query_pos = ops.repeat_elements(query_pos.unsqueeze(2),  query_embed.shape[2], axis=2)
        text_embed = ops.repeat_elements(text_embed.unsqueeze(0), bs, axis=0)

        # decoder
        hs, hs_text, inter_references = self.decoder(
            query_embed, text_embed, reference_points, memory, spatial_shapes, 
            level_start_index, valid_ratios, query_pos, text_pos_embed, mask_flatten, text_mask
        )

        inter_references_out = inter_references
        return hs, hs_text, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact



class TESTRDeformableTransformer(nn.Cell):
    def __init__(self, 
                 hidden_size: int, 
                 ffn_hidden_size: int,
                 pos_embed_scale: float, 
                 return_intermediate_dec:bool,
                 dec_num_points:int,  
                 enc_num_points:int, 
                 num_encoder_layers:int, 
                 num_decoder_layers:int, 
                 num_proposals: int,
                 in_channels: List[int] = [256, 512, 1024, 2048],
                 num_levels:int=4, 
                 num_heads:int=8, 
                 num_classes: int = 1,
                 voc_size: int = 96,
                 num_ctrl_points: int = 16,
                 max_text_len: int = 25,
                 dropout_rate: float = 0.0,
                 attention_dropout_rate: float = 0.0,
                 activation: str = "relu", 
                 softmax_compute_type: mstype.number = mstype.float32,
                 param_init_type: mstype.number = mstype.float32,
                 parallel_config = default_dpmp_config):
        super().__init__()
        self.pos_embed_scale = pos_embed_scale
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.voc_size = voc_size
        self.num_ctrl_points = num_ctrl_points
        self.max_text_len = max_text_len
        self.num_levels = num_levels
        self.num_proposals = num_proposals

        self.in_channels = in_channels
        assert len(self.in_channels)>0

        self.text_pos_embed   = PositionalEncoding1D(self.hidden_size, normalize=True, scale=self.pos_embed_scale)
        self.transformer = DeformableTransformer( hidden_size, ffn_hidden_size,num_levels,
                                                 num_heads, return_intermediate_dec, dec_num_points,
                                                 enc_num_points, num_encoder_layers, num_decoder_layers,
                                                 num_proposals, dropout_rate, attention_dropout_rate, 
                                                 activation, softmax_compute_type,
                                                 param_init_type, 
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
        bias_value = -mnp.log((1 - prior_prob) / prior_prob)
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
    
    def construct(self, samples):
        features, pos = samples

        if self.num_levels == 1:
            features = [features[-1]]
            pos = [pos[-1]]

        srcs = []
        masks = []
        for l, feature in enumerate(features):
            src, mask = feature['tensor'], feature['mask']
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_levels > len(srcs):
            pos_encoder = PositionalEncoding2D(self.hidden_size//2, normalize=True)
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1]['tensor'])
                else:
                    src = self.input_proj[l](srcs[-1])
                m = masks[0]
                try:
                    mask = ops.interpolate(m[ None].float(), size=(10,10) , mode='bilinear')[0].bool()
                except:
                    mask = ops.interpolate(m[ None].float(), sizes=(10,10) , mode='bilinear')[0].bool()
                pos_l = pos_encoder(mask)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # n_points, embed_dim --> n_objects, n_points, embed_dim
        ctrl_point_embed = ops.repeat_elements(self.ctrl_point_embed.embedding_table[None, ...], self.num_proposals, 0)
        text_pos_embed = ops.repeat_elements(self.text_pos_embed(self.text_embed.embedding_table)[None, ...], self.num_proposals, 0)
        text_embed = ops.repeat_elements(self.text_embed.embedding_table[None, ...], self.num_proposals, 0)

        hs, hs_text, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
            srcs, masks, pos, ctrl_point_embed, text_embed, text_pos_embed, text_mask=None)
        
        return hs, hs_text, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact
