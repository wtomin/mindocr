from typing import List, Optional, Type, Union

import mindspore.common.initializer as init
from mindspore import Tensor, Parameter, nn, ops 
import numpy as np
import mindspore as ms
import mindspore.numpy as mnp

class PositionalEncoding(nn.Cell):
    def __init__(self, d_hid, n_position=200):
        super().__init__()
        pos_table = self._get_sinusoid_encoding_table(n_position, d_hid)
        self.pos_table = Parameter(Tensor(pos_table, ms.float32), requires_grad=False) # do not update
        self.add = ops.Add()

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return np.expand_dims(sinusoid_table, axis=0).astype(np.float32)

    def construct(self, x):
        return self.add(x, self.pos_table[:, :x.shape[1], :])


class ScaledDotProductAttention(nn.Cell):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(p=attn_dropout)
        self.softmax = nn.Softmax(axis=2)
        self.bmm = ops.BatchMatMul()
        self.masked_fill = ops.MaskedFill()

    def construct(self, q, k, v, mask=None):
        attn = self.bmm(q, k.transpose((0, 2, 1)))
        attn = attn / self.temperature
        if mask is not None:
            attn = self.masked_fill(attn, mask, -1e9)
        attn = self.softmax(attn)     
        attn = self.dropout(attn)
        output =self.bmm(attn, v)
        return output, attn

class PositionwiseFeedForward(nn.Cell):
    ''' A two-feed-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, kernel_size=1) # position-wise
        self.layer_norm = nn.LayerNorm((d_in, ))
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def construct(self, x):
        residual = x
        x = x.transpose((0, 2, 1))
        x = self.w_2(self.relu(self.w_1(x)))
        x = x.transpose((0, 2, 1))
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x
    
class MultiHeadAttention(nn.Cell):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Dense(d_model, n_head * d_k)
        self.w_ks = nn.Dense(d_model, n_head * d_k)
        self.w_vs = nn.Dense(d_model, n_head * d_v)
        self.w_qs.weight.set_data(init.initializer(init.Normal(sigma=np.sqrt(2.0 / (d_model + d_k))), 
                                                   self.w_qs.weight.shape))
        self.w_ks.weight.set_data(init.initializer(init.Normal(sigma=np.sqrt(2.0 / (d_model + d_k))),
                                                   self.w_ks.weight.shape))
        self.w_vs.weight.set_data(init.initializer(init.Normal(sigma=np.sqrt(2.0 / (d_model + d_v))),
                                                   self.w_vs.weight.shape))
        # here
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm((d_model,))
        self.fc = nn.Dense(n_head * d_v, d_model)
        self.fc.weight.set_data(init.initializer(init.XavierUniform(), self.fc.weight.shape))
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.shape
        sz_b, len_k, _ = k.shape
        sz_b, len_v, _ = v.shape
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)     # 4*21*512 ---- 4*21*8*64
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.transpose((2, 0, 1, 3)).view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.transpose((2, 0, 1, 3)).view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.transpose((2, 0, 1, 3)).view(-1, len_v, d_v) # (n*b) x lv x dv
        mask = ops.stack([mask]*n_head, 0) if mask is not None else None # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.transpose((1, 2, 0, 3)).view(sz_b, len_q, -1) # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn

class EncoderLayer(nn.Cell):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def construct(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.self_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
    
class TransformerEncoder(nn.Cell):
    def __init__(self, d_word_vec=512, n_layers=2, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner=2048, dropout=0.1, n_position=256):
        super().__init__()
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.CellList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm((d_model,), epsilon=1e-6)

    def construct(self, enc_output, src_mask, return_attns=False):
        enc_output = self.dropout(self.position_enc(enc_output))   # position embedding
        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(enc_output, slf_attn_mask=src_mask)
        enc_output = self.layer_norm(enc_output)
        return enc_output


class PP_Layer(nn.Cell):
    def __init__(self, n_dim=512, N_max_character=25, n_position=256):
        super().__init__()
        self.character_len = N_max_character
        self.f0_embedding = nn.Embedding(N_max_character, n_dim)
        self.w0 = nn.Dense(N_max_character, n_position)
        self.wv = nn.Dense(n_dim, n_dim)
        self.we = nn.Dense(n_dim, N_max_character)
        self.active = nn.Tanh()
        self.softmax = nn.Softmax(axis=2)
        self.bmm = ops.BatchMatMul()

    def construct(self, enc_output):
        # enc_output: b,256,512
        reading_order = Tensor(np.arange(self.character_len), dtype=ms.int64)
        reading_order = mnp.repeat(reading_order.unsqueeze(0), enc_output.shape[0], 0)    # (S,) -> (B, S)
        reading_order = self.f0_embedding(reading_order)      # b,max_len,512
        # calculate attention
        t = self.w0(reading_order.transpose((0, 2, 1)))     # b,512,256
        t = self.active(t.transpose((0, 2, 1)) + self.wv(enc_output))     # (b,256,512) + (b, w*h, 512))
        t = self.we(t)  # b,256,max_len
        t = self.softmax(t.transpose((0, 2, 1)))  # b,max_len,256
        g_output = self.bmm(t, enc_output)  # b,max_len,512
        return g_output, t

class Prediction(nn.Cell):
    def __init__(self, n_dim=512, n_position=256, n_class=37, N_max_character=25):
        super().__init__()
        self.pp = PP_Layer(N_max_character=N_max_character, n_position=n_position)
        self.pp_share = PP_Layer(N_max_character=N_max_character, n_position=n_position)
        self.w_vrm = nn.Dense(n_dim, n_class)    # output layer
        self.w_share = nn.Dense(n_dim, n_class)    # output layer
        self.nclass = n_class

    def construct(self, cnn_feature, f_res, f_sub, is_train=False, use_mlm=True):
        if is_train:
            if not use_mlm:
                g_output, _ = self.pp(cnn_feature)  # b,max_len,512
                g_output = self.w_vrm(g_output)
                f_res = 0
                f_sub = 0
                return g_output, f_res, f_sub
            g_output, _ = self.pp(cnn_feature)  # b,max_len,512
            f_res, _ = self.pp_share(f_res)
            f_sub, _ = self.pp_share(f_sub)
            g_output = self.w_vrm(g_output)
            f_res = self.w_share(f_res)
            f_sub = self.w_share(f_sub)
            return g_output, f_res, f_sub
        else:
            g_output, _ = self.pp(cnn_feature)  # b,max_len,512
            g_output = self.w_vrm(g_output)
            return g_output

class MLM(nn.Cell):
    '''
    Architecture of MLM
    '''
    def __init__(self, n_dim=512, n_position=256, max_text_length=25):
        super().__init__()
        self.MLM_SequenceModeling_mask = TransformerEncoder(n_layers=2, n_position=n_position)
        self.MLM_SequenceModeling_WCL = TransformerEncoder(n_layers=1, n_position=n_position)
        self.pos_embedding = nn.Embedding(max_text_length, n_dim)
        self.w0_linear = nn.Dense(1, n_position)
        self.wv = nn.Dense(n_dim, n_dim)
        self.active = nn.Tanh()
        self.we = nn.Dense(n_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def construct(self, input, label_pos, state=False):
        # transformer unit for generating mask_c
        feature_v_seq = self.MLM_SequenceModeling_mask(input, src_mask=None)
        # position embedding layer
        pos_emb = self.pos_embedding(label_pos)
        pos_emb = self.w0_linear(pos_emb.unsqueeze(2)).transpose((0, 2, 1))
        # fusion position embedding with features V & generate mask_c
        att_map_sub = self.active(pos_emb + self.wv(feature_v_seq))
        att_map_sub = self.we(att_map_sub)  # b,256,1
        att_map_sub = self.sigmoid(att_map_sub.transpose((0, 2, 1)))  # b,1,256
        # WCL
        ## generate inputs for WCL
        f_res = input * (1 - att_map_sub.transpose((0, 2, 1))) # second path with remaining string
        f_sub = input * (att_map_sub.transpose((0, 2, 1))) # first path with occluded character
        ## transformer units in WCL
        f_res = self.MLM_SequenceModeling_WCL(f_res, src_mask=None)
        f_sub = self.MLM_SequenceModeling_WCL(f_sub, src_mask=None)
        return f_res, f_sub, att_map_sub
def trans_1d_2d(x):
    b, w_h, c = x.shape  # b, 256, 512
    x = x.transpose((0, 2, 1))
    x = x.view(b, c, 32, 8)
    x = x.transpose((0, 1, 3, 2))  # [16, 512, 8, 32]
    return x


class MLM_VRM(nn.Cell):
    """
    MLM+VRM, MLM is only used in training.
    ratio controls the occluded number in a batch.
    The pipeline of VisionLAN in testing is very concise with only a backbone + sequence modeling(transformer unit) + prediction layer(pp layer).
    x: input image
    label_pos: character index
    training_step: LF or LA process
    output
    text_pre: prediction of VRM
    test_rem: prediction of remaining string in MLM
    text_mas: prediction of occluded character in MLM
    mask_c_show: visualization of Mask_c
    """
    def __init__(self,
                 n_layers=3,
                 n_position=256,
                 n_dim=512,
                 max_text_length=25,
                 nclass=37):
        super().__init__()
        self.MLM = MLM()
        self.SequenceModeling = TransformerEncoder(n_layers=n_layers, n_position=n_position)
        self.Prediction = Prediction(n_dim = n_dim, n_position=n_position, 
                                     N_max_character=max_text_length+1, n_class=nclass) # N_max_character = 1 eos + 25 characters
        self.nclass = nclass
    def construct(self, input, label_pos, training_stp, is_train=False):
        b, c, h, w = input.shape
        nT = 25
        input = input.transpose((0, 1, 3, 2)) # (b, c, w, h)
        input = input.view(b, c, -1)  # (b, c, w*h)
        input = input.transpose((0, 2, 1)) # (b, w*h, c)
        if is_train:
            if training_stp == 'LF_1': # first stage(language-free): train without MLM
                f_res = 0
                f_sub = 0
                input = self.SequenceModeling(input, src_mask=None)
                text_pre, test_rem, text_mas = self.Prediction(input, f_res, f_sub, is_train=True, use_mlm=False)
                return text_pre, text_pre, text_pre, text_pre
            elif training_stp == 'LF_2': # second stage(language-free): train with MLM, finetune the other parts
                # MLM
                f_res, f_sub, mask_c = self.MLM(input, label_pos, state=True)
                input = self.SequenceModeling(input, src_mask=None)
                text_pre, test_rem, text_mas = self.Prediction(input, f_res, f_sub, is_train=True)
                mask_c_show = trans_1d_2d(mask_c.transpose((0, 2, 1)))
                return text_pre, test_rem, text_mas, mask_c_show
            elif training_stp == 'LA': # third stage(language-aware): using generated mask to mask out images
                # MLM
                f_res, f_sub, mask_c = self.MLM(input, label_pos, state=True)
                ## use the mask_c (1 for occluded character and 0 for remaining characters) to occlude input
                ## ratio controls the occluded number in a batch
                ratio = 2
                character_mask = ops.ZerosLike()(mask_c)
                character_mask[0:b // ratio, :, :] = mask_c[0:b // ratio, :, :]
                input = input *( 1- character_mask.transpose((0, 2, 1)).float())
                # VRM
                ## transformer unit for VRM
                input = self.SequenceModeling(input, src_mask=None)
                ## prediction layer for MLM and VSR
                text_pre, test_rem, text_mas = self.Prediction(input, f_res, f_sub, is_train=True)
                mask_c_show = trans_1d_2d(mask_c.transpose((0, 2, 1)))
                return text_pre, test_rem, text_mas, mask_c_show
            else:
                NotImplementedError
        else: # VRM is only used in the testing stage
            f_res = 0
            f_sub = 0
            contextual_feature = self.SequenceModeling(input, src_mask=None)
            C = self.Prediction(contextual_feature, f_res, f_sub, is_train=False, use_mlm=False)
            C = C.transpose((1, 0, 2))  # (max_len, b, 37))
            return C

        
class VisionLANHead(nn.Cell):
    def __init__(self, 
                 in_channels,
                 out_channels= 36,
                 n_layers = 3,
                 n_position = 256,
                 n_dim=512,
                 max_text_length = 25,
                 training_step = 'LA'):
        super().__init__()
        self.MLM_VRM = MLM_VRM(
            n_layers=n_layers,
            n_position=n_position,
            n_dim=n_dim,
            max_text_length=max_text_length,
            nclass=out_channels + 1)
        self.in_channels = in_channels
        self.training_step = training_step
    def construct(self, features, targets=None):
        # MLM + VRM
        if self.training:
            label_pos = targets[-2]
            text_pre, test_rem, text_mas, mask_map = self.MLM_VRM(features, label_pos, self.training_step, is_train=True)
            return text_pre, test_rem, text_mas, mask_map
        else:
            output  = self.MLM_VRM(features, targets, self.training_step, is_train=False)
            return output