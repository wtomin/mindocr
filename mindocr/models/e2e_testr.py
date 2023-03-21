from mindspore import nn
from mindcv.models.utils import load_pretrained
from . base_model import BaseModel
from ._registry import register_model

__all__ = ['TESTR', 'testr_r50']

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 640, 640), 
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        **kwargs
    }


default_cfgs = {
    # ResNet and Wide ResNet
    'testr_r50': _cfg(
        url='xx')
    }


class TESTR(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def testr_r50(pretrained=False, **kwargs):
    model_config = {
            "backbone": {
                'name': 'det_masked_joiner_resnet50',
                'pretrained': False 
                },
            "neck": {
                "name": 'TESTRDeformableTransformer',
                #"out_channels": 1024,
                "batch_size": 2, 
                "src_seq_length": 28*28+14*14+7*7,
                "tgt_seq_length": 100, # same to number of proposals
                "num_levels": 4,
                "hidden_size": 256,
                "num_heads": 8,
                "ffn_hidden_size": 1024,
                "num_encoder_layers": 6,
                "num_decoder_layers": 6,
                "dec_num_points": 4,
                "enc_num_points": 4,
                "num_ctrl_points": 16,
                "dropout_rate": 0.1,
                "attention_dropout_rate": 0.0,
                "hidden_dropout_rate": 0.0,
                "activation": "relu",
                "pos_embed_scale": 6.283185307179586, #2 PI
                "max_text_len": 25,
                "return_intermediate_dec": True,
                "num_classes": 1,
                "voc_size": 96,
                "num_proposals": 100,
                },
            "head": {
                "name": 'TESTRHead',
                "num_classes": 1,
                "num_pred": 6, # same to the number of decoder layers
                "use_polygon": True,
                "aux_loss": True,
                "voc_size": 96,
                "hidden_size": 256,
                },
            "loss": {
                "name": 'TESTRLoss',
                "num_classes": 1,
                "num_ctrl_points": 16,
                "point_class_weight": 2.0,
                "point_coord_weight": 5.0,
                "point_text_weight": 4.0,
                "box_class_weight": 2.0,
                "box_coord_weight": 5.0,
                "box_giou_weight": 2.0,
                "focal_alpha": 0.25,
                "focal_gamma": 2.0
                }
            }
    model = TESTR(model_config)
    
    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['testr_r50']
        load_pretrained(model, default_cfg)

    return model

if __name__ =="__main__":
    model = testr_r50(pretrained=False)
