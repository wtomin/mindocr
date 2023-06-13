from typing import Tuple, List
from mindspore import Tensor, nn, ops
import mindspore.numpy as mnp
import mindspore.common.dtype as mstype
from .mindcv_models.resnet import ResNet, Bottleneck, default_cfgs
from .mindcv_models.utils import load_pretrained
from ._registry import register_backbone, register_backbone_class
from mindocr.models.layers import PositionalEncoding2D

__all__ = ['MaskedResNet', 'det_masked_joiner_resnet50']

@register_backbone_class
class MaskedResNet(ResNet):
    """
    A resnet backbone accepting a NestedTensor as input, which has tensors and masks, image_sizes attributes.
    """
    def __init__(self, block, layers, num_levels, hidden_size, **kwargs):
        super().__init__(block, layers, **kwargs)
        del self.pool, self.classifier  # remove the original header to avoid confusion
        # self.out_indices = out_indices
        self.num_levels = num_levels
        assert self.num_levels<=4, "expected to get num_levels <=4."
        out_channels = [ch * block.expansion for ch in [64, 128, 256, 512]]
        self.out_channels = [out_channels[i] for i in range(4 - self.num_levels, 4)]
        out_strides = [4, 8, 16, 32]
        self.feature_strides = [out_strides[i] for i in range(4 - self.num_levels, 4)]
        self.positional_encoder = PositionalEncoding2D(hidden_size//2, normalize=True)
    

    def construct(self, inputs):
        image, image_mask = inputs
        x = image
        # basic stem
        x = self.conv1(x)  # stride: 2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x) # stride: 4

        x1 = self.layer1(x)  # stride: 4
        x2 = self.layer2(x1)  # stride: 8
        x3 = self.layer3(x2)  # stride: 16
        x4 = self.layer4(x3)  # stride: 32

        multi_level_features =  [x2, x3, x4]
        # masks = self.mask_out_padding(
        #     [features_per_level.shape for features_per_level in multi_level_features],
        #     image_size
        # )
        feature_shapes = [features_per_level.shape[-2:] for features_per_level in multi_level_features]
        masks = [self.rescale_mask(image_mask, feature_shapes[i]) for i in range(len(feature_shapes))]

        pos = []
        for i in range(len(multi_level_features)):
            # position encoding
            pos.append(self.positional_encoder(masks[i]))
        return multi_level_features, masks, pos
    def rescale_mask(self, mask, target_shape):
        mask = ops.interpolate(mask.unsqueeze(1).float(), size=target_shape, mode='bilinear')
        if mask.shape[1] != 1:
            raise ValueError("Expected mask to have 1 channel, got {}".format(mask.shape[1])
                             )
        mask= mask.squeeze(1).bool()
        return ops.stop_gradient(mask)
    # def mask_out_padding(self, multilevel_feat_shapes, image_sizes):
    #     masks = []
    #     assert len(multilevel_feat_shapes) == len(self.feature_strides), "expected to get the same number of feature shapes and feature strides."
    #     N_levels = len(multilevel_feat_shapes)
    #     for idx in range(N_levels):
    #         N, _, H, W = multilevel_feat_shapes[idx]
    #         masks_per_feature_level = ops.ones((N, H, W), dtype=mstype.bool_) # dtype or type depends on the version of mindspore
    #         N_imgs_batch = len(image_sizes)
    #         image_sizes = image_sizes.float() # convert to float
    #         for img_idx in range(N_imgs_batch):
    #             h, w = image_sizes[img_idx]
    #             h_i , w_i = Tensor(mnp.ceil(h / self.feature_strides[idx]), mstype.int32), Tensor( mnp.ceil( w / self.feature_strides[idx]), mstype.int32)
    #             #ValueError: When using JIT Fallback to handle script 'math.ceil(float(h) / self.feature_strides[idx])', 
    #             # the inputs should be constant, but found variable 'AbstractScalar(Type: String, Value: h, Shape: NoShape)' 
    #             # to be nonconstant.
    #             masks_per_feature_level[img_idx, :h_i, :w_i] = 0
    #         masks.append(masks_per_feature_level)
    #     return masks


def det_masked_resnet50(pretrained: bool = True, num_levels: int = 3,  **kwargs):
    model = MaskedResNet(Bottleneck, [3, 4, 6, 3], num_levels, **kwargs)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['resnet50']
        load_pretrained(model, default_cfg)

    return model

# class Joiner(nn.Cell):
#     """
#     A Cell that joins the backbone and the position encoding.
#     """
#     def __init__(self, backbone, position_embedding):
#         super().__init__()
#         self.backbone = backbone
#         self.positional_embedding = position_embedding
#         self.num_levels = backbone.num_levels

#         self.out_channels = backbone.out_channels
#         self.feature_strides = backbone.feature_strides
     
#     def construct(self, *inputs):
#         xs = self.backbone(*inputs) # backbone returns a list of lists consists of features and masks
#         out = []
#         pos = []
#         for i in range(len(xs[0])):
#             out.append(xs[0][i])
#             # position encoding
#             pos.append(self.positional_embedding(xs[1][i]))

#         return out, pos #xs[0], mask_pos

@register_backbone
def det_masked_joiner_resnet50(pretrained: bool = True, num_levels: int = 3, hidden_size: int = 256, **kwargs):
    model = MaskedResNet(Bottleneck, [3, 4, 6, 3], num_levels, hidden_size, **kwargs)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['resnet50']
        load_pretrained(model, default_cfg)

    return model
