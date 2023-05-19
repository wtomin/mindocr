from typing import Tuple, List
from mindspore import Tensor, nn, ops
import mindspore.common.dtype as mstype
from .mindcv_models.resnet import ResNet, Bottleneck, default_cfgs
from .mindcv_models.utils import load_pretrained
from ._registry import register_backbone, register_backbone_class
from mindocr.models.layers import PositionalEncoding2D
import math
__all__ = ['MaskedResNet', 'det_masked_joiner_resnet50']

@register_backbone_class
class MaskedResNet(ResNet):
    """
    A resnet backbone accepting a NestedTensor as input, which has tensors and masks, image_sizes attributes.
    """
    def __init__(self, block, layers, num_levels, **kwargs):
        super().__init__(block, layers, **kwargs)
        del self.pool, self.classifier  # remove the original header to avoid confusion
        # self.out_indices = out_indices
        self.num_levels = num_levels
        assert self.num_levels<=4, "expected to get num_levels <=4."
        out_channels = [ch * block.expansion for ch in [64, 128, 256, 512]]
        self.out_channels = [out_channels[i] for i in range(4 - self.num_levels, 4)]
        out_strides = [4, 8, 16, 32]
        self.feature_strides = [out_strides[i] for i in range(4 - self.num_levels, 4)]
    

    def construct(self, images: List[Tensor]) -> List[Tensor]:
        #images = self.preprocess_image(images)
        # images is a dictionary: {'image': image, 'image_mask': mask, 'image_size': image_size}
        assert isinstance(images, dict), "images should be a dictionary! got {}".format(type(images))
        x = images['image']
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
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in multi_level_features],
            images['image_size']
        )
        output_features = [dict({'tensor': multi_level_features[i], 'mask': masks[i]}) for i in range(self.num_levels)]
        return output_features
        
    def mask_out_padding(self, feature_shapes, image_sizes):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides), "expected to get the same number of feature shapes and feature strides."
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = ops.ones((N, H, W), dtype=mstype.bool_)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(math.ceil(float(h) / self.feature_strides[idx])),
                    : int(math.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


def det_masked_resnet50(pretrained: bool = True, num_levels: int = 3,  **kwargs):
    model = MaskedResNet(Bottleneck, [3, 4, 6, 3], num_levels, **kwargs)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['resnet50']
        load_pretrained(model, default_cfg)

    return model

class Joiner(nn.SequentialCell):
    """
    A SequentialCell that joins the backbone and the position encoding.
    """
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.num_levels = backbone.num_levels

        self.out_channels = backbone.out_channels
        self.feature_strides = backbone.feature_strides
     
    def construct(self, inputs):
        xs = self[0](inputs) # backbone returns a list of dictonaries for multilevels features and masks
        masks = [x['mask'] for x in xs]
        out = []
        pos = []
        for i in range(len(xs)):
            out.append(xs[i])
            # position encoding
            pos.append(self[1](masks[i]))

        return out, pos

@register_backbone
def det_masked_joiner_resnet50(pretrained: bool = True, num_levels: int = 3, hidden_size: int = 256, **kwargs):
    model = MaskedResNet(Bottleneck, [3, 4, 6, 3], num_levels, **kwargs)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['resnet50']
        load_pretrained(model, default_cfg)

    return Joiner(model, PositionalEncoding2D(hidden_size//2, normalize=True))
