from typing import Tuple, List
from mindspore import Tensor, nn, ops
import mindspore.common.dtype as mstype
from mindcv.models.resnet import ResNet, Bottleneck, default_cfgs
from mindcv.models.utils import load_pretrained
from ._registry import register_backbone, register_backbone_class
from mindocr.utils.misc import NestedTensor
from mindocr.models.layers import PositionalEncoding2D
__all__ = ['MaskedResNet', 'det_masked_joiner_resnet50']

@register_backbone_class
class MaskedResNet(ResNet):
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
    
    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images) # the ImageList is imported from detectron2, needs to be replaced.
        return images

    def construct(self, images: NestedTensor) -> List[Tensor]:
        images = self.preprocess_image(images)
        x = images.tensor
        # basic stem
        x = self.conv1(x)  # stride: 2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x) # stride: 4

        x1 = self.layer1(x)  # stride: 4
        x2 = self.layer2(x1)  # stride: 8
        x3 = self.layer3(x2)  # stride: 16
        x4 = self.layer4(x3)  # stride: 32

        multi_level_features =  [x1, x2, x3, x4]
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in multi_level_features],
            images.image_sizes
        )
        output_features = [NestedTensor(multi_level_features[i], masks[i]) for i in range(4-self.num_levels, 4)]
        return output_features
        
    def mask_out_padding(self, feature_shapes, image_sizes):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level =ops.ones((N, H, W), dtype=mstype.bool_)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(ops.ceil(float(h) / self.feature_strides[idx])),
                    : int(ops.ceil(float(w) / self.feature_strides[idx])),
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
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.num_levels = backbone.num_levels

        self.out_channels = backbone.out_channels
        self.feature_strides = backbone.feature_strides
     
    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x))

        return out, pos

@register_backbone
def det_masked_joiner_resnet50(pretrained: bool = True, num_levels: int = 3, hidden_size: int = 1024, **kwargs):
    model = MaskedResNet(Bottleneck, [3, 4, 6, 3], num_levels, **kwargs)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['resnet50']
        load_pretrained(model, default_cfg)

    return Joiner(model, PositionalEncoding2D(hidden_size, normalize=True))
