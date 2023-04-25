from typing import List, Union
import cv2
import numpy as np
from PIL import Image
from mindspore.dataset.vision import RandomColorAdjust as MSRandomColorAdjust
from mindspore.dataset.vision import Rotate, HorizontalFlip, VerticalFlip
#RandomHorizontalFlipWithBBox, RandomVerticalFlipWithBBox, 
import random
from ...data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mindspore.dataset.vision import RandomColorAdjust as MSRandomColorAdjust, ToPIL


__all__ = ['DecodeImage', 'NormalizeImage', 'ToCHWImage', 'PackLoaderInputs', 'ScalePadImage', 'GridResize',
           'RandomScale', 'RandomCropWithBBox', 'RandomColorAdjust', 'RandomFlipWithBBox', 'ResizeShortestEdgeWithBBox', 
           'RandomRotationWithBBox']


# TODO: use mindspore C.decode for efficiency
class DecodeImage:
    """
    img_mode (str): The channel order of the output, 'BGR' and 'RGB'. Default to 'BGR'.
    channel_first (bool): if True, image shpae is CHW. If False, HWC. Default to False
    """
    def __init__(self, img_mode='BGR', channel_first=False, to_float32=False, ignore_orientation=False, **kwargs):
        self.img_mode = img_mode
        self.to_float32 = to_float32
        self.channel_first = channel_first
        self.flag = cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR if ignore_orientation else cv2.IMREAD_COLOR

    def __call__(self, data):
        if 'img_path' in data:
            with open(data['img_path'], 'rb') as f:
                img = f.read()
        elif 'img_lmdb' in data:
            img = data["img_lmdb"]
        img = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(img, self.flag)

        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        if self.to_float32:
            img = img.astype('float32')
        data['image'] = img
        # data['ori_image'] = img.copy()
        return data


class NormalizeImage:
    """
    normalize image, substract mean, divide std
    input image: by default, np.uint8, [0, 255], HWC format.
    return image: float32 numpy array
    """
    def __init__(self, mean: Union[List[float], str] = 'imagenet', std: Union[List[float], str] = 'imagenet',
                 is_hwc=True, bgr_to_rgb=False, rgb_to_bgr=False, **kwargs):
        # By default, imagnet MEAN and STD is in RGB order. inverse if input image is in BGR mode
        self._channel_conversion = False
        if bgr_to_rgb or rgb_to_bgr:
            self._channel_conversion = True

        # TODO: detect hwc or chw automatically
        shape = (3, 1, 1) if not is_hwc else (1, 1, 3)
        self.mean = np.array(self._get_value(mean, 'mean')).reshape(shape).astype('float32')
        self.std = np.array(self._get_value(std, 'std')).reshape(shape).astype('float32')
        self.is_hwc = is_hwc

    def __call__(self, data):
        img = data['image']
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"

        if self._channel_conversion:
            if self.is_hwc:
                img = img[..., [2, 1, 0]]
            else:
                img = img[[2, 1, 0], ...]

        data['image'] = (img.astype('float32') - self.mean) / self.std
        return data

    @staticmethod
    def _get_value(val, name):
        if isinstance(val, str) and val.lower() == 'imagenet':
            assert name in ['mean', 'std']
            return IMAGENET_DEFAULT_MEAN if name == 'mean' else IMAGENET_DEFAULT_STD
        elif isinstance(val, list):
            return val
        else:
            raise ValueError(f'Wrong {name} value: {val}')


class ToCHWImage:
    # convert hwc image to chw image
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        if isinstance(img, Image.Image):
            img = np.array(img)
        data['image'] = img.transpose((2, 0, 1))
        return data


class PackLoaderInputs:
    """
    Args:
        output_columns (list): the keys in data dict that are expected to output for dataloader

    Call:
        input: data dict
        output: data tuple corresponding to the `output_columns`
    """
    def __init__(self, output_columns: List, **kwargs):
        self.output_columns = output_columns

    def __call__(self, data):
        out = []
        for k in self.output_columns:
            assert k in data, f'key {k} does not exists in data, availabe keys are {data.keys()}'
            out.append(data[k])

        return tuple(out)


class ScalePadImage:
    """
    Scale image and polys by the shorter side, then pad to the target_size.
    input image format: hwc

    Args:
        target_size: [H, W] of the output image.
    """
    def __init__(self, target_size: list):
        self._target_size = np.array(target_size)

    def __call__(self, data: dict):
        """
        required keys:
            image, HWC
            (polys)
        modified keys:
            image
            (polys)
        added keys:
            shape: [src_h, src_w, scale_ratio_h, scale_ratio_w]
        """
        size = np.array(data['image'].shape[:2])
        scale = min(self._target_size / size)
        new_size = np.round(scale * size).astype(np.int)

        data['image'] = cv2.resize(data['image'], new_size[::-1])
        data['image'] = np.pad(data['image'],
                               (*tuple((0, ts - ns) for ts, ns in zip(self._target_size, new_size)), (0, 0)))

        if 'polys' in data:
            data['polys'] *= scale

        data['shape'] = np.concatenate((size, np.array([scale, scale])), dtype=np.float32)
        return data


class GridResize:
    """
    Resize image to make it divisible by a specified factor exactly.
    Resize polygons correspondingly, if provided.
    """
    def __init__(self, factor: int = 32):
        self._factor = factor

    def __call__(self, data: dict):
        """
        required keys:
            image, HWC
            (polys)
        modified keys:
            image
            (polys)
        """
        size = np.array(data['image'].shape[:2])
        scale = np.ceil(size / self._factor) * self._factor / size
        data['image'] = cv2.resize(data['image'], None, fx=scale[1], fy=scale[0])

        if 'polys' in data:
            data['polys'] *= scale[::-1]  # w, h order
        return data


class RandomScale:
    """
    Randomly scales an image and its polygons in a predefined scale range.
    Args:
        scale_range: (min, max) scale range.
    """
    def __init__(self, scale_range: Union[tuple, list]):
        self._range = scale_range

    def __call__(self, data: dict):
        """
        required keys:
            image, HWC
            (polys)
        modified keys:
            image
            (polys)
        """
        scale = np.random.uniform(*self._range)
        data['image'] = cv2.resize(data['image'], dsize=None, fx=scale, fy=scale)

        if 'polys' in data:
            data['polys'] *= scale
        return data


class RandomCropWithBBox:
    """
    Randomly cuts a crop from an image along with polygons.

    Args:
        max_tries: number of attempts to try to cut a crop with a polygon in it.
        min_crop_ratio: minimum size of a crop in respect to an input image size.
        crop_size: target size of the crop (resized and padded, if needed), preserves sides ratio.
    """
    def __init__(self, max_tries=10, min_crop_ratio=0.1, pad_if_needed=True, crop_size=(640, 640)):
        """
        Args:
            crop_size: target size of the crop (padded, if needed)
            pad_if_needed: if True, pads the image to the target size, otherwise keep the crop size.
        """
        self._crop_size = crop_size
        self._pad_if_needed = pad_if_needed
        self._ratio = min_crop_ratio
        self._max_tries = max_tries

    def __call__(self, data):
        start, end = self._find_crop(data)
        if self._pad_if_needed:
            scale = min(self._crop_size / (end - start))
            data['image'] = cv2.resize(data['image'][start[0]: end[0], start[1]: end[1]], None, fx=scale, fy=scale)
            data['image'] = np.pad(data['image'],
                               (*tuple((0, cs - ds) for cs, ds in zip(self._crop_size, data['image'].shape[:2])), (0, 0)))
            
        else:
            data['image'] = data['image'][start[0]: end[0], start[1]: end[1]]
            scale = 1
        start, end = start[::-1], end[::-1]     # convert to x, y coord
        new_polys, new_texts, new_ignores = [], [], []
        for _id in range(len(data['polys'])):
            # if the polygon is within the crop
            if (data['polys'][_id].min(axis=0) > start).all() and (data['polys'][_id].max(axis=0) < end).all():   # NOQA
                new_polys.append((data['polys'][_id] - start) * scale)
                new_texts.append(data['texts'][_id])
                new_ignores.append(data['ignore_tags'][_id])

        data['polys'] = np.array(new_polys).astype(np.int64) if isinstance(data['polys'], np.ndarray) else new_polys
        data['texts'] = new_texts
        data['ignore_tags'] = new_ignores

        return data

    def _find_crop(self, data):
        size = np.array(data['image'].shape[:2])
        polys = [poly for poly, ignore in zip(data['polys'], data['ignore_tags']) if not ignore]

        if polys:
            # do not crop through polys => find available "empty" coordinates
            h_array, w_array = np.zeros(size[0], dtype=np.int32), np.zeros(size[1], dtype=np.int32)
            for poly in polys:
                points = np.maximum(np.round(poly).astype(np.int32), 0)
                w_array[points[:, 0].min(): points[:, 0].max() + 1] = 1
                h_array[points[:, 1].min(): points[:, 1].max() + 1] = 1

            if not h_array.all() and not w_array.all():     # if texts do not occupy full image
                # find available coordinates that don't include text
                h_avail = np.where(h_array == 0)[0]
                w_avail = np.where(w_array == 0)[0]

                min_size = np.ceil(size * self._ratio).astype(np.int32)
                for _ in range(self._max_tries):
                    y = np.sort(np.random.choice(h_avail, size=2))
                    x = np.sort(np.random.choice(w_avail, size=2))
                    start, end = np.array([y[0], x[0]]), np.array([y[1], x[1]])

                    if ((end - start) < min_size).any():    # NOQA
                        continue

                    # check that at least one polygon is within the crop
                    for poly in polys:
                        if (poly.max(axis=0) > start[::-1]).all() and (poly.min(axis=0) < end[::-1]).all():     # NOQA
                            return start, end

        # failed to generate a crop or all polys are marked as ignored
        return np.array([0, 0]), size


class RandomColorAdjust:
    def __init__(self, brightness=32.0 / 255, saturation=0.5):
        self._jitter = MSRandomColorAdjust(brightness=brightness, saturation=saturation)
        self._pil = ToPIL()

    def __call__(self, data):
        """
        required keys: image
        modified keys: image
        """
        # there's a bug in MindSpore that requires images to be converted to the PIL format first
        data['image'] = np.array(self._jitter(self._pil(data['image'])))
        return data

class RandomFlipWithBBox:
    def __init__(self, horizontal=True, vertical=False, prob=0.5):
        self.horizontal = horizontal
        self.h_flip = HorizontalFlip()
        self.vertical = vertical 
        self.v_flip = VerticalFlip()
        self.prob = prob
    def horizontal_flip_bbox(self, bbox, image_shape):
        _,width = image_shape
        bbox[:, :, 0] = width - bbox[:, :, 0]
        return bbox
    def vertical_flip_bbox(self, bbox, image_shape):
        height,_ = image_shape
        bbox[:, :, 1] = height - bbox[:, :, 1]
        return bbox
    def __call__(self, data: dict) -> dict:
        image = data["image"]
        bbox = data['polys']
        p = np.random.uniform(0, 1)
        if p<=self.prob:
            if self.horizontal:
                image = self.h_flip(image)
                bbox = self.horizontal_flip_bbox(bbox, image.shape[:2])
            if self.vertical:
                image = self.v_flip(image)
                bbox = self.vertical_flip_bbox(bbox, image.shape[:2])
        data["image"] = image
        data['polys'] = bbox
        return data
class ResizeShortestEdgeWithBBox(object):
    def __init__(self, short_edge_length: List[int], max_size: int, sample_style: str):
        """
        Args:
            short_edge_length (list[int]): list of possible shortest edge length
            max_size (int): maximum allowed longest edge length
            sample_style (str): "choice" or "range". If "choice", a length will be
                randomly chosen from `short_edge_length`. If "range", a length will be
                sampled from the range of min(short_edge_length) and max(short_edge_length).
        """
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self.is_range = sample_style == "range" 
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self.short_edge_length = short_edge_length
        self.max_size = max_size
        self.sample_style = sample_style

    def __call__(self, data):
        h, w = data["image"].shape[:2]
        if self.sample_style == "choice":
            short_edge = random.choice(self.short_edge_length)
        elif self.sample_style == "range":
            short_edge = random.randint(
                min(self.short_edge_length), max(self.short_edge_length)
            )
        else:
            raise ValueError("Unknown sample style: {}".format(self.sample_style))

        # Prevent the biggest axis from being more than max_size
        scale = min(short_edge / min(h, w), self.max_size / max(h, w))
        newh, neww = int(h * scale + 0.5), int(w * scale + 0.5)
        data["image"] = cv2.resize(data["image"], (neww, newh))
        data["polys"] = data["polys"] * scale
        return data

class RandomRotationWithBBox:
    def __init__(self, angle, expand=True, center=None, sample_style="range", interp=None):
        """
        Args:
            angle (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the angle (in degrees).
                If ``sample_style=="choice"``, a list of angles to sample from
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
        """
        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        if isinstance(angle, (float, int)):
            angle = (angle, angle)
        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)
        self.angle = angle
        self.center = center
        self.expand = expand
        self.interp = interp

    def rotate_bbox(self, bbox, angle, center):
        cx, cy = center
        # translate the bounding box to center at origin
        translated_bbox = bbox - np.array([cx, cy])
        # rotate the bounding box
        theta = np.radians(angle)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        rotated_bbox = np.dot(translated_bbox, rotation_matrix)
        rotated_bbox = rotated_bbox + np.array([cx, cy])
        return rotated_bbox
    def __call__(self, data):
        image = data["image"]
        h, w = image.shape[:2]
        bbox = data['polys']
        if self.is_range:
            angle = random.uniform(self.angle[0], self.angle[1])
        else:
            angle = random.choice(self.angle)
        if self.center is not None:
            if self.is_range:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
            else:
                center = random.choice(self.center)
        else:
            center = None
        if center is not None:
            center = (center[0] * w, center[1] * h) # Convert to absolute coordinates
        else:
            center = (0.5 * w, 0.5 * h)
        if angle%360==0:
            return data
        image = Rotate(angle, expand=self.expand, center=center)(image)
        bbox = self.rotate_bbox(bbox, angle, center)

        data["image"] = image
        data['polys'] = bbox.astype(np.int64)
        return data