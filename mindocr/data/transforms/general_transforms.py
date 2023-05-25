import random
from typing import List, Union, Tuple
from copy import copy
import cv2
import warnings
import numpy as np
from PIL import Image
from mindspore.dataset.vision import RandomColorAdjust as MSRandomColorAdjust
from mindspore.dataset.vision import Rotate, HorizontalFlip, VerticalFlip
import random
from ...data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from shapely.geometry import Polygon, box
from mindspore.dataset.vision import RandomColorAdjust as MSRandomColorAdjust, ToPIL

__all__ = ['DecodeImage', 'NormalizeImage', 'ToCHWImage', 'PackLoaderInputs', 'ScalePadImage', 'GridResize',
           'RandomScale', 'RandomCropWithBBox', 'RandomCropWithInstances', 'ResizePadImage', 'RandomColorAdjust',  'ValidatePolygons']


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
        p: probability of the augmentation being applied to an image.
    """
    def __init__(self, scale_range: Union[tuple, list], p: float = 0.5):
        self._range = scale_range
        self._p = p

    def __call__(self, data: dict):
        """
        required keys:
            image, HWC
            (polys)
        modified keys:
            image
            (polys)
        """
        if random.random() < self._p:
            scale = np.random.uniform(*self._range)
            data['image'] = cv2.resize(data['image'], dsize=None, fx=scale, fy=scale)

            if 'polys' in data:
                data['polys'] *= scale
        return data


class RandomCropWithBBox:
    """
    Randomly cuts a crop from an image along with polygons in the way that the crop doesn't intersect any polygons
    (i.e. any given polygon is either fully inside or fully outside the crop).

    Args:
        max_tries: number of attempts to try to cut a crop with a polygon in it. If fails, scales the whole image to
                   match the `crop_size`.
        min_crop_ratio: minimum size of a crop in respect to an input image size.
        crop_size: target size of the crop (resized and padded, if needed), preserves sides ratio.
        p: probability of the augmentation being applied to an image.
    """
    def __init__(self, max_tries=10, min_crop_ratio=0.1, crop_size=(640, 640), p: float = 0.5):
        self._crop_size = crop_size
        self._ratio = min_crop_ratio
        self._max_tries = max_tries
        self._p = p

    def __call__(self, data):
        if random.random() < self._p:   # cut a crop
            start, end = self._find_crop(data)
        else:                           # scale and pad the whole image
            start, end = np.array([0, 0]), np.array(data['image'].shape[:2])

        scale = min(self._crop_size / (end - start))

        data['image'] = cv2.resize(data['image'][start[0]: end[0], start[1]: end[1]], None, fx=scale, fy=scale)
        data['actual_size'] = np.array(data['image'].shape[:2])
        data['image'] = np.pad(data['image'],
                               (*tuple((0, cs - ds) for cs, ds in zip(self._crop_size, data['image'].shape[:2])), (0, 0)))

        data['polys'] = (data['polys'] - start[::-1]) * scale

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

# class ResizeShortestEdgeWithBBox(object):
#     def __init__(self, short_edge_length: List[int], max_size: int, sample_style: str):
#         """
#         Args:
#             short_edge_length (list[int]): list of possible shortest edge length
#             max_size (int): maximum allowed longest edge length
#             sample_style (str): "choice" or "range". If "choice", a length will be
#                 randomly chosen from `short_edge_length`. If "range", a length will be
#                 sampled from the range of min(short_edge_length) and max(short_edge_length).
#         """
#         if isinstance(short_edge_length, int):
#             short_edge_length = (short_edge_length, short_edge_length)
#         self.is_range = sample_style == "range" 
#         if self.is_range:
#             assert len(short_edge_length) == 2, (
#                 "short_edge_length must be two values using 'range' sample style."
#                 f" Got {short_edge_length}!"
#             )
#         self.short_edge_length = short_edge_length
#         self.max_size = max_size
#         self.sample_style = sample_style

#     def __call__(self, data):
#         h, w = data["image"].shape[:2]
#         if self.sample_style == "choice":
#             short_edge = random.choice(self.short_edge_length)
#         elif self.sample_style == "range":
#             short_edge = random.randint(
#                 min(self.short_edge_length), max(self.short_edge_length)
#             )
#         else:
#             raise ValueError("Unknown sample style: {}".format(self.sample_style))

#         # Prevent the biggest axis from being more than max_size
#         scale = min(short_edge / min(h, w), self.max_size / max(h, w))
#         newh, neww = int(h * scale + 0.5), int(w * scale + 0.5)
#         data["image"] = cv2.resize(data["image"], (neww, newh))
#         data["polys"] = data["polys"] * scale
#         for key in ['boxes', 'bboxes']:
#             if key in data:
#                 data[key] = data[key] * scale

class ValidatePolygons:
    """
    Validate polygons by:
     1. filtering out polygons outside an image.
     2. clipping coordinates of polygons that are partially outside an image to stay within the visible region.
    Args:
        min_area: minimum area below which newly clipped polygons considered as ignored.
    """
    def __init__(self, min_area: float = 1.0, affect_keys=['polys', 'ignore_tags']):
        self._min_area = min_area
        self._affect_keys = affect_keys
        assert 'image' not in affect_keys, "image must not be in the affect keys"

    def __call__(self, data: dict):
        # get image size from the actual image/size or the mask
        if 'actual_size' in data:
            size = data['actual_size']
        elif 'image_mask' in data:
            mask = data['image_mask']
            pixel_indices = np.where(mask==0)
            min_x, min_y = pixel_indices[0].min(), pixel_indices[1].min()
            max_x, max_y = pixel_indices[0].max(), pixel_indices[1].max()
            size = np.array([max_x - min_x+1, max_y - min_y+1])
        else:
            size = np.array(data['image'].shape[:2])
        size = size[::-1]# convert to x, y coord 
        border = box(0, 0, *size)
        # affect_keys is a dictionary containing keys that will be affected by the polygon validation
        affect_keys_without_polys = copy(self._affect_keys)
        affect_keys_without_polys.pop(affect_keys_without_polys.index('polys'))
        assert 'polys' in data, "polys must be in the data"
        update_data = {}
        for key in self._affect_keys:
            update_data[key] = []
        
        for i, np_poly in enumerate(data['polys']):
            ignore = False
            include_poly = False
            if ((0 <= np_poly) & (np_poly < size)).all():   # if the polygon is fully within the image
                update_data['polys'].append(np.concatenate([np_poly, np.ones((len(np_poly), 1))], axis=-1))
                include_poly = True
            else:
                poly = Polygon(np_poly)
                if poly.intersects(border):                 # if the polygon is partially within the image
                    intersection = poly.intersection(border)
                    if intersection.area < self._min_area:
                        ignore = True

                    # poly = poly.exterior
                    # poly = poly.coords[::-1] if poly.is_ccw else poly.coords    # sort in clockwise order
                    # new_data['polys'].append(np.array(poly[:-1]))
                    inside = (np_poly >=np.array([0,0])) & (np_poly <= size)
                    inside = inside.all(axis=1)
                    update_data['polys'].append(np.concatenate([np_poly, inside.astype('float32').reshape(-1,1)], axis=-1))
                    include_poly = True
                else:                                       # the polygon is fully outside the image
                    include_poly = False
            if include_poly:
                for key in affect_keys_without_polys: 
                    if key == 'ignore_tags':
                        update_data[key].append(ignore)
                    else:
                        update_data[key].append(data[key][i])

        # update the data
        data.update(update_data) # affect_keys will be replaced by the updated data
        # change the data type
        length = len(data['polys']) 
        assert length>0, "length of polys must be greater than 0"
        for key in data.keys():
            if key in self._affect_keys:
                assert len(data[key]) == length, f"length of {key} must be equal to length of {self._affect_keys[0]}"
                if key in ['polys', 'boxes', 'ignore_tags', 'rec_ids', 'gt_classes']:
                    data[key] = np.array(data[key])
                    if key in ['polys', 'boxes']:
                        data[key] = data[key].astype('float32')
                    elif key in ['rec_ids', 'gt_classes']:
                        data[key] = data[key].astype('int32')
                    elif key in ['ignore_tags']:
                        data[key] = data[key].astype('bool')
        return data

class RandomCropWithInstances:
    """
    Randomly crop the image so that the cropping region contains all the bboxes/instances in the original image
    Args:
        crop_size (int): size of the crop.
        crop_type (str): type of the crop. One of ['relative', 'absolute']
        crop_box (bool): whether to allow cropping bounding boxes that are partially outside the image. 
    """
    def __init__(self, crop_size: int, crop_type:str, keep_all_boxes = False):
        self._crop_type = crop_type
        self._keep_all_boxes = keep_all_boxes
        assert  self._crop_type in ['relative', 'absolute'], f"crop_type must be one of ['relative', 'absolute']. Got {self._crop_type}!"
        if isinstance(crop_size, int):
            self._crop_size = (crop_size, crop_size)
        elif isinstance(crop_size, float):
            self._crop_size = (crop_size, crop_size)
        elif isinstance(crop_size, tuple) or isinstance(crop_size, list):
            assert len(crop_size) == 2, f"crop_size must be a tuple of length 2. Got {crop_size}!"
            self._crop_size = crop_size
    def __call__(self, data):
        keep_all_boxes = self._keep_all_boxes
        image_size = data['image'].shape[:2] # (H, W)
        crop_size = np.array(self._crop_size, dtype=np.int32) if self._crop_type == 'absolute' else (image_size * np.array(self._crop_size)).astype(np.int32)
        assert (crop_size < image_size).all(), f"crop_size must be smaller than image_size. Got {crop_size} and {image_size}!"
        polys = [poly for poly, ignore in zip(data['polys'], data['ignore_tags']) if not ignore]
        crop_found = False
        for _ in range(10):
            # randomly select a polygon 
            # find the the minimum and maximum coordinates for this po
            poly = random.choice(polys)
            points = np.maximum(np.round(poly).astype(np.int32), 0) # non-negative points
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            # find the minimum and maximum coordinates for the crop regio
            y_crop_min = max(0, y_max-crop_size[0] + 1)
            x_crop_min = max(0, x_max-crop_size[1] + 1)
            y_crop_max = min(image_size[0] - crop_size[0], y_min)
            x_crop_max = min(image_size[1] - crop_size[1], x_min)
            if y_crop_min > y_crop_max or x_crop_min > x_crop_max:
                continue
            for _ in range(10):
                y0 = np.random.randint(y_crop_min, y_crop_max + 1)
                x0 = np.random.randint(x_crop_min, x_crop_max + 1)
                assert y0+crop_size[0] <= image_size[0] and x0+crop_size[1] <= image_size[1]
                if x0<=x_min and y0<=y_min and x0+crop_size[1]>=x_max and y0+crop_size[0]>=y_max:
                    crop_found = True
                    break 
            if crop_found:
                break
        if not crop_found:
            print("Cannot find a crop region that contains a random text instance. use the original image instead")
            x0, y0 = 0, 0
            crop_size = np.array(image_size)
        elif keep_all_boxes:
            # Some boxes maybe cropped out so we need to modify th crop box coordinates to make them still contain the boxes
            num_modifications = 0
            modified = True

            while modified:
                modified, x0, y0, crop_size = self.adjust_crop(x0, y0, crop_size, polys)
                num_modifications += 1
                if num_modifications > 100:
                    raise ValueError(
                        "Cannot finished cropping adjustment within 100 tries (#instances {}). Please adjust the crop size to be larger!".format(
                            len(polys)
                        )
                    )
        # crop the image
        x0, y0 = int(x0), int(y0)
        crop_size = crop_size.astype('int32')
        data['image'] = data['image'][y0:y0 + crop_size[0], x0:x0 + crop_size[1]]
        # crop the polygons
        data['polys'] = np.array([poly - np.array([x0, y0]) for poly in data['polys']])
        # crop the bounding boxes
        if 'boxes' in data:
            data['boxes'] = np.array([box - np.array([x0, y0])  for box in data['boxes']])
        return data
        
    def adjust_crop(self, x0, y0, crop_size, instances, eps=1e-3):
        modified = False

        x1 = x0 + crop_size[1]
        y1 = y0 + crop_size[0]

        for bbox in instances:
            xmin, ymin = np.min(bbox, axis=0)[0], np.min(bbox, axis=0)[1]
            xmax, ymax = np.max(bbox, axis=0)[0], np.max(bbox, axis=0)[1]
            if xmin < x0 - eps and xmax > x0 + eps:
                #If the bounding box intersects with the left side of the crop box
                crop_size[1] += x0 - xmin
                x0 = xmin
                modified = True 

            if xmin < x1 - eps and xmax > x1 + eps:
                crop_size[1] += xmax - x1
                x1 = xmax
                modified = True

            if ymin < y0 - eps and ymax > y0 + eps:
                crop_size[0] += y0 - ymin
                y0 = ymin
                modified = True

            if ymin < y1 - eps and ymax > y1 + eps:
                crop_size[0] += ymax - y1
                y1 = ymax
                modified = True

        return modified, x0, y0, crop_size

class ResizePadImage:
    """
    Resize the image to the target size by keeping the aspect ratio the same without distortion.
    If the target size does not match with the aspect ratio of the image, pad the image with the pad_value.
    Args:
        target_size (int or float or tuple): size of the image to pad.
        size_type (str): type of the size. One of ['relative', 'absolute']
        pad_value (int): value to pad with.
    
    """
    def __init__(self, target_size, size_type='absolute', pad_value = 0, method = cv2.INTER_LINEAR):
        self._target_size = target_size
        self._size_type = size_type
        self._pad_value = pad_value
        assert  self._size_type in ['relative', 'absolute'], f"size_type must be one of ['relative', 'absolute']. Got {self._size_type}!"
        if isinstance(target_size, int):
            self._target_size = (target_size, target_size)
        elif isinstance(target_size, float):
            self._target_size = (target_size, target_size)
        elif isinstance(target_size, tuple):
            assert len(target_size) == 2, f"target_size must be a tuple of length 2. Got {target_size}!"
            self._target_size = target_size
    
    def __call__(self, data):
        # pad the image to the specified size
        image_size = data['image'].shape[:2] # (H, W, C)
        target_size = np.array(self._target_size, dtype=np.int32) if self._size_type == 'absolute' else (image_size * np.array(self._target_size)).astype(np.int32)
        aspect_ratio_image = image_size[1] / image_size[0] # h/w
        aspect_ratio_target = target_size[1] / target_size[0] # h/w
        if aspect_ratio_image < aspect_ratio_target:
            # resize based on the height
            scale = target_size[0] / image_size[0] 
            new_h = target_size[0]
            new_w = int(new_h * aspect_ratio_image)
        else:
            # resize based on the width
            scale = target_size[1] / image_size[1]
            new_w = target_size[1]
            new_h = int(new_w / aspect_ratio_image)
        
        # resize the image
        image_resized = cv2.resize(data['image'], (new_w, new_h))
        data['image'] = np.pad(image_resized,
                               (*tuple((0, cs - ds) for cs, ds in zip(target_size, image_resized.shape[:2])), (0, 0)),
                               mode='constant', constant_values = self._pad_value)

        # create image mask
        image_mask = np.ones(target_size, dtype=np.uint8)
        image_mask[:new_h, :new_w] =0
        data['image_mask'] = image_mask.astype(bool)
        data['image_size'] = [new_h, new_w]
        data['polys'] = data['polys'] * scale
        if 'boxes' in data:
            data['boxes'] = data['boxes'] * scale
        return data

# class RandomCropWithInstances:
#     """
#     Randomly crop the image so that the cropping region contains all the bboxes/instances in the original image
#     Args:
#         crop_size (int): size of the crop.
#         crop_type (str): type of the crop. One of ['relative', 'absolute']
#         crop_box (bool): whether to allow cropping bounding boxes that are partially outside the image. 
#     """
#     def __init__(self, crop_size: int, crop_type:str, crop_box = True):
#         self._crop_type = crop_type
#         self._crop_box = crop_box
#         assert  self._crop_type in ['relative', 'absolute'], f"crop_type must be one of ['relative', 'absolute']. Got {self._crop_type}!"
#         if isinstance(crop_size, int):
#             self._crop_size = (crop_size, crop_size)
#         elif isinstance(crop_size, float):
#             self._crop_size = (crop_size, crop_size)
#         elif isinstance(crop_size, tuple):
#             assert len(crop_size) == 2, f"crop_size must be a tuple of length 2. Got {crop_size}!"
#             self._crop_size = crop_size
#     def __call__(self, data):
#         image_size = data['image'].shape[:2] # (H, W, C)
#         crop_size = np.array(self._crop_size, dtype=np.int32) if self._crop_type == 'absolute' else (image_size * np.array(self._crop_size)).astype(np.int32)
#         assert (crop_size < image_size).all(), f"crop_size must be smaller than image_size. Got {crop_size} and {image_size}!"
#         polys = [poly for poly, ignore in zip(data['polys'], data['ignore_tags']) if not ignore]
#         # randomly select a polygon and find the the minimum and maximum coordinates for the crop box based on the center of the bounding box and the desired crop size.
#         bbox = random.choice(polys)
#         center_yx = np.mean(bbox, axis=0)[::-1]
#         min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
#         max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
#         max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

#         y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
#         x0 = np.random.randint(min_yx[1], max_yx[1] + 1)
#         # if some instance is cropped extend the box
#         if not self._crop_box:
#             num_modifications = 0
#             modified = True

#             # convert crop_size to float
#             crop_size = crop_size.astype(np.float32)
#             while modified:
#                 modified, x0, y0, crop_size = self.adjust_crop(x0, y0, crop_size, polys)
#                 num_modifications += 1
#                 if num_modifications > 100:
#                     raise ValueError(
#                         "Cannot finished cropping adjustment within 100 tries (#instances {}).".format(
#                             len(polys)
#                         )
#                     )
#         # crop the image
#         data['image'] = data['image'][y0:y0 + crop_size[0], x0:x0 + crop_size[1]]
#         # crop the polygons
#         data['polys'] = np.array([poly - np.array([x0, y0]) for poly in data['polys']])
#         # crop the bounding boxes
#         if 'boxes' in data:
#             data['boxes'] = np.array([box - np.array([x0, y0])  for box in data['boxes']])
#         return data
        
#     def adjust_crop(self, x0, y0, crop_size, instances, eps=1e-3):
#         modified = False

#         x1 = x0 + crop_size[1]
#         y1 = y0 + crop_size[0]

#         for bbox in instances:
#             xmin, ymin = np.min(bbox, axis=0)[0], np.min(bbox, axis=0)[1]
#             xmax, ymax = np.max(bbox, axis=0)[0], np.max(bbox, axis=0)[1]
#             if xmin < x0 - eps and xmax > x0 + eps:
#                 #If the bounding box intersects with the left side of the crop box
#                 crop_size[1] += x0 - xmin
#                 x0 = xmin
#                 modified = True 

#             if xmin < x1 - eps and xmax > x1 + eps:
#                 crop_size[1] += xmax - x1
#                 x1 = xmax
#                 modified = True

#             if ymin < y0 - eps and ymax > y0 + eps:
#                 crop_size[0] += y0 - ymin
#                 y0 = ymin
#                 modified = True

#             if ymin < y1 - eps and ymax > y1 + eps:
#                 crop_size[0] += ymax - y1
#                 y1 = ymax
#                 modified = True

#         return modified, x0, y0, crop_size

# class PadImage:
#     """
#     Pad the image to the specified size.
#     Args:
#         pad_size (int): size of the padding.
#         size_type (str): type of the padding. One of ['relative', 'absolute']
#     """
#     def __init__(self, pad_size, size_type='absolute'):
#         self._pad_size = pad_size
#         self._size_type = size_type
#         assert  self._size_type in ['relative', 'absolute'], f"size_type must be one of ['relative', 'absolute']. Got {self._size_type}!"
#         if isinstance(pad_size, int):
#             self._pad_size = (pad_size, pad_size)
#         elif isinstance(pad_size, float):
#             self._pad_size = (pad_size, pad_size)
#         elif isinstance(pad_size, tuple):
#             assert len(pad_size) == 2, f"pad_size must be a tuple of length 2. Got {pad_size}!"
#             self._pad_size = pad_size
    
#     def __call__(self, data):
#         # pad the image to the specified size
#         image_size = data['image'].shape[:2] # (H, W, C)
#         pad_size = np.array(self._pad_size, dtype=np.int32) if self._size_type == 'absolute' else (image_size * np.array(self._pad_size)).astype(np.int32)
#         data['image'] = np.pad(data['image'],
#                                (*tuple((0, cs - ds) for cs, ds in zip(pad_size, data['image'].shape[:2])), (0, 0)))

#         # create image mask
#         image_mask = np.ones(pad_size, dtype=np.uint8)
#         image_mask[:image_size[0], :image_size[1]] =0
#         data['image_mask'] = image_mask
#         return data
    
