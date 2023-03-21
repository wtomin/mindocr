"""
transforms for text detection tasks.
"""
import random
import warnings
from typing import List, Tuple, Optional

import json
import cv2
import pyclipper
from shapely.geometry import Polygon
import numpy as np

__all__ = ['DetLabelEncode', 'TESTRLabelEncode', 'BorderMap', 'ShrinkBinaryMap', 'expand_poly']

class TESTRLabelEncode:
    def __init__(self, text_keyname='transcription', bbox_keyname='points', 
                 polygon_keyname = 'polygons', **kwargs):
        self.text_keyname = text_keyname
        self.bbox_keyname = bbox_keyname
        self.polygon_keyname = polygon_keyname
        self.CTLABELS =  [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
        self.vocabulary_size = len(self.CTLABELS)+1
        
    def _decode_string_to_rec_ids(self, rec):
        rec_ids = []
        for char in rec:
            if char=='口':
                rec_ids.append(self.vocabulary_size-1)
            elif char in self.CTLABELS:
                rec_ids.append(self.CTLABELS.index(char))
            else:
                raise ValueError("invalid character: {} not included in vocabulary".format(char))
        LEN =25
        assert len(rec_ids)<=LEN, "rec_ids length should be less than {}".format(LEN)
        rec_ids = rec_ids + [self.vocabulary_size]*(LEN-len(rec_ids))
        return rec_ids

    
    # def expand_points_num(self, boxes):
    #     max_points_num = 0
    #     for box in boxes:
    #         if len(box) > max_points_num:
    #             max_points_num = len(box)
    #     ex_boxes = []
    #     for box in boxes:
    #         ex_box = box + [box[-1]] * (max_points_num - len(box))
    #         ex_boxes.append(ex_box)
    #     return ex_boxes

    def __call__(self, data):
        """
        required keys:
            label (str): string containgin points and transcription in json format
        added keys:
            polys (np.ndarray): polygon boxes in an image, each polygon is represented by points
                            in shape [num_polygons, num_points, 2]
            boxes (np.ndarray): bounding boxes in an image, each box is represented by points
            texts (List(str)): text string
            rec_ids (List(int)): text string in id format
            gt_classes (np.ndarray): text class, all zeros indicating they are text objects.
            ignore_tags (np.ndarray[bool]): indicators for ignorable texts (e.g., '###')
        """
        label = data['label']
        label = json.loads(label)
        nBox = len(label)
        boxes, polys, txts, rec_ids, txt_tags = [], [], [], [], []
        for bno in range(0, nBox):
            box = label[bno][self.bbox_keyname]
            poly = label[bno][self.polygon_keyname]
            txt = label[bno][self.text_keyname]
            rec_id = self._decode_string_to_rec_ids(txt)
            boxes.append(box)
            txts.append(txt)
            rec_ids.append(rec_id)
            polys.append(poly)
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        if len(boxes) == 0:
            return None
        # boxes = self.expand_points_num(boxes)
        # polys = self.expand_points_num(polys)
        boxes = np.array(boxes, dtype=np.float32)
        polys = np.array(polys, dtype=np.float32)
        rec_ids = np.array(rec_ids, dtype=np.int32)
        txt_tags = np.array(txt_tags, dtype=np.bool)

        data['polys'] = polys
        data['boxes'] = boxes
        data['texts'] = txts
        data['rec_ids'] = rec_ids
        data['ignore_tags'] = txt_tags
        data['gt_classes'] = np.zeros(len(txts), dtype=np.int32)
        return data

class DetLabelEncode:
    def __init__(self, **kwargs):
        pass

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def expand_points_num(self, boxes):
        max_points_num = 0
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)
        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes

    def __call__(self, data):
        """
        required keys:
            label (str): string containgin points and transcription in json format
        added keys:
            polys (np.ndarray): polygon boxes in an image, each polygon is represented by points
                            in shape [num_polygons, num_points, 2]
            texts (List(str)): text string
            ignore_tags (np.ndarray[bool]): indicators for ignorable texts (e.g., '###')
        """
        label = data['label']
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]['points']
            txt = label[bno]['transcription']
            boxes.append(box)
            txts.append(txt)
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        if len(boxes) == 0:
            return None
        boxes = self.expand_points_num(boxes)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool)

        data['polys'] = boxes
        data['texts'] = txts
        data['ignore_tags'] = txt_tags
        return data


# FIXME:
#  RuntimeWarning: invalid value encountered in sqrt result = np.sqrt(a_sq * b_sq * sin_sq / c_sq)
#  RuntimeWarning: invalid value encountered in true_divide cos = (a_sq + b_sq - c_sq) / (2 * np.sqrt(a_sq * b_sq))
warnings.filterwarnings("ignore")
class BorderMap:
    def __init__(self, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
        self._shrink_ratio = shrink_ratio
        self._thresh_min = thresh_min
        self._thresh_max = thresh_max
        self._dist_coef = 1 - self._shrink_ratio ** 2

    def __call__(self, data):
        border = np.zeros(data['image'].shape[:2], dtype=np.float32)
        mask = np.zeros(data['image'].shape[:2], dtype=np.float32)

        for i in range(len(data['polys'])):
            if not data['ignore_tags'][i]:
                self._draw_border(data['polys'][i], border, mask=mask)
        border = border * (self._thresh_max - self._thresh_min) + self._thresh_min

        data['thresh_map'] = border
        data['thresh_mask'] = mask
        return data

    def _draw_border(self, np_poly, border, mask):
        # draw mask
        poly = Polygon(np_poly)
        distance = self._dist_coef * poly.area / poly.length
        padded_polygon = np.array(expand_poly(np_poly, distance)[0], dtype=np.int32)
        cv2.fillPoly(mask, [padded_polygon], 1.0)

        # draw border
        min_vals, max_vals = np.min(padded_polygon, axis=0), np.max(padded_polygon, axis=0)
        width, height = max_vals - min_vals + 1
        np_poly -= min_vals

        xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = [self._distance(xs, ys, p1, p2) for p1, p2 in zip(np_poly, np.roll(np_poly, 1, axis=0))]
        distance_map = np.clip(np.array(distance_map, dtype=np.float32) / distance, 0, 1).min(axis=0)   # NOQA

        min_valid = np.clip(min_vals, 0, np.array(border.shape[::-1]) - 1)  # shape reverse order: w, h
        max_valid = np.clip(max_vals, 0, np.array(border.shape[::-1]) - 1)

        border[min_valid[1]: max_valid[1] + 1, min_valid[0]: max_valid[0] + 1] = np.fmax(
            1 - distance_map[min_valid[1] - min_vals[1]: max_valid[1] - max_vals[1] + height,
                             min_valid[0] - min_vals[0]: max_valid[0] - max_vals[0] + width],
            border[min_valid[1]: max_valid[1] + 1, min_valid[0]: max_valid[0] + 1]
        )

    @staticmethod
    def _distance(xs, ys, point_1, point_2):
        """
        compute the distance from each point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        """
        a_sq = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        b_sq = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        c_sq = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cos = (a_sq + b_sq - c_sq) / (2 * np.sqrt(a_sq * b_sq))
        sin_sq = np.nan_to_num(1 - np.square(cos))
        result = np.sqrt(a_sq * b_sq * sin_sq / c_sq)

        result[cos >= 0] = np.sqrt(np.fmin(a_sq, b_sq))[cos >= 0]
        return result



class ShrinkBinaryMap:
    """
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    """
    def __init__(self, min_text_size=8, shrink_ratio=0.4, train=True):
        self._min_text_size = min_text_size
        self._shrink_ratio = shrink_ratio
        self._train = train
        self._dist_coef = 1 - self._shrink_ratio ** 2

    def __call__(self, data):
        gt = np.zeros(data['image'].shape[:2], dtype=np.float32)
        mask = np.ones(data['image'].shape[:2], dtype=np.float32)

        if len(data['polys']):
            if self._train:
                self._validate_polys(data)

            for i in range(len(data['polys'])):
                min_side = min(np.max(data['polys'][i], axis=0) - np.min(data['polys'][i], axis=0))

                if data['ignore_tags'][i] or min_side < self._min_text_size:
                    cv2.fillPoly(mask, [data['polys'][i].astype(np.int32)], 0)
                    data['ignore_tags'][i] = True
                else:
                    poly = Polygon(data['polys'][i])
                    shrunk = expand_poly(data['polys'][i], distance=-self._dist_coef * poly.area / poly.length)

                    if shrunk:
                        cv2.fillPoly(gt, [np.array(shrunk[0], dtype=np.int32)], 1)
                    else:
                        cv2.fillPoly(mask, [data['polys'][i].astype(np.int32)], 0)
                        data['ignore_tags'][i] = True

        data['binary_map'] = np.expand_dims(gt, axis=0)
        data['mask'] = mask
        return data

    @staticmethod
    def _validate_polys(data):
        data['polys'] = np.clip(data['polys'], 0, np.array(data['image'].shape[1::-1]) - 1)  # shape reverse order: w, h

        for i in range(len(data['polys'])):
            poly = Polygon(data['polys'][i])
            if poly.area < 1:
                data['ignore_tags'][i] = True
            if not poly.exterior.is_ccw:
                data['polys'][i] = data['polys'][i][::-1]


# random crop algorithm similar to https://github.com/argman/EAST
# TODO: check randomness, seems to crop the same region every time
class EastRandomCropData:
    """
    Code adopted from https://github1s.com/WenmuZhou/DBNet.pytorch/blob/master/data_loader/modules/random_crop_data.py

    Randomly select a region and crop images to a target size and make sure
    to contain text region. This transform may break up text instances, and for
    broken text instances, we will crop it's bbox and polygon coordinates. This
    transform is recommend to be used in segmentation-based network.
    """
    def __init__(self, size=(640, 640), max_tries=50, min_crop_side_ratio=0.1, require_original_image=False,
                 keep_ratio=True):
        self.size = size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.require_original_image = require_original_image
        self.keep_ratio = keep_ratio

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'polys':,'texts':,'ignore_tags':}
        :return:
        """
        im = data['image']
        text_polys = data['polys']
        ignore_tags = data['ignore_tags']
        texts = data['texts']
        all_care_polys = [text_polys[i] for i, tag in enumerate(ignore_tags) if not tag]
        # 计算crop区域
        crop_x, crop_y, crop_w, crop_h = self.crop_area(im, all_care_polys)
        print('crop area: ', crop_x, crop_y, crop_w, crop_h)
        # crop 图片 保持比例填充
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        if self.keep_ratio:
            if len(im.shape) == 3:
                padimg = np.zeros((self.size[1], self.size[0], im.shape[2]), im.dtype)
            else:
                padimg = np.zeros((self.size[1], self.size[0]), im.dtype)
            padimg[:h, :w] = cv2.resize(im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
            img = padimg
        else:
            img = cv2.resize(im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], tuple(self.size))
        # crop 文本框
        text_polys_crop = []
        ignore_tags_crop = []
        texts_crop = []
        for poly, text, tag in zip(text_polys, texts, ignore_tags):
            poly = ((poly - (crop_x, crop_y)) * scale).tolist()
            if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                text_polys_crop.append(poly)
                ignore_tags_crop.append(tag)
                texts_crop.append(text)
        data['image'] = img
        data['polys'] = np.float32(text_polys_crop)
        data['ignore_tags'] = ignore_tags_crop
        data['texts'] = texts_crop
        return data

    def is_poly_in_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    def is_poly_outside_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions, max_size):
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    def crop_area(self, im, text_polys):
        h, w = im.shape[:2]
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in text_polys:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                # area too small
                continue
            num_poly_in_rect = 0
            for poly in text_polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h


class PSERandomCrop:
    """
    Code adopted from https://github1s.com/WenmuZhou/DBNet.pytorch/blob/master/data_loader/modules/random_crop_data.py
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        imgs = data['imgs']

        h, w = imgs[0].shape[0:2]
        th, tw = self.size
        if w == tw and h == th:
            return imgs

        # label中存在文本实例，并且按照概率进行裁剪，使用threshold_label_map控制
        if np.max(imgs[2]) > 0 and random.random() > 3 / 8:
            # 文本实例的左上角点
            tl = np.min(np.where(imgs[2] > 0), axis=1) - self.size
            tl[tl < 0] = 0
            # 文本实例的右下角点
            br = np.max(np.where(imgs[2] > 0), axis=1) - self.size
            br[br < 0] = 0
            # 保证选到右下角点时，有足够的距离进行crop
            br[0] = min(br[0], h - th)
            br[1] = min(br[1], w - tw)

            for _ in range(50000):
                i = random.randint(tl[0], br[0])
                j = random.randint(tl[1], br[1])
                # 保证shrink_label_map有文本
                if imgs[1][i:i + th, j:j + tw].sum() <= 0:
                    continue
                else:
                    break
        else:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)

        # return i, j, th, tw
        for idx in range(len(imgs)):
            if len(imgs[idx].shape) == 3:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
            else:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw]
        data['imgs'] = imgs
        return data

# class RandomCropWithInstance:
#     """
#     Code adopted from https://github.com/mlpc-ucsd/TESTR/blob/43df00d60efb1fcd71372dec70980d1cf54ad1a9/adet/data/augmentation.py
#     Instance Aware random cropping

#     Args:
#         crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
#         crop_size (tuple[float, float]): two floats, explained below.
#     - "relative": crop a (H * crop_size[0], W * crop_size[1]) region from an input image of
#         size (H, W). crop size should be in (0, 1]
#     - "relative_range": uniformly sample two values from [crop_size[0], 1]
#         and [crop_size[1]], 1], and use them as in "relative" crop type.
#     - "absolute" crop a (crop_size[0], crop_size[1]) region from input image.
#         crop_size must be smaller than the input image size.
#     - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
#         [crop_size[0], min(H, crop_size[1])] and W_crop in [crop_size[0], min(W, crop_size[1])].
#         Then crop a region (H_crop, W_crop).

#     """
#     def __init__(self, crop_type:str, crop_size: Tuple[float, float], crop_instance=True):
#         self.crop_type = crop_type
#         self.crop_size = crop_size
#         self.crop_instance = crop_instance
#         self.input_args = ("image", "boxes")
#         assert self.crop_type  in ["relative_range", "relative", "absolute", "absolute_range"], "Expect crop_type to be one of relative_range, relative, absolute, absolute_range, but get {}".format(self.crop_type)
    
#     def get_crop_size(self, image_size):
#         """
#         Args:
#             image_size (tuple): height, width
#         Returns:
#             crop_size (tuple): height, width in absolute pixels
#         """
#         h, w = image_size
#         if self.crop_type == "relative":
#             ch, cw = self.crop_size
#             return int(h * ch + 0.5), int(w * cw + 0.5)
#         elif self.crop_type == "relative_range":
#             crop_size = np.asarray(self.crop_size, dtype=np.float32)
#             ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
#             return int(h * ch + 0.5), int(w * cw + 0.5)
#         elif self.crop_type == "absolute":
#             return (min(self.crop_size[0], h), min(self.crop_size[1], w))
#         elif self.crop_type == "absolute_range":
#             assert self.crop_size[0] <= self.crop_size[1]
#             ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
#             cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
#             return ch, cw
#         else:
#             raise NotImplementedError("Unknown crop type {}".format(self.crop_type))
   
#     def get_transform(self, img, boxes):
#         image_size = img.shape[:2]
#         crop_size = self.get_crop_size(image_size)
#         return self.gen_crop_transform_with_instance(crop_size, image_size, boxes)
    
#     def gen_crop_transform_with_instance(self, crop_size, image_size, instances, crop_box =True):
#         """
#         Generate a CropTransform so that the cropping region contains
#         the center of the given instance.
#         Args:
#             crop_size (tuple): h, w in pixels
#             image_size (tuple): h, w
#             instance (dict): an annotation dict of one instance
#         """
#         crop_box = self.crop_instance
#         bbox = random.choice(instances)
#         crop_size = np.asarray(crop_size, dtype=np.int32)
#         center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
#         assert (
#             image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
#         ), "The annotation bounding box is outside of the image!"
#         assert (
#             image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
#         ), "Crop size is larger than image size!"

#         min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
#         max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
#         max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

#         y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
#         x0 = np.random.randint(min_yx[1], max_yx[1] + 1)

#         # if some instance is cropped extend the box
#         if not crop_box:
#             num_modifications = 0
#             modified = True

#             # convert crop_size to float
#             crop_size = crop_size.astype(np.float32)
#             while modified:
#                 modified, x0, y0, crop_size = adjust_crop(x0, y0, crop_size, instances)
#                 num_modifications += 1
#                 if num_modifications > 100:
#                     raise ValueError(
#                         "Cannot finished cropping adjustment within 100 tries (#instances {}).".format(
#                             len(instances)
#                         )
#                     )
#         return CropTransform(*map(int, (x0, y0, crop_size[1], crop_size[0])))
    
#     def __call__(self, data):
#         image = data['image']
#         boxes = data['polys']
#         crop_size = self.get_transform(image, boxes)

# class CropTransform(object):
#     """
#     Code adopted from https://github.com/facebookresearch/fvcore/blob/51092b5515cbb493f73de079743dd6b11cc4bbf1/fvcore/transforms/transform.py#L643
#     """
#     def __init__(
#         self,
#         x0: int,
#         y0: int,
#         w: int,
#         h: int,
#         orig_w: Optional[int] = None,
#         orig_h: Optional[int] = None,
#     ):
#         """
#         Args:
#             x0, y0, w, h (int): crop the image(s) by img[y0:y0+h, x0:x0+w].
#             orig_w, orig_h (int): optional, the original width and height
#                 before cropping. Needed to make this transform invertible.
#         """
#         super().__init__()
#         self._set_attributes(locals())

#     def apply_image(self, img: np.ndarray) -> np.ndarray:
#         """
#         Crop the image(s).
#         Args:
#             img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
#                 of type uint8 in range [0, 255], or floating point in range
#                 [0, 1] or [0, 255].
#         Returns:
#             ndarray: cropped image(s).
#         """
#         if len(img.shape) <= 3:
#             return img[self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w]
#         else:
#             return img[..., self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w, :]

#     def apply_coords(self, coords: np.ndarray) -> np.ndarray:
#         """
#         Apply crop transform on coordinates.
#         Args:
#             coords (ndarray): floating point array of shape Nx2. Each row is
#                 (x, y).
#         Returns:
#             ndarray: cropped coordinates.
#         """
#         coords[:, 0] -= self.x0
#         coords[:, 1] -= self.y0
#         return coords

#     def apply_polygons(self, polygons: list) -> list:
#         """
#         Apply crop transform on a list of polygons, each represented by a Nx2 array.
#         It will crop the polygon with the box, therefore the number of points in the
#         polygon might change.
#         Args:
#             polygon (list[ndarray]): each is a Nx2 floating point array of
#                 (x, y) format in absolute coordinates.
#         Returns:
#             ndarray: cropped polygons.
#         """
#         import shapely.geometry as geometry

#         # Create a window that will be used to crop
#         crop_box = geometry.box(
#             self.x0, self.y0, self.x0 + self.w, self.y0 + self.h
#         ).buffer(0.0)

#         cropped_polygons = []

#         for polygon in polygons:
#             polygon = geometry.Polygon(polygon).buffer(0.0)
#             # polygon must be valid to perform intersection.
#             if not polygon.is_valid:
#                 continue
#             cropped = polygon.intersection(crop_box)
#             if cropped.is_empty:
#                 continue
#             if isinstance(cropped, geometry.collection.BaseMultipartGeometry):
#                 cropped = cropped.geoms
#             else:
#                 cropped = [cropped]
#             # one polygon may be cropped to multiple ones
#             for poly in cropped:
#                 # It could produce lower dimensional objects like lines or
#                 # points, which we want to ignore
#                 if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
#                     continue
#                 coords = np.asarray(poly.exterior.coords)
#                 # NOTE This process will produce an extra identical vertex at
#                 # the end. So we remove it. This is tested by
#                 # `tests/test_data_transform.py`
#                 cropped_polygons.append(coords[:-1])
#         return [self.apply_coords(p) for p in cropped_polygons]

#     # def inverse(self) -> Transform:
#     #     assert (
#     #         self.orig_w is not None and self.orig_h is not None
#     #     ), "orig_w, orig_h are required for CropTransform to be invertible!"
#     #     pad_x1 = self.orig_w - self.x0 - self.w
#     #     pad_y1 = self.orig_h - self.y0 - self.h
#     #     return PadTransform(
#     #         self.x0, self.y0, pad_x1, pad_y1, orig_w=self.w, orig_h=self.h
#     #     )


# def adjust_crop(x0, y0, crop_size, instances, eps=1e-3):
#     modified = False

#     x1 = x0 + crop_size[1]
#     y1 = y0 + crop_size[0]

#     for bbox in instances:

#         if bbox[0] < x0 - eps and bbox[2] > x0 + eps:
#             crop_size[1] += x0 - bbox[0]
#             x0 = bbox[0]
#             modified = True

#         if bbox[0] < x1 - eps and bbox[2] > x1 + eps:
#             crop_size[1] += bbox[2] - x1
#             x1 = bbox[2]
#             modified = True

#         if bbox[1] < y0 - eps and bbox[3] > y0 + eps:
#             crop_size[0] += y0 - bbox[1]
#             y0 = bbox[1]
#             modified = True

#         if bbox[1] < y1 - eps and bbox[3] > y1 + eps:
#             crop_size[0] += bbox[3] - y1
#             y1 = bbox[3]
#             modified = True

#     return modified, x0, y0, crop_size
def expand_poly(poly, distance: float, joint_type=pyclipper.JT_ROUND) -> List[list]:
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(poly, joint_type, pyclipper.ET_CLOSEDPOLYGON)
    return offset.Execute(distance)

