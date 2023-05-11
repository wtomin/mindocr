"""
transforms for text detection tasks.
"""
import warnings
from typing import List

import json
import cv2
import pyclipper
from shapely.geometry import Polygon
import numpy as np

__all__ = ['DetLabelEncode', 'TESTRLabelEncode',  'PadTESTRLabel', 'BorderMap', 'ShrinkBinaryMap', 'expand_poly']

class TESTRLabelEncode:
    def __init__(self, text_keyname='transcription', bbox_keyname='points', 
                 polygon_keyname = 'polys'):
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
        txt_tags = np.array(txt_tags, dtype=bool)

        data['polys'] = polys
        data['boxes'] = boxes
        data['texts'] = txts
        data['rec_ids'] = rec_ids
        data['ignore_tags'] = txt_tags
        data['gt_classes'] = np.zeros(len(txts), dtype=np.int32)
        return data
class PadTESTRLabel:
    def __init__(self, pad_len=20):
        self.pad_len = pad_len
    def __call__(self, data):
        nBox = len(data['polys'])
        if nBox<self.pad_len:
            data['polys'] = np.concatenate([data['polys'], np.zeros((self.pad_len-nBox, 16, 2), dtype=np.float32)], axis=0)
            data['boxes'] = np.concatenate([data['boxes'], np.zeros((self.pad_len-nBox, 4, 2), dtype=np.float32)], axis=0)
            data['texts'] = data['texts'] + ['###']*(self.pad_len-nBox)
            data['rec_ids'] = np.concatenate([data['rec_ids'], 96*np.ones((self.pad_len-nBox, 25), dtype=np.int32)], axis=0)
            data['ignore_tags'] = np.concatenate([data['ignore_tags'], np.ones((self.pad_len-nBox), dtype=bool)], axis=0)
            data['gt_classes'] = np.concatenate([data['gt_classes'], np.ones((self.pad_len-nBox), dtype=np.int32)], axis=0)
        else:
            data['polys'] = data['polys'][:self.pad_len]
            data['boxes'] = data['boxes'][:self.pad_len]
            data['texts'] = data['texts'][:self.pad_len]
            data['rec_ids'] = data['rec_ids'][:self.pad_len]
            data['ignore_tags'] = data['ignore_tags'][:self.pad_len]
            data['gt_classes'] = data['gt_classes'][:self.pad_len]
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
        self._thresh_min = thresh_min
        self._thresh_max = thresh_max
        self._dist_coef = 1 - shrink_ratio ** 2

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
    def __init__(self, min_text_size=8, shrink_ratio=0.4):
        self._min_text_size = min_text_size
        self._dist_coef = 1 - shrink_ratio ** 2

    def __call__(self, data):
        gt = np.zeros(data['image'].shape[:2], dtype=np.float32)
        mask = np.ones(data['image'].shape[:2], dtype=np.float32)

        if len(data['polys']):
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


def expand_poly(poly, distance: float, joint_type=pyclipper.JT_ROUND) -> List[list]:
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(poly, joint_type, pyclipper.ET_CLOSEDPOLYGON)
    return offset.Execute(distance)