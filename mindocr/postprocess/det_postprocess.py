from typing import Tuple, Union
import cv2
import numpy as np
import mindspore.numpy as np
from shapely.geometry import Polygon
from mindspore import ops, Tensor
from mindspore import Tensor

from ..data.transforms.det_transforms import expand_poly

__all__ = ['DBPostprocess', 'TESTRPostprocess']

class TESTRPostprocess:
    def __init__(self, test_score_threshold, use_polygon = True) -> None:
        self.test_score_threshold = test_score_threshold
        self.use_polygon = use_polygon
    
    def __call__(self, preds, image_sizes, out_image_sizes):
        ctrl_point_cls = preds["pred_logits"]
        ctrl_point_coord = preds["pred_ctrl_points"]
        text_pred = preds["pred_texts"]
        results = self.get_inferences(ctrl_point_cls, ctrl_point_coord, text_pred, image_sizes)
        processed_results = []
        for results_per_image, out_image_size in zip(results, out_image_sizes):
            height, width = out_image_size
            res = self.rescale_points(results_per_image, height, width)
            processed_results.append(res)
        return processed_results

    def get_inferences(self, ctrl_point_cls, ctrl_point_coord, text_pred, image_sizes):
        assert len(ctrl_point_cls) == len(image_sizes)
        results = []

        text_pred = ops.softmax(text_pred, -1)
        prob = ctrl_point_cls.mean(-2).sigmoid()
        scores, labels = prob.max(-1)

        for scores_per_image, labels_per_image, ctrl_point_per_image, text_per_image, image_size in zip(
            scores, labels, ctrl_point_coord, text_pred, image_sizes
        ):
            selector = scores_per_image >= self.test_score_threshold
            scores_per_image = scores_per_image[selector]
            labels_per_image = labels_per_image[selector]
            ctrl_point_per_image = ctrl_point_per_image[selector]
            text_per_image = text_per_image[selector]
            ctrl_point_per_image[..., 0] *= image_size[1]
            ctrl_point_per_image[..., 1] *= image_size[0]
            if self.use_polygon:
                polygons = ctrl_point_per_image.flatten(1)
            else:
               beziers = ctrl_point_per_image.flatten(1)
            _, topi = text_per_image.topk(1)
            recs = topi.squeeze(-1)
            results.append({'polygons': polygons, 'beziers': beziers, 
                            'recs': recs, 'scores': scores_per_image, 'labels': labels_per_image,
                            'image_size': image_size})
        return results

    def rescale_points(self, results, output_height, output_width):
        """
        scale alignment for bezier points and polygons
        """
        scale_x, scale_y = (output_width / results['image_size'][1], output_height / results['image_size'][0])

        # scale bezier points
        if 'beziers' in results:
            beziers = results['beziers']
            # scale and clip in place
            h, w = results['image_size']
            np.clip(beziers[:, 0], 0, w, out=beziers[:, 0])
            np.clip(beziers[:, 1], 0, h, out=beziers[:, 1])
            np.clip(beziers[:, 6], 0, w, out=beziers[:, 6])
            np.clip(beziers[:, 7], 0, h, out=beziers[:, 7])
            np.clip(beziers[:, 8], 0, w, out=beziers[:, 8])
            np.clip(beziers[:, 9], 0, h, out=beziers[:, 9])
            np.clip(beziers[:, 14], 0, w, out=beziers[:, 14])
            np.clip(beziers[:, 15], 0, h, out=beziers[:, 15])
            beziers[:, 0::2] *= scale_x
            beziers[:, 1::2] *= scale_y

        if 'polygons' in results:
            polygons = results['polygons']
            polygons[:, 0::2] *= scale_x
            polygons[:, 1::2] *= scale_y

        return results

class DBPostprocess:
    def __init__(self, binary_thresh=0.3, box_thresh=0.7, max_candidates=1000, expand_ratio=1.5,
                 output_polygon=False, pred_name='binary'):
        self._min_size = 3
        self._binary_thresh = binary_thresh
        self._box_thresh = box_thresh
        self._max_candidates = max_candidates
        self._expand_ratio = expand_ratio
        self._out_poly = output_polygon
        self._name = pred_name
        self._names = {'binary': 0, 'thresh': 1, 'thresh_binary': 2}

    def __call__(self, pred, **kwargs):
        """
        pred (Union[Tensor, Tuple[Tensor], np.ndarray]):
            binary: text region segmentation map, with shape (N, 1, H, W)
            thresh: [if exists] threshold prediction with shape (N, 1, H, W) (optional)
            thresh_binary: [if exists] binarized with threshold, (N, 1, H, W) (optional)
        Returns:
            result (dict) with keys:
                polygons: np.ndarray of shape (N, K, 4, 2) for the polygons of objective regions if region_type is 'quad'
                scores: np.ndarray of shape (N, K), score for each box
        """
        if isinstance(pred, tuple):
            pred = pred[self._names[self._name]]
        if isinstance(pred, Tensor):
            pred = pred.asnumpy()
        pred = pred.squeeze(1)

        segmentation = pred >= self._binary_thresh

        # FIXME: dest_size is supposed to be the original image shape (pred.shape -> batch['shape'])
        dest_size = np.array(pred.shape[:0:-1])  # w, h order
        scale = dest_size / np.array(pred.shape[:0:-1])

        # TODO:
        # FIXME: output as dict, keep consistent return format to recognition
        return [self._extract_preds(pr, segm, scale, dest_size) for pr, segm in zip(pred, segmentation)]

    def _extract_preds(self, pred, bitmap, scale, dest_size):
        outs = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:  # FIXME: update to OpenCV 4.x and delete this
            _, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        polys, scores = [], []
        for contour in contours[:self._max_candidates]:
            contour = contour.squeeze(1)
            score = self._calc_score(pred, bitmap, contour)
            if score < self._box_thresh:
                continue

            if self._out_poly:
                epsilon = 0.005 * cv2.arcLength(contour, closed=True)
                points = cv2.approxPolyDP(contour, epsilon, closed=True).squeeze(1)
                if points.shape[0] < 4:
                    continue
            else:
                points, min_side = self._fit_box(contour)
                if min_side < self._min_size:
                    continue

            poly = Polygon(points)
            poly = np.array(expand_poly(points, distance=poly.area * self._expand_ratio / poly.length))
            if self._out_poly and len(poly) > 1:
                continue
            poly = poly.reshape(-1, 2)

            _box, min_side = self._fit_box(poly)
            if min_side < self._min_size + 2:
                continue
            if not self._out_poly:
                poly = _box

            # TODO: an alternative solution to avoid calling self._fit_box twice:
            # box = Polygon(points)
            # box = np.array(expand_poly(points, distance=box.area * self._expand_ratio / box.length, joint_type=pyclipper.JT_MITER))
            # assert box.shape[0] == 4, print(f'box shape is {box.shape}')

            # predictions may not be the same size as the input image => scale it
            polys.append(np.clip(np.round(poly * scale), 0, dest_size - 1).astype(np.int16))
            scores.append(score)

        if self._out_poly:
            return polys, scores
        return np.array(polys), np.array(scores).astype(np.float32)

    @staticmethod
    def _fit_box(contour):
        """
        Finds a minimum rotated rectangle enclosing the contour.
        """
        # box = cv2.minAreaRect(contour)  # returns center of a rect, size, and angle
        # # TODO: does the starting point really matter?
        # points = np.roll(cv2.boxPoints(box), -1, axis=0)  # extract box points from a rotated rectangle
        # return points, min(box[1])
        # box = cv2.minAreaRect(contour)  # returns center of a rect, size, and angle
        # # TODO: does the starting point really matter?
        # points = np.roll(cv2.boxPoints(box), -1, axis=0)  # extract box points from a rotated rectangle
        # return points, min(box[1])

        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        # index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    @staticmethod
    def _calc_score(pred, mask, contour):
        """
        calculates score (mean value) of a prediction inside a given contour.
        """
        min_vals = np.clip(np.floor(np.min(contour, axis=0)), 0, np.array(pred.shape[::-1]) - 1).astype(np.int32)
        max_vals = np.clip(np.ceil(np.max(contour, axis=0)), 0, np.array(pred.shape[::-1]) - 1).astype(np.int32)
        return cv2.mean(pred[min_vals[1]:max_vals[1] + 1, min_vals[0]:max_vals[0] + 1],
                        mask[min_vals[1]:max_vals[1] + 1, min_vals[0]:max_vals[0] + 1].astype(np.uint8))[0]
