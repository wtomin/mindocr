from typing import List, Tuple, Union
import numpy as np
from mindspore import ops

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
