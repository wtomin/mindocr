import numpy as np
import imgaug
import imgaug.augmenters as iaa

__all__ = ['IaaAugment']


class IaaAugment:
    def __init__(self, **augments):
        self._augmenter = iaa.Sequential([getattr(iaa, aug)(**args) for aug, args in augments.items()])

    def __call__(self, data):
        aug = self._augmenter.to_deterministic()    # to augment an image and its keypoints identically
        for key in ['polys', 'boxes']:
            if key in data:
                new_array = []
                for poly in data[key]:
                    kps = imgaug.KeypointsOnImage([imgaug.Keypoint(p[0], p[1]) for p in poly], shape=data['image'].shape)
                    kps = aug.augment_keypoints(kps)
                    new_array.append(np.array([[kp.x, kp.y] for kp in kps.keypoints]))

                data[key] = np.array(new_array) if isinstance(data[key], np.ndarray) else new_array
        
        data['image'] = aug.augment_image(data['image'])

        return data
