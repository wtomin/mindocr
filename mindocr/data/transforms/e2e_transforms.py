"""
transforms for text spotting tasks.
"""
import json
import random
import cv2
import numpy as np

__all__ = [
    "TESTRLabelEncode", 
    "PadTESTRLabel",
    "RandomCropWithInstances",
    "ResizePadImage"]

class TESTRLabelEncode:
    def __init__(self, text_keyname='transcription', bbox_keyname='points', 
                 polygon_keyname = 'polys', **kwargs):
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
        boxes = np.array(boxes, dtype=np.float32)
        polys = np.array(polys, dtype=np.float32)
        rec_ids = np.array(rec_ids, dtype=np.int32)
        txt_tags = np.array(txt_tags, dtype=bool)

        data['polys'] = polys
        data['boxes'] = boxes
        # data['texts'] = txts
        data['rec_ids'] = rec_ids
        data['ignore_tags'] = txt_tags
        data['gt_classes'] = np.zeros(len(txts), dtype=np.int32)
        return data
class PadTESTRLabel:
    def __init__(self, target_len=30, affect_keys=['polys', 'boxes', 'texts', 'rec_ids', 'ignore_tags', 'gt_classes'],
                 **kwargs):
        self.target_len = target_len
        self._affect_keys = affect_keys
        assert 'polys' in self._affect_keys, "polys should be included in affect_keys"
    def __call__(self, data):
        nBox = len(data['polys'])
        assert nBox>0, "nBox should be greater than 0"
        if nBox>self.target_len:
            print(f"nBox {nBox} is greater than target_len {self.target_len}, so we will truncate the data")
            for key in self._affect_keys:
                data[key] = data[key][:self.target_len]
        else:
            for key in self._affect_keys:
                if key == 'polys':
                    data['polys'] = np.concatenate([data['polys'].astype(np.float32), np.zeros((self.target_len-nBox, 16, 3), dtype=np.float32)], axis=0)
                elif key == 'boxes':
                    data['boxes'] = np.concatenate([data['boxes'].astype(np.float32), np.zeros((self.target_len-nBox, 4, 2), dtype=np.float32)], axis=0)
                elif key == 'texts':
                    data['texts'] = data['texts'] + ['###']*(self.target_len-nBox)
                elif    key == 'rec_ids':
                    data['rec_ids'] = np.concatenate([data['rec_ids'].astype(np.int32), 96*np.ones((self.target_len-nBox, 25), dtype=np.int32)], axis=0)
                elif key == 'ignore_tags':
                    data['ignore_tags'] = np.concatenate([data['ignore_tags'], np.ones((self.target_len-nBox), dtype=bool)], axis=0)
                elif key == 'gt_classes':
                    data['gt_classes'] = np.concatenate([data['gt_classes'].astype(np.int32), np.ones((self.target_len-nBox), dtype=np.int32)], axis=0)
        for key in self._affect_keys:
            assert len(data[key]) == self.target_len, "length of {} should be {}".format(key, self.target_len)
        return data
    
class RandomCropWithInstances:
    """
    Randomly crop the image so that the cropping region contains all the bboxes/instances in the original image
    Args:
        crop_size (int): size of the crop.
        crop_type (str): type of the crop. One of ['relative', 'absolute']
        crop_box (bool): whether to allow cropping bounding boxes that are partially outside the image. 
    """
    def __init__(self, crop_size: int, crop_type:str, keep_all_boxes = False, **kwargs):
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
    def __init__(self, target_size, size_type='absolute', pad_value = 0, method = cv2.INTER_LINEAR, **kwargs):
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