import os
import json
import glob
import tqdm
class COCO_Converter(object):
    '''
    Format annotation to standard form from the coco format annotations. 
    The original annotation file is a COCO-format JSON annotation file. 
    When loaded with json library, it is a dictionary data with following keys:
    dict_keys(['licenses', 'info', 'images', 'annotations', 'categories'])
    An example of data['images'] (a list of dictionaries):
    {'width': 400, 'date_captured': '', 'license': 0, 'flickr_url': '', 'file_name': '0060001.jpg', 'id': 60001, 'coco_url': '', 'height': 600}
    An example of data['annotations'] (a list of dictionaries):
    {'image_id': 60001, 'bbox': [218.0, 406.0, 138.0, 47.0], 'area': 6486.0, 'rec': [95, ..., 96], 'category_id': 1, 'iscrowd': 0, 'id': 1, 
    'bezier_pts': [219.0, ..., 218.0, 452.0]}
    'bbox' is defined by [x_min, y_min, width, height] in coco format.

    self._format_det_label transforms the annotations into a single det label file with a format like:
    0060001.jpg	[{"transcription": "the", "points":[[153, 347], ..., [177, 357]], 'beizers':[], 'polygons': []}]
    '''
    def __init__(self, path_mode='relative', extra_label_keys = []):
        self.path_mode = path_mode
        self.extra_label_keys = extra_label_keys
        self.CTLABELS =  [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
        self.vocabulary_size = len(self.CTLABELS)+1

    def convert(self, task='det', image_dir=None, label_path=None, output_path=None):
        self.label_path = label_path
        assert os.path.exists(label_path), f'{label_path} no exist!'
        
        if task == 'det':
            self._format_det_label(image_dir, self.label_path, output_path)
        if task == 'rec':
            raise ValueError("MLT 2017 dataset has no cropped word images and recognition labels.")

    def _decode_rec_ids_to_string(self, rec):
        transcription = ""
        if rec is None:
            return None
        for index in rec:
            if index==self.vocabulary_size-1:
                transcription+=u'口'
            elif index<self.vocabulary_size-1:
                transcription+=self.CTLABELS[index]
        return transcription


    def _format_det_label(self, image_dir, label_path, output_path):

        with open(output_path, 'w') as out_file:
            coco_json_data = json.load(open(label_path, 'r'))
            # coco_json_data is a dictionary with keys: licenses, info, images, annotations, categories
            images_list = coco_json_data['images'] # a list of dictionaries, each dictionary has keys: width, date_captured, license, flickr_url, file_name, id, coco_url, height
            annotations_list = coco_json_data['annotations'] # a list of dictionaries. Each dictionary may have keys: 'image_id', 'file_name', 'bbox', 'rec', 'bezier_pts', 'area', 'segm' 
            images_to_annoations = dict([(image_info['id'], []) for image_info in images_list])
            images_to_file_names = dict([(image_info['id'], image_info['file_name']) for image_info in images_list])
            for annot_obj in annotations_list:
                image_id = annot_obj['image_id']
                assert image_id in images_to_annoations, f'found {image_id} image in the annotation list but not in images_list'
                annot_obj.update({"file_name": images_to_file_names[image_id]})
                images_to_annoations[image_id].append(annot_obj) 
            images_to_annoations = dict(sorted(images_to_annoations.items()))

            ann_keys = ['image_id', 'file_name', 'bbox', 'rec', 'bezier_pts', 'area', 'segm'] + self.extra_label_keys or []
            save_image_labels = {}
            num_instances_with_valid_segmentation = 0
            for image_id in tqdm.tqdm(images_to_annoations, total=len(images_to_annoations)):
                annotations_per_image = images_to_annoations[image_id]
                if len(annotations_per_image)==0:
                    continue
                img_path = os.path.join(image_dir, annotations_per_image[0]['file_name'])
                assert os.path.exists(img_path), f'{img_path} not exist! Please check the input image_dir {image_dir}'
                if self.path_mode == 'relative':
                    img_basename = os.path.basename(img_path)
                if img_basename not in save_image_labels:
                    save_image_labels[img_basename] = []
                #parse annotations
                for annot in annotations_per_image:
                    assert annot['image_id'] == image_id
                    segm = annot.get('segmentation', None)
                    if segm:# either list[list[float]] or dict
                        if not isinstance(segm, dict):
                            segm = [pts for pts in segm if len(pts)%2==0 or len(pts) >=6]
                            if len(segm)==0:
                                num_instances_with_valid_segmentation +=1
                                continue # skip this instance
                    image_annot = {}
                    transcription = self._decode_rec_ids_to_string(annot.get('rec', None)) # needs to translate from character ids to characters.
                    if transcription:
                        image_annot['transcription'] = transcription
                    if 'bbox' in annot:
                        bbox = annot['bbox'] # [x_min, y_min, width, height]
                        bbox = [[bbox[0], bbox[1]], [bbox[0]+bbox[2], bbox[1]], [bbox[0]+bbox[2], bbox[1]+bbox[3]], [bbox[0], bbox[1]+bbox[3]]]
                        bbox = [[x[0], x[1]] for x in bbox]
                        image_annot['points'] = bbox # in other dataset converters, annt['points'] are always bbox, usually in four points, sometimes 5 or 6 points
                    polys = annot.get('polys', None)
                    if polys:
                        polys = [[x, y] for x, y in zip(polys[0::2], polys[1::2])]
                        image_annot['polys'] = polys
                    bezier = annot.get('bezier_pts', None)
                    if bezier:
                        image_annot["beziers"]=bezier

                    save_image_labels[img_basename].append(image_annot)
            for img_basename in save_image_labels: 
                annotations = save_image_labels[img_basename]
                out_file.write(img_basename + '\t' + json.dumps(
                    annotations, ensure_ascii=False) + '\n')
