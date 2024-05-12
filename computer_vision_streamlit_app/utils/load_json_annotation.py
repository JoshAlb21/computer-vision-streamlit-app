import os
import json
import numpy as np
from PIL import Image
import torch

#TODO maybe create an ABC Base class for AnnotationLoader and AnnotationConverter (both can inherit from it)

class AnnotationLoader:
    def __init__(self, file_names_wo_ending:str, json_path:str, image_path:str):
        if '.' in file_names_wo_ending:
            raise ValueError('file_names_wo_ending contains probably a file ending. str must not contain a "." to count as clear file name')
        
        self.json_file_path = os.path.join(json_path, file_names_wo_ending + '.json')
        self.img_file_path = os.path.join(image_path, file_names_wo_ending + '.jpg')
        self.labels = {0: 'dog', 1: 'bicycle', 2: 'umbrella'}
    
    def _load_json(self):
        with open(self.json_file_path) as f:
            return json.load(f)

    def _load_image(self):
        self.img = Image.open(self.img_file_path)
        return np.array(self.img), self.img

    def _extract_data_from_shapes(self, annotation_data):
        boxes, masks, scores, all_cls = [], [], [], []
        mask_w_label = {'dog': [], 'bicycle': [], 'umbrella': []}
    
        for shape in annotation_data['shapes']:

            if shape['label'] not in self.labels.values():
                continue
            
            # Extract bounding box from shape
            x_coords = [point[0] for point in shape['points']]
            y_coords = [point[1] for point in shape['points']]
            x1 = min(x_coords)
            y1 = min(y_coords)
            x2 = max(x_coords)
            y2 = max(y_coords)
            boxes.append([x1, y1, x2, y2])  # Now the box is in the [x1, y1, x2, y2] format

            mask = [[point[0] / self.img.width, point[1] / self.img.height] for point in shape['points']]
            masks.append(mask)
            mask_w_label[shape['label']].append(mask)
            scores.append(1)
            i_cls = list(self.labels.keys())[list(self.labels.values()).index(shape['label'])]
            all_cls.append(i_cls)

        masks = [np.array(mask) for mask in masks]
        mask_w_label = {key: [np.array(mask) for mask in masks] for key, masks in mask_w_label.items()}
        all_cls = torch.Tensor(all_cls)

        return boxes, masks, scores, all_cls, mask_w_label

    def load(self):
        annotation_data = self._load_json()
        img_np, img = self._load_image()
        boxes, masks, scores, all_cls, mask_w_label = self._extract_data_from_shapes(annotation_data)
        
        return img_np, img, boxes, masks, scores, all_cls, mask_w_label, self.labels


class AnnotationConverter(AnnotationLoader):
    "Converts a json annotation file from memory to the AnnotationLoader format"
    def __init__(self, json_annotation:dict, img:np.ndarray):
        self.json_annotation = json_annotation
        #np.array(imgs_upload[0])
        #img = np.asarray(img, dtype='float64')
        #self.img = Image.fromarray((img * 255).astype(np.uint8)) # Because array is normalized
        self.img = img
        self.labels = {0: 'dog', 1: 'bicycle', 2: 'umbrella'}

    def load(self):
        boxes, masks, scores, all_cls, mask_w_label = self._extract_data_from_shapes(self.json_annotation)

        img_np = np.array(self.img)
        
        return img_np, self.img, boxes, masks, scores, all_cls, mask_w_label, self.labels

if __name__ == '__main__':
    json_path = '/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/data/prediction/ver1/CAS001_CAS0000057_stacked_02_H.json'
    img_path = '/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/data/img/ver1/CAS001_CAS0000057_stacked_02_H.jpg'

    with open(json_path) as f:
        json_annotation = json.load(f)

    img = Image.open(img_path)

    converter = AnnotationConverter(json_annotation, img)
    img_np, img, boxes, masks, scores, all_cls, mask_w_label, labels = converter.load()

    loader = AnnotationLoader('CAS001_CAS0000057_stacked_02_H', '/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/data/prediction/ver1/', '/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/data/img/ver1/')
    img_np_2, img_2, boxes_2, masks_2, scores_2, all_cls_2, mask_w_label_2, labels_2 = loader.load()

    print(img_np.shape)

    