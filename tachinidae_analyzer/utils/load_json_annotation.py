import os
import json
import numpy as np
from PIL import Image
import torch


class AnnotationLoader:
    def __init__(self, file_names_wo_ending:str, json_path:str, image_path:str):
        if '.' in file_names_wo_ending:
            raise ValueError('file_names_wo_ending contains probably a file ending. str must not contain a "." to count as clear file name')
        
        self.json_file_path = os.path.join(json_path, file_names_wo_ending + '.json')
        self.img_file_path = os.path.join(image_path, file_names_wo_ending + '.jpg')
        self.labels = {0: 'head', 1: 'thorax', 2: 'abdomen'}
    
    def _load_json(self):
        with open(self.json_file_path) as f:
            return json.load(f)

    def _load_image(self):
        self.img = Image.open(self.img_file_path)
        return np.array(self.img), self.img


    def _extract_data_from_shapes(self, annotation_data):
        boxes, masks, scores, all_cls = [], [], [], []
        mask_w_label = {'head': [], 'thorax': [], 'abdomen': []}
    
        for shape in annotation_data['shapes']:
            
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
