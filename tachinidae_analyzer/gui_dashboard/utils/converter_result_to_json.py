import numpy as np
import cv2
import json
import os

import streamlit as st


class ResultsConverter:
    def __init__(self, results, set_default_group_id:int=1, img_name:str=None):
        """
        Convert a Results object to a JSON format that can be used by the LabelMe tool.

        Parameters:
        - results: An instance of ultralytics.engine.results.Results
        - set_default_group_id: The group ID to be used for all shapes. Default: 1
        - img_name: The name of the image. Note only required if the image name is not stored in the Results object.
            e.g. if inference was directly performed on a np.array or PIL.Image object. and NOT on a folder.
        """
        self.results = results
        self.set_default_group_id = set_default_group_id
        self.img_name = img_name

    def _mask_to_points(self, xyn_s) -> list:
        # Assuming xyn_s is a list of normalized points for a mask
        w, h = self.results.orig_img.shape[1], self.results.orig_img.shape[0]
        denormalized_points = []
        for xyn in xyn_s:
            mask_points = xyn.copy()
            mask_points[:, 0] *= w
            mask_points[:, 1] *= h
            denormalized_points.append(mask_points.astype(np.int32).tolist())
        return np.squeeze(denormalized_points).tolist()

    def _convert_shapes(self):
        shapes = []
        for mask, label_idx in zip(self.results.masks, self.results.boxes.cls.tolist()):#self.results.names):
            shape = {
                'label': self.results.names[label_idx],
                'points': self._mask_to_points(mask.xyn),
                'group_id': self.set_default_group_id,  # Add group ID if relevant
                'shape_type': 'polygon',  # Assuming masks are polygons
                'flags': {}
            }
            shapes.append(shape)
        
        # sort shapes by label order given in order_labels
        order_labels = ['head', 'thorax', 'abdomen']
        shapes = sorted(shapes, key=lambda k: order_labels.index(k['label']))
        return shapes

    def convert_to_json_format(self):
        def get_file_name_from_path(path:str):
            " Get the file name from a path (on any OS)"
            if self.img_name is not None:
                file_name = self.img_name
            else:
                file_name = path.split(os.sep)[-1]
            return file_name

        json_data = {
            'version': '4.5.6',  # Example version, adjust as needed
            'flags': {},  # Add any relevant flags if needed
            'shapes': self._convert_shapes(),
            'imagePath': get_file_name_from_path(self.results.path),
            'imageData': None,  # Assuming imageData is not available in Results
            'imageHeight': self.results.orig_img.shape[0] if self.results.orig_img is not None else 0,
            'imageWidth': self.results.orig_img.shape[1] if self.results.orig_img is not None else 0,
            'lineColor': [0, 255, 0, 128],  # Example line color, adjust as needed
            'fillColor': [255, 0, 0, 128]   # Example fill color, adjust as needed
        }
        return json_data
    
    def save_to_file(self, json_file_path):
        json_data = self.convert_to_json_format()
        with open(json_file_path, 'w') as file:
            json.dump(json_data, file, indent=4)
    
    def get_as_json(self):
        json_data = self.convert_to_json_format()
        return json_data


# Example usage
# Assuming you have a Results object
if __name__ == "__main__":
    import pickle
    with open('prediction.pickle', 'rb') as handle:
        results = pickle.load(handle)

    converter = ResultsConverter(results[0][0])


    #converter.save_to_file('output.json')

    res = converter.get_as_json()

    print(res)
