'''
from tachinidae_analyzer.perform_inference.inference_yolov8seg import inference_yolov8seg_on_folder

from tachinidae_analyzer.analyze_segments.segment_extractors import SegmentColor, SegmentArea, segment_area_comparison
from tachinidae_analyzer.analyze_segments.xyn_to_bin_mask import xyn_to_bin_mask
'''

import tachinidae_analyzer as ta

from collections import defaultdict
import os

import pandas as pd
from tqdm import tqdm

'''
This example shows how to apply YOLOv8seg to extract pseudo labels
these pseudo labels can be used to train a segmentation model (after supervision)

Besides the txt file also the "names" information is required (can be found in yolov8.Results object)

Label will be saved in runs/segment folder
'''


folder_path = "/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/img/ver2/"
model_path = "/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/models/YOLOv8_seg/weights/last.pt"
save_dir = "/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/prediction"
limit_img = 2
names = {0: 'head', 1: 'abdomen', 2: 'thorax'}

#***************
# Inference
#***************

#predictions = ta.perform_inference.inference_yolov8seg.inference_yolov8seg_on_folder(folder_path, model_path, limit_img=limit_img, save_txt=True, save_dir=save_dir)

#***************
# Convert to json
#***************

# Convert TXT to JSON and save the result to a new file
json_output_path = '/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/prediction/ver2'
txt_file_path = '/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/runs/segment/predict/labels'

dir_converter = ta.prepare_data.converter_annotation.DirAnnotationConverter(img_width=4032, img_height=3040)
dir_converter.txt_to_json_dir(txt_dir=txt_file_path, json_output_dir=json_output_path, names=names)
