'''
from computer_vision_streamlit_app.perform_inference.inference_yolov8seg import inference_yolov8seg_on_folder

from computer_vision_streamlit_app.analyze_segments.segment_extractors import SegmentColor, SegmentArea, segment_area_comparison
from computer_vision_streamlit_app.analyze_segments.xyn_to_bin_mask import xyn_to_bin_mask
'''

import computer_vision_streamlit_app as ta

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

# Load config.
file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
config_path = os.path.join(file_path, "data_integration_config.json")
config = ta.utils.load_config.load_config(config_path)


folder_path = config["paths"]["folder_path"]
model_path = config["paths"]["model_path"]
save_dir = config["paths"]["save_dir"]
limit_img = config["general"]["limit_img"]

names = {0: 'dog', 1: 'umbrella', 2: 'bicycle'}

#***************
# Inference
#***************

predictions = ta.perform_inference.inference_yolov8seg.inference_yolov8seg_on_folder(folder_path, model_path, limit_img=limit_img, save_txt=True, save_dir=save_dir)

#***************
# Convert to json
#***************

# Convert TXT to JSON and save the result to a new file
json_output_path = '/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/data/prediction/ver1'
txt_file_path = '/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/runs/segment/predict/labels'

dir_converter = ta.prepare_data.converter_annotation.DirAnnotationConverter(img_width=4032, img_height=3040)
dir_converter.txt_to_json_dir(txt_dir=txt_file_path, json_output_dir=json_output_path, names=names)
