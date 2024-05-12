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


# Load config.
file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
config_path = os.path.join(file_path, "prediction_extraction_config.json")
config = ta.utils.load_config.load_config(config_path)

folder_path = config["paths"]["folder_path"]
model_path = config["paths"]["model_path"]
limit_img = config["general"]["limit_img"]
save_dir_extract_data = config["paths"]["save_dir_extract_data"]
csv_name = config["paths"]["extract_csv_name"]

predictions = ta.perform_inference.inference_yolov8seg.inference_yolov8seg_on_folder(folder_path, model_path, limit_img=limit_img)
# shape of "predictions": (num_imgs, num_predictions_per_img)

bin_masks = ta.analyze_segments.xyn_to_bin_mask.xyn_to_bin_mask(predictions[0][0].masks.xyn, predictions[0][0].orig_img.shape[1], predictions[0][0].orig_img.shape[0], predictions[0][0].orig_img)

segment_color = ta.analyze_segments.segment_extractors.SegmentColor(predictions[0][0].orig_img, bin_masks[0])
hist_stats = segment_color.extract_histogram_statistics(segment_color.calculate_color_histogram())

all_rows = []
print("Calculating segment statistics...")
for prediction in predictions:
    segment_areas = defaultdict(list) # each cls can have multiple segments
    calc_segment_areas = defaultdict(list)
    segment_colors = defaultdict(list) 
    print("Prediction:")
    labels = prediction[0].names
    bin_masks = ta.analyze_segments.xyn_to_bin_mask.xyn_to_bin_mask(prediction[0].masks.xyn, prediction[0].orig_img.shape[1], prediction[0].orig_img.shape[0], prediction[0].orig_img)
    #the order of masks (=order boxes) is NOT the same as the order of the labels
    for cls, mask in zip(prediction[0].boxes.cls.tolist(), bin_masks):
        segment_area_obj = ta.analyze_segments.segment_extractors.SegmentArea(prediction[0].orig_img, mask)
        segment_areas[labels[cls]].append(segment_area_obj)

        segment_color = ta.analyze_segments.segment_extractors.SegmentColor(prediction[0].orig_img, mask)
        segment_colors[labels[cls]].append(segment_color)

        calc_segment_areas[labels[cls]].append(segment_area_obj.calculate_area())

    area_ratios = ta.analyze_segments.segment_extractors.segment_area_comparison(segment_areas)
    color_hists = ta.analyze_segments.segment_extractors.segment_color_comparison(segment_colors)
    calc_segment_areas = {key:sum(val) / len(val) for key, val in calc_segment_areas.items()}
    df_segment_areas = pd.DataFrame(calc_segment_areas, index=[0])

    total_row = pd.concat([area_ratios, color_hists, df_segment_areas], axis=1)
    all_rows.append(total_row)

    # Save ID of image
    img_path = prediction[0].path
    img_id = os.path.basename(img_path).split(".")[0]
    total_row["img_id"] = img_id

df_segments = pd.concat(all_rows, axis=0, ignore_index=True)

print(f"Save csv file to {csv_name}")
df_segments.to_csv(os.path.join(save_dir_extract_data, csv_name), index=False)

#Check rows with NaN
#print(df_segments[df_segments.isna().any(axis=1)])
