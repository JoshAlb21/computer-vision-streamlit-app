from tachinidae_analyzer.plotting.inference_results import plot_segments_from_results
from tachinidae_analyzer.perform_inference.inference_yolov8seg import inference_yolov8seg_on_folder
from tachinidae_analyzer.plotting.color_distribution import plot_color_histogram
from tachinidae_analyzer.plotting.visualize_bin_mask import visualize_bin_mask
from tachinidae_analyzer.plotting.area_ratio_barplot import plot_grouped_ratio_barplot_with_labels

from tachinidae_analyzer.analyze_segments.segment_extractors import SegmentColor, SegmentArea, segment_area_comparison
from tachinidae_analyzer.analyze_segments.xyn_to_bin_mask import xyn_to_bin_mask

from collections import defaultdict


folder_path = "/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/img/ver1/"
model_path = "/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/models/YOLOv8_seg/weights/last.pt"

predictions = inference_yolov8seg_on_folder(folder_path, model_path, limit_img=1)
bin_masks = xyn_to_bin_mask(predictions[0][0].masks.xyn, predictions[0][0].orig_img.shape[1], predictions[0][0].orig_img.shape[0], predictions[0][0].orig_img)

plot_segments_from_results(predictions[0][0])

# Histogram
predictions = inference_yolov8seg_on_folder(folder_path, model_path, limit_img=1)
bin_masks = xyn_to_bin_mask(predictions[0][0].masks.xyn, predictions[0][0].orig_img.shape[1], predictions[0][0].orig_img.shape[0], predictions[0][0].orig_img)
segment_color = SegmentColor(predictions[0][0].orig_img, bin_masks[0])
plot_color_histogram(segment_color.calculate_color_histogram())

# Bin mask
visualize_bin_mask(bin_masks[2])

# Area ratio
segment_areas = defaultdict(list) # each cls can have multiple segments
labels = predictions[0][0].names
for cls, mask in zip(predictions[0][0].boxes.cls.tolist(), bin_masks):
    print("Process segment of class:", labels[cls])
    segment_area_obj = SegmentArea(predictions[0][0].orig_img, mask)
    print(f"{labels[cls]}, {segment_area_obj.calculate_area()}")
    segment_areas[labels[cls]].append(segment_area_obj)

area_ratios = segment_area_comparison(segment_areas)
plot_grouped_ratio_barplot_with_labels(area_ratios, list(predictions[0][0].names.values()))