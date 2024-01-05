from tachinidae_analyzer.plotting.inference_results import plot_segments_from_results
from tachinidae_analyzer.perform_inference.inference_yolov8seg import inference_yolov8seg_on_folder
from tachinidae_analyzer.plotting.color_distribution import plot_color_histogram
from tachinidae_analyzer.plotting.visualize_bin_mask import visualize_bin_mask
from tachinidae_analyzer.plotting.area_ratio_barplot import plot_grouped_ratio_barplot_with_labels, plot_single_segmented_ratio_barplot

from tachinidae_analyzer.analyze_segments.segment_extractors import SegmentColor, SegmentArea, segment_area_comparison
from tachinidae_analyzer.analyze_segments.xyn_to_bin_mask import xyn_to_bin_mask

from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import matplotlib.patches as patches
import numpy as np

def plot_mask(image, masks, single_mask_index=None, single_mask_array:np.ndarray=None):
    """
    Plot masks on the image.

    :param image: Original image as a numpy array.
    :param masks: List of masks, each mask is an array of normalized [x, y] points.
    :param single_mask_index: Index of the single mask to be plotted (optional).
    """
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)

    img_height, img_width = image.shape[:2]

    if single_mask_index is not None:
        # Plot only the specified mask
        mask = masks[single_mask_index]
        pixel_coords = mask * np.array([img_width, img_height])
        polygon = patches.Polygon(pixel_coords, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(polygon)
    else:
        # Plot all masks
        if single_mask_array is not None:
            pixel_coords = single_mask_array * np.array([img_width, img_height])
            polygon = patches.Polygon(pixel_coords, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(polygon)
        else:
            for mask in masks:
                pixel_coords = mask * np.array([img_width, img_height])
                polygon = patches.Polygon(pixel_coords, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(polygon)

    plt.show()

def plot_binary_mask(mask):
    """
    Plot a binary mask using matplotlib.

    Parameters:
    - mask (np.ndarray): A 2D numpy array with binary values.
    """
    plt.imshow(mask, cmap='gray')
    plt.colorbar(label='Pixel Value')
    plt.title('Binary Mask Visualization')
    plt.xlabel('X-axis (Pixels)')
    plt.ylabel('Y-axis (Pixels)')
    plt.show()


folder_path = '/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/fix_bugs/fix_data/ver2_example/'
model_path = "/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/models/YOLOv8_seg/yolov8l_seg_w_aug_v2.pt" #yolov8l_seg_w_aug_v2.pt" #yolov8n_seg_w_aug.pt"


predictions = inference_yolov8seg_on_folder(folder_path, model_path, limit_img=1)
bin_masks = xyn_to_bin_mask(predictions[0][0].masks.xyn, predictions[0][0].orig_img.shape[1], predictions[0][0].orig_img.shape[0], predictions[0][0].orig_img)
item = 1
predictions[0][0].boxes.cls[item].item()

print("CLS outside", predictions[0][0].boxes.cls)

print(predictions[0][0].names[predictions[0][0].boxes.cls[item].item()])

plot_mask(predictions[0][0].orig_img, None, None, predictions[0][0].masks.xyn[item])

plot_binary_mask(predictions[0][0].masks.data[item])

#plot_segments_from_results(predictions[0][0], return_image=False)

# Load image as np array
image = predictions[0][0].orig_img

