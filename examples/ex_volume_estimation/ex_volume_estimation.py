import tachinidae_analyzer as ta
import json
from PIL import Image
import numpy as np
import torch
import os

import matplotlib.pyplot as plt


def method2_connect_cog(ordered_cogs, random_points, combined_mask, alpha=1, iterations=500, n_samples=100):
    orderer = ta.extract_skeleton.point_orderer.PointOrderer(ordered_cogs)
    reference_points_extended = np.array(orderer.extended_reference_points)
    refiner = ta.extract_skeleton.line_refiner.GraphRefiner(reference_points_extended, random_points, alpha=alpha, iterations=iterations)
    optimized_points = refiner.get_refined_points()
    control_points = refiner.trim_by_mask(combined_mask)
    fitted_points = refiner.sample_points_from_segments(control_points, n=n_samples)

    return fitted_points

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


if __name__ == "__main__":

    json_path = '/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/prediction/ver2'
    image_path = '/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/img/ver2'

    target_plot_dir = '/Users/joshuaalbiez/Documents/masterthesis_iai/generated_plots/skeleton_extraction'


    all_json_files = os.listdir(json_path)
    all_img_files = os.listdir(image_path)
    file_names_wo_ending = [file_name.split('.')[0] for file_name in all_json_files]

    reduce_to = 10
    file_names_wo_ending = file_names_wo_ending[:reduce_to]

    for file_name in file_names_wo_ending:
        json_file_path = os.path.join(json_path, file_name + '.json')
        img_file_path = os.path.join(image_path, file_name + '.jpg')

        #***************
        # Load JSON
        #***************

        # Load JSON file
        with open(json_file_path) as f:
            annotation_data1 = json.load(f)

        #***************
        # Load image
        #***************

        # load image np.ndarray
        img = Image.open(img_file_path)
        img_np = np.array(img)  # Convert PIL Image to numpy array

        boxes = []
        masks = []
        scores = []
        all_cls = []

        labels = {0: 'head', 1: 'thorax', 2: 'abdomen'}

        mask_w_label = {'head': [], 'thorax': [], 'abdomen': []}

        for shape in annotation_data1['shapes']:
            
            # Extract bounding box from shape
            x_coords = [point[0] for point in shape['points']]
            y_coords = [point[1] for point in shape['points']]
            x1 = min(x_coords)
            y1 = min(y_coords)
            x2 = max(x_coords)
            y2 = max(y_coords)
            boxes.append([x1, y1, x2, y2])  # Now the box is in the [x1, y1, x2, y2] format


            mask = [[point[0] / img.width, point[1] / img.height] for point in shape['points']]
            masks.append(mask)
            mask_w_label[shape['label']].append(mask)
            scores.append(1)
            i_cls = list(labels.keys())[list(labels.values()).index(shape['label'])]
            all_cls.append(i_cls)

        masks = [np.array(mask) for mask in masks]
        mask_w_label = {key: [np.array(mask) for mask in masks] for key, masks in mask_w_label.items()}
        all_cls = torch.Tensor(all_cls)

        bin_masks = ta.analyze_segments.xyn_to_bin_mask.xyn_to_bin_mask(masks, img.width, img.height, img_np)
        for key, masks in mask_w_label.items():
            mask_w_label[key] = ta.analyze_segments.xyn_to_bin_mask.xyn_to_bin_mask(masks, img.width, img.height, img_np)
            #print("Plot binary mask", key)
            #plot_binary_mask(mask_w_label[key][0])

        # Combine binary masks per segment
        mask_w_label = ta.extract_skeleton.combine_masks.combine_masks(mask_w_label)

        #***************
        # Compute COGs
        #***************
        cogs = ta.analyze_segments.calc_2d_cog_binary_mask.compute_cogs(bin_masks, all_cls, labels)

        cogs_list = []
        for cog_i in range(3):
            try:
                cog_tuple = cogs[str(cog_i)]
                cogs_list.append(cog_tuple)
            except:
                pass
        cogs_array = np.array(cogs_list)

        ordered_cogs = ta.plotting.inference_results.order_cog_dict(cogs, max_i=3)
        
        #***************
        # Compute skeleton
        #***************
        n_polynom = 2
        weight = 20
        cog_weights = np.array([weight]*cogs_array.shape[0])
        # Compute skeleton
        generator = ta.extract_skeleton.polynom_regression_in_mask.MaskPointGenerator(bin_masks, cogs_array, cog_weights)
        combined_mask = generator.get_combined_mask()
        random_points = generator.get_points()

        # Use ODR instead of OLS
        #fitted_points = generator.fit_get_odr(degree=n_polynom)

        # Use method 2
        alpha = 1
        iterations = 500
        n_samples = 100
        fitted_points = method2_connect_cog(ordered_cogs, random_points, combined_mask, alpha=alpha, iterations=iterations, n_samples=n_samples)
        
        #***************
        # Calculate orthogonal lines
        #***************
        num_lines = 50
        generator = ta.extract_skeleton.orthogonal_slicer.OrthogonalLinesGenerator(fitted_points, combined_mask, separate_masks=mask_w_label)
        generator.generate_orthogonal_lines(num_lines=num_lines)
        # Remove intersecting lines
        # TODO takes to much time, find algorithm to speed up, e.g. bentley ottmann
        # generator.remove_intersecting_lines()
        # Get the remaining lines
        lines = generator.get_orthogonal_lines()
        
        #***************
        # Plot skeleton with orthogonal lines
        #***************
        drawer = ta.extract_skeleton.plot_skeleton.LineDrawer(points=fitted_points, image=img_np, orthogonal_lines=lines)
        drawer.show_image()

        #***************
        # Compute volume
        #***************
        k_mm_per_px = 1
        h_value = generator.get_h_mean()
        estimator = ta.length_estimation.volume_estimation.VolumeEstimator(lines, h_value, k_conv_factor=k_mm_per_px)
        total_estimated_volume, body_part_volumes = estimator.calculate_volume_in_mm_3(round_to=1)
        print(f"total Estimated Volume: {total_estimated_volume} mm^3")
        print(f"body part volumes: {body_part_volumes} mm^3")

        #***************
        # Compute length
        #***************
        estimator = ta.length_estimation.length_estimation.LengthEstimator(fitted_points, mask_w_label, k_mm_per_px)
        length_per_segment = estimator.calculate_lengths(round_to=1)
        total_length = estimator.calculate_total_length(round_to=1)
        print(f"total length: {total_length} mm")
        print(f"lengths: {length_per_segment} mm")
