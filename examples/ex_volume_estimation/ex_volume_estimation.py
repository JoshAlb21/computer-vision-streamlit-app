import tachinidae_analyzer as ta
import json
from PIL import Image
import numpy as np
import torch
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Load config.
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    config_path = os.path.join(file_path, "volume_extraction_config.json")
    config = ta.utils.load_config.load_config(config_path)

    all_json_files = os.listdir(config["paths"]["json_path"])
    all_img_files = os.listdir(config["paths"]["image_path"])
    file_names_wo_ending = [file_name.split('.')[0] for file_name in all_json_files]

    file_names_wo_ending = file_names_wo_ending[:config["general"]["reduce_to"]]

    for file_name in file_names_wo_ending:

        loader = ta.utils.load_json_annotation.AnnotationLoader(file_name, config["paths"]["json_path"], config["paths"]["image_path"])
        img_np, img, boxes, masks, scores, all_cls, mask_w_label, labels = loader.load()

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
        alpha = config["skeleton_extraction_w_cog"]["alpha"]
        iterations = config["skeleton_extraction_w_cog"]["iterations"]
        n_samples = config["skeleton_extraction_w_cog"]["n_samples"]
        fitted_points = ta.extract_skeleton.extract_skeleton_w_cogs.method2_connect_cog(ordered_cogs, random_points, combined_mask, alpha=alpha, iterations=iterations, n_samples=n_samples)
        
        #***************
        # Calculate orthogonal lines
        #***************
        num_lines = config["volume_settings"]["num_orthogonal_lines"] 
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
        k_mm_per_px = config["volume_settings"]["k_mm_per_px"] 
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
