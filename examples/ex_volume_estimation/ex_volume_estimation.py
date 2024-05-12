# Extracting the volume for images in ver1 and ver2 datasets

import computer_vision_streamlit_app as ta
import json
from PIL import Image
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":

    # Load config.
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    config_path = os.path.join(file_path, "volume_extraction_config.json")
    config = ta.utils.load_config.load_config(config_path)

    all_rows = []

    for version in config["general"]["versions"]:
        print(f"Extracting from version {version} dataset...")
        json_path = os.path.join(config["paths"]["json_path"], version)
        image_path = os.path.join(config["paths"]["image_path"], version)

        all_json_files = os.listdir(json_path)
        all_img_files = os.listdir(image_path)
        file_names_wo_ending = [file_name.split('.')[0] for file_name in all_json_files]

        if config["general"]["reduce_to"]:
            file_names_wo_ending = file_names_wo_ending[:config["general"]["reduce_to"]]

        for file_name in tqdm(file_names_wo_ending):

            loader = ta.utils.load_json_annotation.AnnotationLoader(file_name, json_path, image_path)
            img_np, img, boxes, masks, scores, all_cls, mask_w_label, labels = loader.load()

            #***************
            # Get binary masks
            #***************
            bin_masks, mask_w_label = ta.length_estimation.utils_vol_estimation.get_binary_masks(masks, img_np, img, mask_w_label)
            
            #***************
            # Compute COGs
            #***************
            cogs_array, ordered_cogs = ta.length_estimation.utils_vol_estimation.get_cogs(bin_masks, all_cls, labels)
            
            #***************
            # Compute skeleton
            #***************
            n_polynom = 2
            #weight = 20
            #cog_weights = np.array([weight]*cogs_array.shape[0])
            # Compute skeleton
            generator = ta.extract_skeleton.polynom_regression_in_mask.MaskPointGenerator(bin_masks, cogs_array)
            combined_mask = generator.get_combined_mask()
            #random_points = generator.get_points()

            # Use ODR instead of OLS
            #fitted_points = generator.fit_get_odr(degree=n_polynom)OrthogonalLinesGenerator

            # Use method 2
            '''
            alpha = config["skeleton_extraction_w_cog"]["alpha"]
            iterations = config["skeleton_extraction_w_cog"]["iterations"]
            n_samples = config["skeleton_extraction_w_cog"]["n_samples"]
            fitted_points = ta.extract_skeleton.extract_skeleton_w_cogs.method2_connect_cog(ordered_cogs, random_points, combined_mask, alpha=alpha, iterations=iterations, n_samples=n_samples)
            if fitted_points is None:
                print("Could not fit points with method 2. Fall back to regression...")
                fitted_points = generator.fit_get_odr(degree=n_polynom)
            '''

            # Method 3 (Fallback method=Regression)
            fitted_points = ta.length_estimation.utils_vol_estimation.get_middle_line_points(generator, cogs_array, combined_mask, n_polynom=n_polynom)

            #***************
            # Calculate orthogonal lines
            #***************
            num_lines = config["volume_settings"]["num_orthogonal_lines"]
            lines, generator = ta.length_estimation.utils_vol_estimation.get_orth_lines(num_lines, fitted_points, combined_mask, mask_w_label)
            
            #***************
            # Plot skeleton with orthogonal lines
            #***************
            drawer = ta.extract_skeleton.plot_skeleton.LineDrawer(points=fitted_points, image=img_np, orthogonal_lines=lines)
            drawer_bin = ta.extract_skeleton.plot_skeleton.LineDrawer(points=fitted_points, image=combined_mask, bin_mask=True, orthogonal_lines=None)
            if config["general"]["show_plots"]:
                #ta.plotting.inference_results.plot_segments_w_cogs(img_np, boxes, masks, all_cls, labels, cogs=cogs, alpha=0.5)
                
                ta.plotting.inference_results.plot_segments(img_np, boxes, masks, all_cls, labels, conf=scores, score=False)
                drawer.draw_line()
                drawer.show_image()
                drawer_bin.draw_line()
                drawer_bin.show_image()
                visualizer = ta.plotting.volume_visualizer.BodyVolumeVisualizer(lines)
                visualizer.visualize()

            #***************
            # Compute volume
            #***************
            k_mm_per_px = config["volume_settings"]["k_mm_per_px"]
            volumes = ta.length_estimation.utils_vol_estimation.get_volume_from_lines(lines, generator, k_mm_per_px)
            #print(f"total Estimated Volume: {total_estimated_volume} mm^3")
            #print(f"object_part volumes: {body_part_volumes} mm^3")

            #***************
            # Compute length
            #***************
            estimator = ta.length_estimation.length_estimation.LengthEstimator(fitted_points, mask_w_label, k_mm_per_px)
            length_per_segment = estimator.calculate_lengths(round_to=3)
            total_length = estimator.calculate_total_length(round_to=3)
            print(f"total length: {total_length} mm")
            print(f"lengths: {length_per_segment} mm")
            lengths = {"total_length": total_length, **length_per_segment}
            lengths = pd.DataFrame(lengths, index=[0])

            #***************
            # Save results
            #***************
            total_row = pd.concat([volumes, lengths], axis=1)
            total_row["file_name"] = file_name
            all_rows.append(total_row)
    
    if config["general"]["save_csv"]:
        df_volumes = pd.concat(all_rows, axis=0, ignore_index=True)
        df_volumes.to_csv(os.path.join(config["paths"]["target_df_dir"], config["paths"]["target_df_name"]), index=False)