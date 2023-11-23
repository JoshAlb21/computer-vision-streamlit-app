import numpy as np
import pandas as pd

import tachinidae_analyzer as ta

import streamlit as st


def get_binary_masks(masks, img_np, img, mask_w_label):
    bin_masks = ta.analyze_segments.xyn_to_bin_mask.xyn_to_bin_mask(masks, img.width, img.height, img_np)
    for key, i_masks in mask_w_label.items():
        mask_w_label[key] = ta.analyze_segments.xyn_to_bin_mask.xyn_to_bin_mask(i_masks, img.width, img.height, img_np)
        #print("Plot binary mask", key)
        #plot_binary_mask(mask_w_label[key][0])

    # Combine binary masks per segment
    mask_w_label = ta.extract_skeleton.combine_masks.combine_masks(mask_w_label)

    return bin_masks, mask_w_label

def get_cogs(bin_masks, all_cls, labels):
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

    return cogs_array, ordered_cogs

def get_orth_lines(num_lines:int, fitted_points, combined_mask, mask_w_label):
    generator = ta.extract_skeleton.orthogonal_slicer.OrthogonalLinesGenerator(fitted_points, combined_mask, separate_masks=mask_w_label)
    generator.generate_orthogonal_lines(num_lines=num_lines)
    lines = generator.get_orthogonal_lines()
    return lines, generator

def get_middle_line_points(generator, cogs_array, combined_mask, n_polynom:int=2):
    try:
        fitted_points = generator.interpolate_points_parametric_spline(given_points=cogs_array)
    except ValueError:
        print("Could not fit points with method 3 (Less than 2 CoGs). Fall back to regression...")
        fitted_points = generator.fit_get_odr(degree=n_polynom)
    fitted_points = ta.extract_skeleton.line_refiner.trim_line(combined_mask, fitted_points)
    fitted_points = ta.extract_skeleton.line_refiner.sample_points_from_segments(fitted_points, n=110)
    return fitted_points

def get_volume_from_lines(lines, generator, k_mm_per_px) -> pd.DataFrame:
    h_value = generator.get_h_mean()
    estimator = ta.length_estimation.volume_estimation.VolumeEstimator(lines, h_value, k_conv_factor=k_mm_per_px)
    total_estimated_volume, body_part_volumes = estimator.calculate_volume_in_mm_3(round_to=3)
    volumes = {"total_volume": total_estimated_volume, **body_part_volumes}
    volumes = pd.DataFrame(volumes, index=[0])
    return volumes

def get_length_from_lines(fitted_points, mask_w_label, k_mm_per_px) -> pd.DataFrame:
    estimator = ta.length_estimation.length_estimation.LengthEstimator(fitted_points, mask_w_label, k_mm_per_px)
    length_per_segment = estimator.calculate_lengths(round_to=3)
    total_length = estimator.calculate_total_length(round_to=3)
    print(f"total length: {total_length} mm")
    print(f"lengths: {length_per_segment} mm")
    lengths = {"total_length": total_length, **length_per_segment}
    lengths = pd.DataFrame(lengths, index=[0])
    return lengths