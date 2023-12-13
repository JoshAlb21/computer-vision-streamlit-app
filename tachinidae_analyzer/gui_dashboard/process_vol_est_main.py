import streamlit as st
from PIL import Image
from io import BytesIO
import pandas as pd

import tachinidae_analyzer as ta

def all_vol_est_main(imgs_upload, predictions, k_mm_per_px, n_polynom_fallback, num_orthogonal_lines) -> tuple[pd.DataFrame, dict]:
    'Collects all necessary functions for volume estimation'
    all_rows = []
    first = True
    first_res = {'lines': None, 'fitted_points': None, 'img_np': None}
    for img_upload, prediction in zip(imgs_upload, predictions):
        img_pil = Image.open(img_upload)
        converter = ta.gui_dashboard.utils.converter_result_to_json.ResultsConverter(prediction[0], img_name=img_upload)
        json_anno_data = converter.get_as_json()
        loader = ta.utils.load_json_annotation.AnnotationConverter(json_annotation=json_anno_data, img=img_pil)
        img_np, img, boxes, masks, scores, all_cls, mask_w_label, labels = loader.load()

        #***************
        # Get binary masks
        #***************
        bin_masks, mask_w_label = ta.length_estimation.utils_vol_estimation.get_binary_masks(masks, img_np, img, mask_w_label)

        #***************
        # Compute COGs
        #***************
        cogs_array, ordered_cogs = ta.length_estimation.utils_vol_estimation.get_cogs(bin_masks, all_cls, labels)

        present_label_names = [labels[i_cls.item()] for i_cls in all_cls]
        present_label_names = list(dict.fromkeys(present_label_names)) # Remove duplicates (in case a label is present multiple times)

        cog_dict = {}
        for idx, present_label in enumerate(present_label_names):
            cog_dict[f'x_{present_label}'] = cogs_array[idx][0]
            cog_dict[f'y_{present_label}'] = cogs_array[idx][1]

        df_cogs = pd.DataFrame(cog_dict, index=[0])

        #***************
        # Compute skeleton
        #***************
        generator = ta.extract_skeleton.polynom_regression_in_mask.MaskPointGenerator(bin_masks, cogs_array)
        combined_mask = generator.get_combined_mask()

        fitted_points = ta.length_estimation.utils_vol_estimation.get_middle_line_points(generator, cogs_array, combined_mask, n_polynom=n_polynom_fallback)

        #***************
        # Calculate orthogonal lines
        #***************
        lines, generator = ta.length_estimation.utils_vol_estimation.get_orth_lines(num_orthogonal_lines, fitted_points, combined_mask, mask_w_label)

        #***************
        # Compute volume
        #***************
        volumes = ta.length_estimation.utils_vol_estimation.get_volume_from_lines(lines, generator, k_mm_per_px)

        if first:
            first_res['lines'] = lines
            first_res['fitted_points'] = fitted_points
            first_res['img_np'] = img_np
            first = False

        #***************
        # Compute length
        #***************
        estimator = ta.length_estimation.length_estimation.LengthEstimator(fitted_points, mask_w_label, k_mm_per_px)
        length_per_segment = estimator.calculate_lengths(round_to=3)
        total_length = estimator.calculate_total_length(round_to=3)
        lengths = {"total_length": total_length, **length_per_segment}
        lengths = pd.DataFrame(lengths, index=[0])

        # Combine to df_cogs and volumes
        df_res_row = pd.concat([df_cogs, volumes, lengths], axis=1)
        all_rows.append(df_res_row)

    df_vol_res = pd.concat(all_rows, axis=0, ignore_index=True)

    return df_vol_res, first_res