import streamlit as st

from tachinidae_analyzer.analyze_segments.xyn_to_bin_mask import xyn_to_bin_mask
from tachinidae_analyzer.analyze_segments.segment_extractors import SegmentColor, SegmentArea, segment_area_comparison
from tachinidae_analyzer.plotting.area_ratio_barplot import plot_single_segmented_ratio_barplot
from tachinidae_analyzer.plotting.color_distribution import plot_color_histogram

from collections import defaultdict


def show_color_histogram(first_prediction, col_to_display:st.columns):

    # Histogram
    bin_masks = xyn_to_bin_mask(first_prediction[0].masks.xyn, first_prediction[0].orig_img.shape[1], first_prediction[0].orig_img.shape[0], first_prediction[0].orig_img)
    segment_color = SegmentColor(first_prediction[0].orig_img, bin_masks[0])
    fig = plot_color_histogram(segment_color.calculate_color_histogram(), return_image=True)
    if fig is not None:
        with col_to_display:
            st.pyplot(fig)

def show_area_ratio(first_prediction, col_to_display:st.columns):
    
    bin_masks = xyn_to_bin_mask(first_prediction[0].masks.xyn, first_prediction[0].orig_img.shape[1], first_prediction[0].orig_img.shape[0], first_prediction[0].orig_img)
    # Area ratio
    segment_areas = defaultdict(list) # each cls can have multiple segments
    labels = first_prediction[0].names
    for cls, mask in zip(first_prediction[0].boxes.cls.tolist(), bin_masks):
        segment_area_obj = SegmentArea(first_prediction[0].orig_img, mask)
        segment_areas[labels[cls]].append(segment_area_obj)

    area_ratios = segment_area_comparison(segment_areas)
    area_ratios = area_ratios.iloc[0].to_dict()

    fig = plot_single_segmented_ratio_barplot(area_ratios, list(first_prediction[0].names.values()), return_image=True)

    if fig is not None:
        with col_to_display:
            st.pyplot(fig)