import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import pandas as pd
import json
from ultralytics import YOLO
import mpld3
import streamlit.components.v1 as components
import subprocess
import sys
import time

try:
    import computer_vision_streamlit_app as ta
except ModuleNotFoundError as e:
    #install local package
    subprocess.Popen([f'{sys.executable} -m pip install -e .'], shell=True)
    # wait for subprocess to install package before running your actual code below
    time.sleep(90)

st.set_page_config(layout="wide", page_title="Object classifier/instance segmentation")

from computer_vision_streamlit_app.plotting.inference_results import plot_segments_from_results


def add_units_to_df_vol_result(df_vol_res:pd.DataFrame):
    '''
    Example:
        x_head	3,001.1772
        y_head	2,334.0549
        x_thorax	2,720.7536
        y_thorax	2,160.7216
        x_abdomen	2,174.2892
        y_abdomen	2,084.0675
        total_volume	0.2160
        volume_head	0.0700
        volume_thorax	0.0870
        volume_abdomen	0.0590
        total_length	4.0100
        length_head	0.9560
        length_thorax	1.1770
        length_abdomen	1.3240
    '''

    # Currently df is one row
    # transpose it
    df_vol_res = df_vol_res.iloc[0].squeeze().to_frame()

    # Add "Unit" column to df_vol_res, fill value separately for each row
    df_vol_res = df_vol_res.assign(Unit=['pixel', 'pixel', 'pixel', 'pixel', 'pixel', 'pixel', 
                                        'mm^3', 'mm^3', 'mm^3', 'mm^3', 'mm', 'mm', 'mm', 'mm'])

    return df_vol_res

#MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB # Not required for offline app

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

@st.cache_data
def convert_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def run_seg_inference(imgs_upload:list, path_selected_model:str, task:str="segment"):
    #col1.write(f"Show 1 image out of {len(imgs_upload)} images)")
    col1.write("Raw Image")
    col2.write("Image with segmentation")
    #model_path = "/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/data/models/YOLOv8_seg/"
    model = YOLO(path_selected_model, task=task)
    predictions = []
    img_names = []
    first_segment_image = None
    for idx, image in enumerate(imgs_upload):
        img_names.append(image.name)

        # Raw image
        image = Image.open(image)
        if idx == 0:
            col1.image(image)

        # Run inference on the image
        prediction = model.predict(image)
        predictions.append(prediction)
        
        #img_w_segmentation = ta.gui_dashboard.button_inference.button_inference_folder('test', 'test')
        img_w_segmentation = plot_segments_from_results(prediction[0], return_image=True)
        if idx == 0:
            first_segment_image = img_w_segmentation
            col2.image(img_w_segmentation)
        #st.sidebar.markdown("\n")
        #st.sidebar.download_button("Download segmentation image", convert_image(img_w_segmentation), "img_w_segmentation.png", "image/png")

    return predictions, img_names, first_segment_image

def run_cls_inference(imgs_upload:list, path_selected_model:str, task:str="classify", show_first_raw:bool=False):

    model = YOLO(path_selected_model, task=task)
    predictions = []
    first_cls = None
    first_conf = None
    for idx, image in enumerate(imgs_upload):
        # Raw image
        image = Image.open(image)
        if idx == 0 and show_first_raw:
            col1.write("Raw Image")
            col1.image(image)

        # Run inference on the image
        prediction = model.predict(image)
        predictions.append(prediction)
        
        if idx == 0:
            first_cls, first_conf = prediction[0].probs.top1, prediction[0].probs.top1conf

    return predictions, first_cls, first_conf

if __name__ == '__main__':
    st.write("## Classify and segment objects")
    st.write(
        "This app allows you to classify and segment objects. Use SOTA models for your entomological research."
    )
    st.write("### Preview of first selected image")
    st.markdown("""---""")
    imgs_upload = ta.gui_dashboard.display_sidebar_elements.disp_upload_section()

    col1, col2 = st.columns(2)

    path_selected_model_seg, path_selected_model_cls = ta.gui_dashboard.display_sidebar_elements.disp_model_settings()

    k_mm_per_px, n_polynom_fallback, num_orthogonal_lines = ta.gui_dashboard.display_sidebar_elements.disp_volume_est_settings()

    st.write("### Insights&Analysis single image")
    st.markdown("""---""")
    ia_col1, ia_col2 = st.columns(2)
    ia_col1.write("#### General information")
    ia_col2.write("#### Area ratio")

    seg_task_done = False
    cls_task_done = False

    #***************
    # Run segmentation
    #***************
    if imgs_upload is not None and path_selected_model_seg is not None:

        df_name = pd.DataFrame([img.name.split('.')[0] for img in imgs_upload], columns=['file_name'])
        
        # Not required for offline app
        #upload_size = sum([img.size for img in imgs_upload])
        #if upload_size > MAX_FILE_SIZE:
        
        predictions, img_names, first_segment_image = run_seg_inference(imgs_upload=imgs_upload, path_selected_model=path_selected_model_seg)

        if predictions:

            # Analysis for single image (first image that was uploaded)
            #ta.gui_dashboard.show_results.show_color_histogram(first_prediction=predictions[0], col_to_display=ia_col1)
            #ta.gui_dashboard.show_results.show_area_ratio(first_prediction=predictions[0], col_to_display=ia_col2)

            # Annotations json files for all predictions
            zip_buffer_annotations = ta.gui_dashboard.process_zip_annotations.zip_all_annotations(predictions=predictions, img_names=img_names)

            # Volume estimation, Not supported here - check out my other repository for the full code
            df_vol_res = None

            #***************
            # Plot skeleton with orthogonal lines
            #***************
            drawer = ta.extract_skeleton.plot_skeleton.LineDrawer(points=first_res['fitted_points'], image=first_res['img_np'], orthogonal_lines=first_res['lines'], conv_2_rgb=False)
            img_w_center_line = drawer.get_img()
            col1.image(img_w_center_line)
            visualizer = ta.plotting.volume_visualizer.BodyVolumeVisualizer(first_res['lines'])
            volume_3d_fig = visualizer.visualize(return_fig=True)
            with col2:
                st.pyplot(volume_3d_fig)
            
            seg_task_done = True

    #***************
    # Run classification
    #***************
    if imgs_upload is not None and path_selected_model_cls is not None:

        cls_predictions, first_cls, first_conf = run_cls_inference(imgs_upload=imgs_upload, path_selected_model=path_selected_model_cls, show_first_raw=False)

        if cls_predictions:
            # Classification results
            class_names = cls_predictions[0][0].names
            classes = [class_names[pred[0].probs.top1] for pred in cls_predictions]
            confidences = [round(pred[0].probs.top1conf.item(), 4) for pred in cls_predictions]
            df_cls = pd.DataFrame({"class": classes, "confidence": confidences})
            col1.write(f"Class: {cls_predictions[0][0].names[cls_predictions[0][0].probs.top1]},\n Confidence: {round(cls_predictions[0][0].probs.top1conf.item(), 4)}")

            cls_task_done = True

    #***************
    # General Processing
    #***************
    valid_case:bool = False
    if not seg_task_done and not cls_task_done: # No segmentation and no classification
        df_combined = None
    elif seg_task_done and not cls_task_done: # Only segmentation
        df_combined = pd.concat([df_vol_res, df_name], axis=1)
        valid_case = True
    elif not seg_task_done and cls_task_done: # Only classification
        df_combined = pd.concat([df_cls, df_name], axis=1)
        valid_case = True
    elif seg_task_done and cls_task_done: # Segmentation and classification
        # Combine csv files before downloading
        assert df_vol_res.shape[0] == df_cls.shape[0], "Number of rows in df_vol_res and df_cls must be equal"
        df_combined = pd.concat([df_vol_res, df_cls, df_name], axis=1)
        valid_case = True

    if valid_case:
        # Download all data selected
        all_data = {
                "cog_object_parts.csv": BytesIO(b'a,b,c'),
                "segmented_image.png": BytesIO(convert_image(first_segment_image)),
                "volume_estimation_img.png": BytesIO(convert_image(img_w_center_line)),
                "annotations.zip": zip_buffer_annotations,
                "information.csv": convert_to_csv(df_combined)
        }
        ta.gui_dashboard.button_download.button_download_all(all_data=all_data)


    #***************
    # Show Insights&Analysis multiple images
    #***************
    if imgs_upload is not None and path_selected_model_seg is not None and df_combined is not None:
        st.write("### Insights&Analysis multiple images")
        st.markdown("""---""")
        ia_multi_col1, ia_multi_col2 = st.columns(2)
        ia_multi_col1.write("#### Volume estimation")
        ia_multi_col2.write("#### Length estimation")

        cols_len = ["length_head", "length_thorax", "length_abdomen"]
        cols_vol = ["volume_head", "volume_thorax", "volume_abdomen"]

        vol_plot, len_plot = ta.plotting.pairwise_vol_len.plot_pairwise_vol_len(df_combined, cols_vol=cols_vol, cols_len=cols_len, hue="class")

        ia_multi_col1.pyplot(vol_plot)
        ia_multi_col2.pyplot(len_plot)
