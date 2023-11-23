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
    import tachinidae_analyzer as ta
except ModuleNotFoundError as e:
    #install local package
    subprocess.Popen([f'{sys.executable} -m pip install -e .'], shell=True)
    # wait for subprocess to install package before running your actual code below
    time.sleep(90)

st.set_page_config(layout="wide", page_title="Insect classifier/instance segmentation")

from tachinidae_analyzer.plotting.inference_results import plot_segments_from_results


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
    #model_path = "/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/models/YOLOv8_seg/"
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
    st.write("## Classify and segment insects :fly:")
    st.write(
        "This app allows you to classify and segment insects. Use SOTA models for your entomological research."
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
    ia_col1.write("#### Color histogram")
    ia_col2.write("#### Area ratio")

    seg_task_done = False
    cls_task_done = False

    #***************
    # Run segmentation
    #***************
    if imgs_upload is not None and path_selected_model_seg is not None:
        
        # Not required for offline app
        #upload_size = sum([img.size for img in imgs_upload])
        #if upload_size > MAX_FILE_SIZE:
        
        predictions, img_names, first_segment_image = run_seg_inference(imgs_upload=imgs_upload, path_selected_model=path_selected_model_seg)

        if predictions:

            # Analysis for single image (first image that was uploaded)
            ta.gui_dashboard.show_results.show_color_histogram(first_prediction=predictions[0], col_to_display=ia_col1)
            ta.gui_dashboard.show_results.show_area_ratio(first_prediction=predictions[0], col_to_display=ia_col2)

            # Annotations json files for all predictions
            zip_buffer_annotations = ta.gui_dashboard.process_zip_annotations.zip_all_annotations(predictions=predictions, img_names=img_names)

            # Volume estimation
            df_vol_res, first_res = ta.gui_dashboard.process_vol_est_main.all_vol_est_main(imgs_upload, predictions, k_mm_per_px, n_polynom_fallback, num_orthogonal_lines)
            ia_col1.table(df_vol_res.iloc[0].squeeze())
            vol_est_csv = convert_to_csv(df_vol_res)

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
        df_combined = df_vol_res
        valid_case = True
    elif not seg_task_done and cls_task_done: # Only classification
        df_combined = df_cls
        valid_case = True
    elif seg_task_done and cls_task_done: # Segmentation and classification
        # Combine csv files before downloading
        assert df_vol_res.shape[0] == df_cls.shape[0], "Number of rows in df_vol_res and df_cls must be equal"
        df_combined = pd.concat([df_vol_res, df_cls], axis=1)
        valid_case = True

    if valid_case:
        # Download all data selected
        all_data = {
                "cog_body_parts.csv": BytesIO(b'a,b,c'),
                "segmented_image.png": BytesIO(convert_image(first_segment_image)),
                "volume_estimation_img.png": BytesIO(convert_image(img_w_center_line)),
                "annotations.zip": zip_buffer_annotations,
                "information.csv": convert_to_csv(df_combined)
        }
        ta.gui_dashboard.button_download.button_download_all(all_data=all_data)

    st.write("### Insights&Analysis multiple images")
    st.markdown("""---""")


