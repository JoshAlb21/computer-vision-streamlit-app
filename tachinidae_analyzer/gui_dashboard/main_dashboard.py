import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from ultralytics import YOLO

st.set_page_config(layout="wide", page_title="Insect classifier/instance segmentation")

import tachinidae_analyzer as ta
from tachinidae_analyzer.plotting.inference_results import plot_segments_from_results


MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def run_inference(imgs_upload:list, path_selected_model:str):
    #col1.write(f"Show 1 image out of {len(imgs_upload)} images)")
    col1.write("Raw Image")
    col2.write("Image with segmentation")
    #model_path = "/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/models/YOLOv8_seg/weights/last.pt"
    model = YOLO(path_selected_model)
    predictions = []
    first_segment_image = None
    for idx, image in enumerate(imgs_upload):

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

    return predictions, first_segment_image

if __name__ == '__main__':
    st.write("## Classify and segment insects :fly:")
    st.write(
        "This app allows you to classify and segment insects. Use SOTA models for your entomological research."
    )
    st.write("### Preview of first selected image")
    st.sidebar.write("## Upload and download :gear:")

    col1, col2 = st.columns(2)

    # Upload an image
    imgs_upload = st.sidebar.file_uploader("Upload one or multiple image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    # Enter a Folder or Image path
    with st.sidebar:
        models_path = st.text_input(
            "Enter the folder that contains your models",
            placeholder=".pt or .pth file"
        )
        if models_path:
            path_selected_model = ta.gui_dashboard.button_model_selection.model_selector(models_path)
        else:
            path_selected_model = None
        
        if path_selected_model is None:
            st.error("Please select a valid directory that contains your models")
        else:
            st.success("Model selected")

    st.write("### Insights&Analysis single image")
    st.markdown("""---""")
    ia_col1, ia_col2 = st.columns(2)
    ia_col1.write("#### Color histogram")
    ia_col2.write("#### Area ratio")

    if imgs_upload is not None and path_selected_model is not None:
        upload_size = sum([img.size for img in imgs_upload])
        if upload_size > MAX_FILE_SIZE:
            st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
        else:
            predictions, first_segment_image = run_inference(imgs_upload=imgs_upload, path_selected_model=path_selected_model)

            if predictions:
                ta.gui_dashboard.show_results.show_color_histogram(first_prediction=predictions[0], col_to_display=ia_col1)
                ta.gui_dashboard.show_results.show_area_ratio(first_prediction=predictions[0], col_to_display=ia_col2)

                all_data = {
                        "data1.csv": BytesIO(b'a,b,c'), 
                        "data2.csv": BytesIO(b'd,e,f'),
                        "segmented_image.png": BytesIO(convert_image(first_segment_image))
                }
                ta.gui_dashboard.button_download.button_download_all(all_data=all_data)


    st.write("### Insights&Analysis multiple images")
    st.markdown("""---""")

