import streamlit as st

import computer_vision_streamlit_app as ta


def disp_upload_section():
    st.sidebar.write("## Upload :gear:")

    # Upload an image
    imgs_upload = st.sidebar.file_uploader("Upload one or multiple image(s)", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)
    st.sidebar.success(f"{len(imgs_upload)} image(s) uploaded")

    return imgs_upload

def disp_model_settings():
    # Enter a Folder or Image path
    with st.sidebar:
        st.write("## Model :gear:")

        st.write("### Segmentation model")
        ##### Selection of segmentation model
        models_path_seg = st.text_input(
            "Enter the folder that contains your segmentation models",
            placeholder=".pt or .pth file",
            value = "data/models/YOLOv8_seg/"
        )
        
        #model_type_1 = st.radio("Choose model type",
        #                        options=["Segmentation", "Classification"],
        #                        key="model_type_selected",
        #                        horizontal=True)
        
        if models_path_seg:
            path_selected_model_seg = ta.gui_dashboard.button_model_selection.model_selector(models_path_seg)
        else:
            path_selected_model_seg = None

        if path_selected_model_seg is None:
            st.error("Please select a valid directory that contains your models")
        else:
            st.success("Model selected")
        
        st.write("### Classification model")
        ##### Selection of classification model
        models_path_cls = st.text_input(
            "Enter the folder that contains your classification models",
            placeholder=".pt or .pth file",
            value = "data/models/YOLOv8_cls/"
        )
        if models_path_cls:
            path_selected_model_cls = ta.gui_dashboard.button_model_selection.model_selector(models_path_cls)
        else:
            path_selected_model_cls = None

        if path_selected_model_cls is None:
            st.error("Please select a valid directory that contains your models")
        else:
            st.success("Model selected")

    return path_selected_model_seg, path_selected_model_cls

def disp_volume_est_settings():
    'Display the volume estimation settings on the sidebar'
    # Enter a Folder or Image path
    with st.sidebar:
        st.write("## Volume estimation :gear:")
        ##### Conversion factor 
        k_mm_per_px = st.text_input(
            "k [mm/pix] - conversion factor from pixel to mm",
            placeholder="e.g. 0.003118",
            value = 0.003118
        )
        if k_mm_per_px:
            try:
                k_mm_per_px = float(k_mm_per_px)
            except:
                st.error("Please enter a valid float number")
        else:
            st.info("If no conversion factor is entered, results are in pixels")
            k_mm_per_px = None
        
        ##### Degree of polynomial for fallback method
        n_polynom_fallback = st.text_input(
            "degree of polynomial for regression (fallback method in case of less than 2 CoGs)",
            placeholder="e.g. 2",
            value="2"
        )
        if n_polynom_fallback:
            try:
                n_polynom_fallback = int(n_polynom_fallback)
            except:
                st.error("Please enter a valid int number")
        else:
            st.error("No valid number entered. Default value 2 will be used.")
            n_polynom_fallback = 2

        ##### Number of orthogonal lines
        num_orthogonal_lines = st.text_input(
            "Number of orthogonal lines",
            placeholder="e.g. 150",
            value="150"
        )
        if num_orthogonal_lines:
            try:
                num_orthogonal_lines = int(num_orthogonal_lines)
            except:
                st.error("Please enter a valid int number")
        else:
            st.error("No valid number entered. Default value 150 will be used.")
            num_orthogonal_lines = 150

    return k_mm_per_px, n_polynom_fallback, num_orthogonal_lines
