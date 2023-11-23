import streamlit as st
import os

def model_selector(folder_path='.'):
    try:
        filenames = os.listdir(folder_path)
    except (NotADirectoryError, FileNotFoundError) as e:
        return None
    selected_filename = st.selectbox('Select a model', filenames)

    if selected_filename is None:
        model_path = None
    else:
        if selected_filename.endswith(".pt") or selected_filename.endswith(".pth"):
            model_path = os.path.join(folder_path, selected_filename)
        else:
            model_path = None
            st.error("Please select a valid model file (.pt or .pth)")
            
    return model_path