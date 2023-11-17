import streamlit as st
import os

def model_selector(folder_path='.'):
    try:
        filenames = os.listdir(folder_path)
    except (NotADirectoryError, FileNotFoundError) as e:
        return None
    selected_filename = st.selectbox('Select a model', filenames)
    return os.path.join(folder_path, selected_filename)