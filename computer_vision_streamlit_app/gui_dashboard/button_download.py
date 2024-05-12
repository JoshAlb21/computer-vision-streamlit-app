from io import BytesIO
import base64
import streamlit as st
from datetime import datetime
from typing import Dict
import ultralytics
import json
import os

import computer_vision_streamlit_app as ta


def button_download_all(all_data:Dict[str, BytesIO]):
    '''
    Download all data as a zip file

    Args:
        all_data (Dict[str, BytesIO]): Dictionary with the key as the filename and the value as the data to be stored
            Example: {"test.csv": BytesIO(b"test,test,test")}
    '''

    # Set title and check boxes
    with st.sidebar:
        st.write("## Download :gear:")
        st.checkbox("json annotation", value=True, key="check_json_download")
        st.checkbox("img w. segments", value=True, key="check_img_w_segm_download")
        st.checkbox("img w. orth. lines", value=True, key="check_img_w_orth_lines_download")
        st.checkbox("information csv", value=True, key="df_combined_download")
    
    # Prepare data to download
    data_to_keep = {}
    data_to_keep['cog_object_parts.csv'] = all_data['cog_object_parts.csv']
    if st.session_state.check_img_w_segm_download:
        data_to_keep["segmented_image.png"] = all_data["segmented_image.png"]
    if st.session_state.check_img_w_orth_lines_download:
        data_to_keep["volume_estimation_img.png"] = all_data["volume_estimation_img.png"]
    if st.session_state.check_json_download:
        #data_to_keep["annotations.json"] = all_data["annotations"]
        data_to_keep["annotations.zip"] = all_data["annotations.zip"]
    if st.session_state.df_combined_download:
        data_to_keep["information.csv"] = all_data["information.csv"]

    # Prepare multiple files as a zip file
    zip_buffer = ta.gui_dashboard.utils.save_as_zip.buffer_in_zip(data_to_keep)

    with st.sidebar:
        st.sidebar.write("CoG for each object_part are downloaded by default. Select further files to download.")
        st.download_button(
            "Download csv", 
            file_name=f"results_{datetime.now()}.zip", 
            mime="application/zip", 
            data=zip_buffer
        )
