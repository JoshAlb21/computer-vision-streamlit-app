from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO
import base64
import streamlit as st
from datetime import datetime
from typing import Dict


def button_download_all(all_data:Dict[str, BytesIO]):
    '''
    Download all data as a zip file

    Args:
        all_data (Dict[str, BytesIO]): Dictionary with the key as the filename and the value as the data to be stored
            Example: {"test.csv": BytesIO(b"test,test,test")}
    '''

    zip_buffer = BytesIO()

    with ZipFile(zip_buffer, "a", ZIP_DEFLATED, False) as zip_file:
        for file_name, data in all_data.items():
            zip_file.writestr(file_name, data.getvalue())

    st.download_button(
        "Download csv", 
        file_name=f"results_{datetime.now()}.zip", 
        mime="application/zip", 
        data=zip_buffer
    )
