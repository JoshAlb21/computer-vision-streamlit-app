from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from typing import Dict

import tachinidae_analyzer as ta

def buffer_in_zip(data_to_save:Dict[str, BytesIO]) -> BytesIO:
    '''
    Create a zip file.
    Note: only way to download multiple files at once (at least for now)
    '''
    zip_buffer = BytesIO()

    with ZipFile(zip_buffer, "a", ZIP_DEFLATED, False) as zip_file:
        for file_name, data in data_to_save.items():
            if isinstance(data, BytesIO):
                data = data.getvalue()
            zip_file.writestr(file_name, data)

    return zip_buffer