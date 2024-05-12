import ultralytics
import json
import os

import computer_vision_streamlit_app as ta

def zip_all_annotations(predictions:list[list[ultralytics.engine.results.Results]], img_names:list[str]):
    """
    Zip the results of a folder as JSON files.
    """
    annot_data = {}
    for prediction, img_name in zip(predictions, img_names):
        converter = ta.gui_dashboard.utils.converter_result_to_json.ResultsConverter(prediction[0], img_name=img_name)
        json_anno_data = converter.get_as_json()
        json_string_anno = json.dumps(json_anno_data, indent=4)

        base_name = os.path.splitext(img_name)[0]

        annot_data[f'{base_name}.json'] = json_string_anno
    zip_buffer = ta.gui_dashboard.utils.save_as_zip.buffer_in_zip(annot_data)

    return zip_buffer