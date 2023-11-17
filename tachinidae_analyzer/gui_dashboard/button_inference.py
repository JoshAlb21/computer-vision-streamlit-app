from PIL import Image

from tachinidae_analyzer.perform_inference.inference_yolov8seg import inference_yolov8seg_on_folder
from tachinidae_analyzer.plotting.inference_results import plot_segments_from_results


def button_inference_folder(folder_path:str, model_path:str):
    folder_path = "/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/img/ver1/"
    model_path = "/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/models/YOLOv8_seg/weights/last.pt"

    predictions = inference_yolov8seg_on_folder(folder_path, model_path, limit_img=1)

    image = plot_segments_from_results(predictions[0][0], return_image=True)
    image = Image.fromarray(image)

    if image is None:
        raise ValueError("Predicted image is None")

    return image
