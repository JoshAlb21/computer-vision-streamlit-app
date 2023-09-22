import os

from ultralytics import YOLO
from PIL import Image


def get_img_lst_from_dir(dir:str) -> list:
    img_lst = []
    for file_name in os.listdir(dir):
        if file_name.endswith('.jpg'):
            img_lst.append(dir + file_name)
    return img_lst

def inference_yolov8seg_on_folder(folder_path:str, model_path:str, limit_img:int=None):

    images = get_img_lst_from_dir(folder_path)
    model = YOLO(model_path)

    #***************
    # Inference
    #***************
    images = images[:limit_img]
    predictions = []
    for image in images:
        prediction = model.predict(image)
        predictions.append(prediction)

    print("Predictions")
    print(predictions)

    #predictions[0][0].plot()
    return predictions
