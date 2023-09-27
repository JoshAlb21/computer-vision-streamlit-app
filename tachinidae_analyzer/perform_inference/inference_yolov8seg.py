import os

from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

import tachinidae_analyzer as ta


def get_img_lst_from_dir(dir:str) -> list:
    img_lst = []
    for file_name in os.listdir(dir):
        if file_name.endswith('.jpg'):
            img_lst.append(dir + file_name)
    return img_lst

def inference_yolov8seg_on_folder(folder_path:str, model_path:str, limit_img:int=None, save_txt:bool=False, save_dir:str=None):

    ta.prepare_data.fix_broken_jpg.detect_and_fix_dir(folder_path)
    images = get_img_lst_from_dir(folder_path)
    model = YOLO(model_path)

    #***************
    # Inference
    #***************
    if limit_img is not None:
        images = images[:limit_img]
    predictions = []
    print("Predicting...")
    for image in tqdm(images):
        if save_dir:
            prediction = model.predict(image, save_txt=save_txt, save_dir=save_dir)
        else:
            prediction = model.predict(image, save_txt=save_txt)
        predictions.append(prediction)

    print("Predictions")
    print(predictions)

    #predictions[0][0].plot()
    return predictions
