import cv2
import itertools
import torch
from typing import Union
import numpy as np


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    font_scale_inverse = 3
    lw = max(round(sum(image.shape) / font_scale_inverse * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, cv2.FONT_HERSHEY_TRIPLEX, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
    return image


def color_generator():
    # Define a list of colors
    # Use red, blue, greeen, yellow, orange, purple, brown, pink
    base_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 165, 0), (128, 0, 128), (165, 42, 42), (255, 192, 203)]
    return itertools.cycle(base_colors)

def plot_segments(image, boxes, masks, cls:torch.tensor, labels:dict, conf:Union[torch.tensor, np.ndarray], score:bool=False, alpha=0.5):
    colors = color_generator()
    cls = cls.numpy()
    h, w = image.shape[:2]

    for i, (box, mask_points_normalized) in enumerate(zip(boxes, masks)):
        if score:
            label = labels[cls[i]] + " " + str(round(100 * float(conf[i]), 1)) + "%"
        else:
            label = labels.get(cls[i], "Unknown")

        color = next(colors)
        
        # Draw bounding box
        box_label(image, box, label, color)

        # Denormalize mask points
        mask_points = mask_points_normalized.copy()
        mask_points[:, 0] *= w
        mask_points[:, 1] *= h

        # Convert denormalized mask points to a binary mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [mask_points.astype(np.int32)], 1)

        # Draw segmentation mask
        overlay = image.copy()
        for c in range(3):  # For each RGB channel
            overlay[..., c] = np.where(mask == 1, color[c], overlay[..., c])
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    cv2.imshow('Segments', image)
    cv2.waitKey(0)

# Official ultralytics.engine.results.Results.plot() method
# Show the results
'''
for r in predictions[0]:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image

'''


def plot_segments_from_results(result):
    """
    Plot bounding boxes and segmentation masks using a ultralytics.engine.results.Results object.

    Parameters:
    - result: An instance of ultralytics.engine.results.Results
    """
    # Extracting necessary attributes from the result object
    image = result.orig_img.copy()
    boxes = result.boxes.xyxy
    masks = result.masks.xyn
    cls = result.boxes.cls
    labels = result.names
    conf = result.boxes.conf
    score = True if conf is not None else False

    # Visualize bounding boxes and segmentations
    plot_segments(image, boxes, masks, cls, labels, conf, score)
