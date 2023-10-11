import tachinidae_analyzer as ta
import json
from PIL import Image
import numpy as np
import torch

#***************
# Load JSON
#***************

# Load JSON file
json_file_path = '/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/prediction/ver2/CAS002_CAS0000157_stacked_01_H.json'
with open(json_file_path) as f:
    annotation_data1 = json.load(f)

#***************
# Load image
#***************

# Load image
image_path = '/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/img/ver2/CAS002_CAS0000157_stacked_01_H.jpg'

# load image np.ndarray
img = Image.open(image_path)
img_np = np.array(img)  # Convert PIL Image to numpy array

boxes = []
masks = []
scores = []
all_cls = []

labels = {0: 'head', 1: 'thorax', 2: 'abdomen'}

for shape in annotation_data1['shapes']:
    
    # Extract bounding box from shape
    x_coords = [point[0] for point in shape['points']]
    y_coords = [point[1] for point in shape['points']]
    x1 = min(x_coords)
    y1 = min(y_coords)
    x2 = max(x_coords)
    y2 = max(y_coords)
    boxes.append([x1, y1, x2, y2])  # Now the box is in the [x1, y1, x2, y2] format


    mask = [[point[0] / img.width, point[1] / img.height] for point in shape['points']]
    masks.append(mask)
    scores.append(1)
    i_cls = list(labels.keys())[list(labels.values()).index(shape['label'])]
    all_cls.append(i_cls)

masks = [np.array(mask) for mask in masks]
all_cls = torch.Tensor(all_cls)

# Ensure you have implementations for color_generator and box_label
#ta.plotting.inference_results.plot_segments(img_np, boxes, masks, all_cls, labels, conf=scores, score=False)

bin_masks = ta.analyze_segments.xyn_to_bin_mask.xyn_to_bin_mask(masks, img.width, img.height, img_np)


#cog = ta.analyze_segments.calc_2d_cog_binary_mask.compute_2d_cog(bin_mask)
cogs = ta.analyze_segments.calc_2d_cog_binary_mask.compute_cogs(bin_masks, all_cls, labels)
    
#cog_s.append(cogs) 
print(cogs)


ta.plotting.inference_results.plot_segments_w_cogs(img_np, boxes, masks, all_cls, labels, cogs=cogs, alpha=0.5)