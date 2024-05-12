import computer_vision_streamlit_app as ta
import json
from PIL import Image
import numpy as np
import torch
import os

json_path = '/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/data/prediction/ver2'
image_path = '/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/data/img/ver2'

target_plot_dir = '/Users/joshuaalbiez/Documents/masterthesis_iai/generated_plots/skeleton_extraction'


all_json_files = os.listdir(json_path)
all_img_files = os.listdir(image_path)
file_names_wo_ending = [file_name.split('.')[0] for file_name in all_json_files]

reduce_to = 20
file_names_wo_ending = file_names_wo_ending[10:reduce_to]

for file_name in file_names_wo_ending:
    json_file_path = os.path.join(json_path, file_name + '.json')
    img_file_path = os.path.join(image_path, file_name + '.jpg')

    #***************
    # Load JSON
    #***************

    # Load JSON file
    with open(json_file_path) as f:
        annotation_data1 = json.load(f)

    #***************
    # Load image
    #***************

    # load image np.ndarray
    img = Image.open(img_file_path)
    img_np = np.array(img)  # Convert PIL Image to numpy array

    boxes = []
    masks = []
    scores = []
    all_cls = []

    labels = {0: 'dog', 1: 'bicycle', 2: 'umbrella'}

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

    bin_masks = ta.analyze_segments.xyn_to_bin_mask.xyn_to_bin_mask(masks, img.width, img.height, img_np)

    #***************
    # Visualize cogs
    #***************
    cogs = ta.analyze_segments.calc_2d_cog_binary_mask.compute_cogs(bin_masks, all_cls, labels)
    ta.plotting.inference_results.plot_segments_w_cogs(img_np, boxes, masks, all_cls, labels, cogs=cogs, alpha=0.5)

    cogs_list = []
    for cog_i in range(3):
        try:
            cog_tuple = cogs[str(cog_i)]
            cogs_list.append(cog_tuple)
        except:
            pass
    cogs_array = np.array(cogs_list)

    ordered_cogs = ta.plotting.inference_results.order_cog_dict(cogs, max_i=3)
    print(ordered_cogs)

    #***************
    # Compute skeleton
    #***************

    n_polynom = 2 #5
    weight = 20
    num_points = 3000
    cog_weights = np.array([weight]*cogs_array.shape[0])
    # Compute skeleton
    generator = ta.extract_skeleton.polynom_regression_in_mask.MaskPointGenerator(bin_masks, cogs_array, cog_weights, num_points)
    generator.fit_polynomial(degree=n_polynom)
    combined_mask = generator.get_combined_mask()
    random_points = generator.get_points()

    # save random_points as pickle
    # save combined_mask as pickle
    #import pickle
    #pickle.dump( random_points, open( "random_points2.pickle", "wb" ) )
    #exit()
    # old approach ordinary linear regression
    '''
    # Get (x, y) pairs for all x coordinates in the combined mask
    fitted_points = generator.get_fitted_points()
    '''
    #fitted_points = generator.get_fitted_points()
    # Use ODR instead of OLS
    #fitted_points = generator.fit_get_odr(degree=n_polynom)
    #fitted_points = generator.interpolate_points(given_points=cogs_array, kind="quadratic")
    #fitted_points = generator.interpolate_points_spline(given_points=cogs_array)
    #fitted_points = generator.parametric_polynomial_regression(degree=n_polynom)

    # New methhod where points must be ordered

    #***************
    # Method 1
    #***************
    orderer = ta.extract_skeleton.point_orderer.PointOrderer(ordered_cogs)
    ordered_points = orderer.order_points(random_points)
    ordered_points = np.array(ordered_points)
    x_values = ordered_points[:, 0]
    y_values = ordered_points[:, 1]
    #fitted_points = generator.parametric_regression(x_values, y_values, degree=n_polynom)

    print("extended points:", np.array(orderer.extended_reference_points))

    #***************
    # Method 2
    #***************
    reference_points_extended = np.array(orderer.extended_reference_points)
    refiner = ta.extract_skeleton.line_refiner.GraphRefiner(reference_points_extended, ordered_points, alpha=1, iterations=500)
    optimized_points = refiner.get_refined_points()
    trimmed_points = refiner.trim_by_mask(combined_mask)
    #fitted_points = trimmed_points

    #***************
    # Method 3
    #***************
    fitted_points = generator.interpolate_points_parametric_spline(given_points=cogs_array)
    fitted_points = ta.extract_skeleton.line_refiner.trim_line(combined_mask, fitted_points)

    #***************
    # Plot skeleton
    #***************
    drawer = ta.extract_skeleton.plot_skeleton.LineDrawer(fitted_points, combined_mask, bin_mask=True, scatter_points=None)#random_points)
    #drawer = ta.extract_skeleton.plot_skeleton.LineDrawer(fitted_points, img_np, bin_mask=False)
    drawer.draw_line()
    drawer.show_image()
    #drawer.save_image(target_plot_dir, file_name + '_skeleton.png')

    