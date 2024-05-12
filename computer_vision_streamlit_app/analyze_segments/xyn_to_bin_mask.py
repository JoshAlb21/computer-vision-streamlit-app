import torch
import numpy as np
import cv2

def xyn_to_bin_mask(xyn_s:list[np.ndarray], w:int, h:int, image:np.ndarray) -> list[np.ndarray]:
    '''
    Convert a normalized mask to a binary mask.

    Parameters:
        xyn: (N, 2) torch.tensor
        w: image width
        h: image height
    '''       
    bin_masks = []
    for xyn in xyn_s:
        mask_points = xyn.copy()
        mask_points[:, 0] *= w
        mask_points[:, 1] *= h

        # Convert denormalized mask points to a binary mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [mask_points.astype(np.int32)], 1)
        bin_masks.append(mask)

    return bin_masks
