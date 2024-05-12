import cv2
import numpy as np

def combine_masks(masks: dict) -> dict:
    """Combine multiple masks (stored in dict value) per segment (stored in dict key) into a single mask and return a dictionary."""
    
    combined_masks_dict = {}
    
    for seg, mask_lst in masks.items():
        # Initialize a mask with zeros using the shape of the first mask in the list
        if not mask_lst:
            continue
        combined_mask = np.zeros_like(mask_lst[0], dtype=np.uint8)
        
        for mask in mask_lst:
            # Check and raise an error if any mask doesn't match the expected shape
            if mask.shape != combined_mask.shape:
                raise ValueError(f"Mask in segment {seg} has a mismatched shape.")
            
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        combined_masks_dict[seg] = combined_mask
            
    return combined_masks_dict
