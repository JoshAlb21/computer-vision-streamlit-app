import numpy as np
import matplotlib.pyplot as plt

def visualize_bin_mask(mask):
    """
    Visualize a binary mask using matplotlib.

    Parameters:
        mask (np.ndarray): Binary mask to visualize.
    """
    plt.imshow(mask, cmap='gray')
    plt.title('Binary Mask Visualization')
    plt.axis('off')
    plt.show()