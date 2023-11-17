import matplotlib.pyplot as plt

def plot_color_histogram(hist_data, return_image=False):
    """
    Plot the color histograms for the RGB channels.

    Parameters:
        hist_data: dict
            A dictionary with keys 'red', 'green', and 'blue'. 
            Each key maps to a histogram for the respective color channel.
    """
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms for each channel
    plt.plot(hist_data['red'], color='red', label='Red Channel')
    plt.plot(hist_data['green'], color='green', label='Green Channel')
    plt.plot(hist_data['blue'], color='blue', label='Blue Channel')
    
    plt.title('Color Histograms')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    if return_image:
        return plt.gcf()
    else:
        plt.show()