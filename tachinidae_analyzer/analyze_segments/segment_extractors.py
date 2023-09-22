import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd

class BaseSegmentExtractor(ABC):
    def __init__(self, image, mask):
        """
        Initialize the BaseSegmentExtractor.

        Parameters:
        - image: Input image.
        - mask: Binary mask representing the segment.
            single mask belonging to a single segment class.
        """
        self.image = image
        self.mask = mask

    @abstractmethod
    def _extract_segment(self):
        """
        Abstract method to extract the segment from the image using the mask.
        """
        pass

class SegmentArea(BaseSegmentExtractor):
    def _extract_segment(self):
        """
        Extract the segment from the image using the mask.
        """
        segment = cv2.bitwise_and(self.image, self.image, mask=self.mask)
        return segment
    
    def calculate_area(self):
        """
        Calculate the area of the segment.
        """
        return np.sum(self.mask)


class SegmentColor(BaseSegmentExtractor):

    def _extract_segment(self):
        """
        Extract the segment from the image using the mask.
        """
        segment = cv2.bitwise_and(self.image, self.image, mask=self.mask)
        return segment

    def calculate_average_color(self):
        """
        Calculate the average color of the segment.
        """
        segment = self._extract_segment()
        total_pixels = np.sum(self.mask)
        avg_color = np.sum(segment, axis=(0,1)) / total_pixels
        return avg_color
    
    def calculate_color_histogram(self, bins=256) -> dict:
        """
        Calculate the color histogram for the segment.
        
        Parameters:
            bins: int
                Number of histogram bins.
                
        Returns:
            hist: dict
                A dictionary with keys 'red', 'green', and 'blue'. 
                Each key maps to a histogram for the respective color channel.
        """
        segment = self._extract_segment()
        
        # Compute histograms for each channel
        red_hist = cv2.calcHist([segment], [0], self.mask, [bins], [0,256])
        green_hist = cv2.calcHist([segment], [1], self.mask, [bins], [0,256])
        blue_hist = cv2.calcHist([segment], [2], self.mask, [bins], [0,256])
        
        hist = {
            'red': red_hist,
            'green': green_hist,
            'blue': blue_hist
        }
        
        return hist
    
    @staticmethod
    def extract_histogram_statistics(hist_data, ret_as_df:bool=True) -> Union[dict, pd.DataFrame]:
        """
        Extract statistics from a color histogram.

        Parameters:
            hist_data: dict
                A dictionary with keys 'red', 'green', and 'blue'. 
                Each key maps to a histogram for the respective color channel.
            ret_as_df: bool
                If True, return the statistics as a pandas.DataFrame.
                If False, return the statistics as a dictionary.

        Returns:
            A dictionary containing statistics for each channel.
        """
        stats = {}
        for channel, histogram in hist_data.items():
            stats[channel] = {
                "min_val": np.min(histogram),
                "max_val": np.max(histogram),
                "std": np.std(histogram),
                "mean_val": np.mean(histogram),
                "mean_num": np.mean(np.nonzero(histogram)),
                "max_num": np.max(np.nonzero(histogram))
            }

        if ret_as_df:
            flattened_data = {f"{color}_{stat}": value for color, stats_dict in stats.items() for stat, value in stats_dict.items()}
            stats = pd.DataFrame(flattened_data, index=[0])

        return stats
    

def segment_area_comparison(segment_areas_dict: dict, ret_as_df:bool=True) -> Union[dict, pd.DataFrame]:
    """
    Calculate the area of each segment and the ratio of each segment to the total area.
    Also calculate the ratio of each segment to each other segment.

    Parameters:
    - segment_areas_dict: Dictionary of segment areas. Keys are the segment labels and values are SegmentArea objects.
        example = {
            "head": [SegmentArea(...), ]
            "abdomen": [SegmentArea(...), ]
            "thorax": [SegmentArea(...), ]
        }
    """

    ratios = {}
    total_area = 0
    
    # Calculate total area by summing all segment areas
    for label, segment_area_obj in segment_areas_dict.items():
        for multi_seg in segment_area_obj: # each cls can have multiple segments
            area = multi_seg.calculate_area()
            total_area += area
    
    # Calculate ratios
    for label, segment_area_obj in segment_areas_dict.items():
        area = 0
        for multi_seg in segment_area_obj:
            area += multi_seg.calculate_area()
        ratio_to_total = area / total_area
        ratios[f"{label}_total"] = ratio_to_total
        
        # Comparing with other segments
        for other_label, other_segment_area_obj in segment_areas_dict.items():
            if label != other_label:
                other_area = 0
                for multi_seg in other_segment_area_obj:
                    other_area += multi_seg.calculate_area()
                ratio = area / other_area
                ratios[f"{label}_{other_label}"] = ratio

    if ret_as_df:
        ratios = pd.DataFrame(ratios, index=[0])

    return ratios

def segment_color_comparison(segment_colors_dict: dict, bins:int=256, ret_as_df:bool=True) -> Union[dict, pd.DataFrame]:
    """
    Calculate the color histogram for each segment.

    Parameters:
    - segment_colors_dict: Dictionary of segment colors. Keys are the segment labels and values are SegmentColor objects.
        example = {
            "head": [SegmentColor(...), ]
            "abdomen": [SegmentColor(...), ]
            "thorax": [SegmentColor(...), ]
        }
    - bins: Number of histogram bins.
    """

    histograms = {}
    
    # Calculate histograms for each segment
    for label, segment_color_objs in segment_colors_dict.items():
        aggregated_histogram = {
            'red': [],
            'green': [],
            'blue': []
        }
        
        for segment_color_obj in segment_color_objs:
            hist = segment_color_obj.calculate_color_histogram(bins=bins)
            
            # Aggregate histograms (if there are multiple segments of the same label)
            #TODO maybe we should aggreate the SegmentColor objects instead of the histograms?
            for channel in ['red', 'green', 'blue']:
                aggregated_histogram[channel].extend(hist[channel])
                
        histograms[label] = aggregated_histogram

    # If required, convert histograms to DataFrame
    if ret_as_df:
        flattened_data = {f"{segment}_{channel}": hist for segment, channels in histograms.items() for channel, hist in channels.items()}
        histograms = pd.DataFrame(flattened_data)

    return histograms

