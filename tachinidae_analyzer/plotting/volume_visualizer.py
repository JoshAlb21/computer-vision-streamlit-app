import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import Dict, List, Tuple


class BodyVolumeVisualizerOLD:
    
    def __init__(self, lines: Dict[str, List[List[Tuple[int, int]]]]):
        """Initialize the visualizer with the orthogonal lines.
        
        Args:
            lines: A dictionary of body parts and their orthogonal lines.
                Example: {'head': [[(3130.2561840489834, 663),
                            (3131.0021849901395, 664), ...], ...], ...}
        """
        self.lines = lines
        self.theta = np.linspace(0, 2 * np.pi, 201)
    
    @staticmethod
    def calculate_distance(point1, point2):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def plot_circles_from_midpoints(self, ax, lines_for_part, color, label=None):
        """Plot circles for a given body part using the midpoints of the orthogonal lines."""
        for idx, line in enumerate(lines_for_part):
            if len(line) > 1:  # Ensure the line isn't empty or a single point
                mid_point = line[len(line) // 2]
                radius = self.calculate_distance(mid_point, line[0])
                
                y_circle = mid_point[0] + radius * np.cos(self.theta)
                z_circle = mid_point[1] + radius * np.sin(self.theta)
                x_circle = np.full_like(y_circle, (line[0][0] + line[1][0]) / 2)  # Average X-coordinate
                
                # Add label only for the first circle to avoid repetition in the legend
                ax.plot(x_circle, y_circle, z_circle, color=color, alpha=0.6, label=label if idx == 0 else "")
    
    def visualize(self):
        """Visualize the circles for each body part in 3D."""
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['red', 'green', 'blue']
        for idx, (body_part, lines_for_part) in enumerate(self.lines.items()):
            self.plot_circles_from_midpoints(ax, lines_for_part, colors[idx], label=body_part)

        # Setting labels, legend, and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title("3D Visualization of Circles Using Midpoints of Orthogonal Lines")

        plt.show()

class BodyVolumeVisualizer:
    
    def __init__(self, lines: Dict[str, List[List[Tuple[int, int]]]]):
        """Initialize the visualizer with the orthogonal lines.
        
        Args:
            lines: A dictionary of body parts and their orthogonal lines.
                Example: {'head': [[(3130.2561840489834, 663),
                            (3131.0021849901395, 664), ...], ...], ...}
        """
        self.lines = lines
        self.theta = np.linspace(0, 2 * np.pi, 201)
    
    @staticmethod
    def calculate_distance(point1, point2):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def calculate_slope(point1, point2):
        """Calculate the slope of the line formed by two points."""
        # Avoid division by zero
        if point2[0] - point1[0] == 0:
            return np.inf
        return (point2[1] - point1[1]) / (point2[0] - point1[0])
    
    def plot_circles_from_midpoints(self, ax, lines_for_part, color, label=None):
        """Plot circles for a given set of lines using the midpoints of the orthogonal lines."""
        for idx, line in enumerate(lines_for_part):
            if len(line) > 1:  # Ensure the line isn't empty or a single point
                mid_point = line[len(line) // 2]
                radius = self.calculate_distance(mid_point, line[0])
                
                slope = self.calculate_slope(line[0], line[1])
                angle = np.arctan(slope)
                
                x_circle = mid_point[0] + radius * np.sin(self.theta) * np.cos(angle)
                y_circle = mid_point[1] + radius * np.sin(self.theta) * np.sin(angle)
                z_circle = mid_point[1] + radius * np.cos(self.theta)  # All circles share the same Z value

                z_circle -= mid_point[1]
                
                # Add label only for the first circle to avoid repetition in the legend
                ax.plot(x_circle, y_circle, z_circle, color=color, alpha=0.6, label=label if idx == 0 else "")
    
    def visualize(self, return_fig:bool=False):
        """Visualize the circles for each body part in 3D."""
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['red', 'green', 'blue']
        for idx, (body_part, lines_for_part) in enumerate(self.lines.items()):
            self.plot_circles_from_midpoints(ax, lines_for_part, colors[idx], label=body_part)

        # Setting labels, legend, and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set aspect ratio and tick interval
        max_limit = max(ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1])
        ax.set_xlim(0, max_limit)
        ax.set_ylim(0, max_limit)
        ax.set_zlim(0, 500)
        ax.set_xticks(np.arange(0, max_limit+1, 200))
        ax.set_yticks(np.arange(0, max_limit+1, 200))
        ax.set_zticks(np.arange(0, max_limit+1, 200))

        ax.legend()
        ax.set_title("3D Visualization of Circles Using Midpoints of Orthogonal Lines")

        if return_fig:
            return plt.gcf()
        else:
            plt.show()