import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
import json

class AnnotationVisualizer:
    
    def __init__(self, annotation_data, image_path):
        self.annotation_data = annotation_data
        self.image_path = image_path
        self.colors = {
            "fly": "red",
            "head": "green",
            "thorax": "blue",
            "abdomen": "yellow"
        }
        
    def get_color(self, label):
        """Retrieve color for a given label. If label not in predefined colors, generate a random color."""
        return self.colors.get(label, (random.random(), random.random(), random.random()))
        
    def visualize(self):
        img = Image.open(self.image_path)
        
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(img)

        for shape in self.annotation_data['shapes']:
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]
            width = x2 - x1
            height = y2 - y1

            color = self.get_color(shape['label'])
            rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, shape['label'], color='white', backgroundcolor=color)

        plt.axis('off')
        plt.tight_layout()
        plt.show()


class DualAnnotationVisualizer:
    
    def __init__(self, annotation_data1, annotation_data2, image_path):
        self.annotation_data1 = annotation_data1
        self.annotation_data2 = annotation_data2
        self.image_path = image_path
        self.colors = {
            "fly": "red",
            "head": "green",
            "thorax": "blue",
            "abdomen": "yellow"
        }
        
    def get_color(self, label):
        """Retrieve color for a given label. If label not in predefined colors, generate a random color."""
        return self.colors.get(label, (random.random(), random.random(), random.random()))
        
    def visualize(self):
        img = Image.open(self.image_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
        ax1.imshow(img)
        ax2.imshow(img)

        # Remove scales
        ax1.axis('off')
        ax2.axis('off')

        # Visualize annotations from the first JSON on the left subplot
        for shape in self.annotation_data1['shapes']:
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]
            width = x2 - x1
            height = y2 - y1

            color = self.get_color(shape['label'])
            rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor=color, facecolor='none')
            ax1.add_patch(rect)
            ax1.text(x1, y1, shape['label'], color='white', backgroundcolor=color)

        # Visualize annotations from the second JSON on the right subplot
        for shape in self.annotation_data2['shapes']:
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]
            width = x2 - x1
            height = y2 - y1

            color = self.get_color(shape['label'])
            rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
            ax2.text(x1, y1, shape['label'], color='white', backgroundcolor=color)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':

    path_to_json_file = "/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/prediction/ver2/CAS002_CAS0000157_stacked_01_H.json"
    path_to_image_file = "/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/img/ver2/CAS002_CAS0000157_stacked_01_H.jpg"


    # Example usage:
    with open(path_to_json_file, "r") as file:
        annotation_data = json.load(file)
    visualizer = AnnotationVisualizer(annotation_data, path_to_image_file)
    visualizer.visualize()

    # Visualize the annotations side by side without scales
    #visualizer = DualAnnotationVisualizer(annotation_data1, annotation_data2, "/mnt/data/CAS001_CAS0000001_stacked_01_H.png")
    #visualizer.visualize()
