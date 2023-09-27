import json
import os

from tqdm import tqdm


class AnnotationConverter:
    def __init__(self, img_width: int, img_height: int):
        self.img_width = img_width
        self.img_height = img_height

    def txt_to_json(self, txt_file_path: str, json_output_path: str):
        """
        Convert TXT annotation file to JSON annotation file.
        """
        with open(txt_file_path, 'r') as file:
            txt_content = file.readlines()

        # Convert TXT content to JSON format
        json_content = {
            "version": "4.5.7",
            "flags": {},
            "shapes": [],
            "imagePath": "",
            "imageHeight": self.img_height,
            "imageWidth": self.img_width,
            "lineColor": [0, 255, 0, 128],
            "fillColor": [255, 0, 0, 128]
        }

        for line in txt_content:
            coords = list(map(float, line.strip().split()))
            obj_id = int(coords[0])  # Extract object ID
            points = [(coords[i] * self.img_width, coords[i+1] * self.img_height) for i in range(1, len(coords)-1, 2)]
            shape = {
                "label": "1",
                "points": points,
                "group_id": obj_id,
                "shape_type": "polygon",
                "flags": {}
            }
            json_content['shapes'].append(shape)

        # Write JSON content to file
        with open(json_output_path, 'w') as json_file:
            json.dump(json_content, json_file, indent=4)

    def json_to_txt(self, json_file_path: str, txt_output_path: str):
        """
        Convert JSON annotation file to TXT annotation file.
        """
        with open(json_file_path, 'r') as json_file:
            json_content = json.load(json_file)

        # Convert JSON content to TXT format
        txt_content = ""
        for shape in json_content['shapes']:
            obj_id = 1  # Set object ID to 1 as per user's instruction
            points = shape['points']
            normalized_coords = [coord / dim for point in points for coord, dim in zip(point, [self.img_width, self.img_height])]
            txt_content += f"{obj_id} " + " ".join(map(str, normalized_coords)) + "\n"

        # Write TXT content to file
        with open(txt_output_path, 'w') as txt_file:
            txt_file.write(txt_content)


class DirAnnotationConverter:
    def __init__(self, img_width: int, img_height: int):
        self.converter = AnnotationConverter(img_width, img_height)

    def txt_to_json_dir(self, txt_dir: str, json_output_dir: str):
        """
        Convert all TXT annotation files within a directory to JSON format.
        """
        # Create output directory if it doesn't exist
        os.makedirs(json_output_dir, exist_ok=True)
        
        # Iterate over all TXT files in the directory and convert them to JSON
        for file_name in tqdm(os.listdir(txt_dir)):
            if file_name.endswith('.txt'):
                txt_file_path = os.path.join(txt_dir, file_name)
                json_file_name = file_name.replace('.txt', '.json')
                json_output_path = os.path.join(json_output_dir, json_file_name)
                self.converter.txt_to_json(txt_file_path, json_output_path)

    def json_to_txt_dir(self, json_dir: str, txt_output_dir: str):
        """
        Convert all JSON annotation files within a directory to TXT format.
        """
        # Create output directory if it doesn't exist
        os.makedirs(txt_output_dir, exist_ok=True)
        
        # Iterate over all JSON files in the directory and convert them to TXT
        for file_name in tqdm(os.listdir(json_dir)):
            if file_name.endswith('.json'):
                json_file_path = os.path.join(json_dir, file_name)
                txt_file_name = file_name.replace('.json', '.txt')
                txt_output_path = os.path.join(txt_output_dir, txt_file_name)
                self.converter.json_to_txt(json_file_path, txt_output_path)

if __name__ == '__main__':
    # Initialize the converter with the provided image dimensions
    converter = AnnotationConverter(img_width=4032, img_height=3040)

    # Convert TXT to JSON and save the result to a new file
    json_output_path = '/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/prediction/ver2'
    txt_file_path = '/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/runs/segment/predict/labels'

    dir_converter = DirAnnotationConverter(img_width=4032, img_height=3040)
    dir_converter.txt_to_json_dir(txt_dir=txt_file_path, json_output_dir=json_output_path)
