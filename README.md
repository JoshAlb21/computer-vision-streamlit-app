# tachinidae_analyzer

## Introduction
Tachinidae Analyzer is a Streamlit-based web application designed for classifying and segmenting insects in images. This tool leverages state-of-the-art models to aid entomological research and analysis.

## Features
- Upload and process insect images.
- Perform image segmentation to identify insect parts.
- Display color histograms and area ratios for analyzed images.
- Download results in various formats, including segmented images and analysis data.


## Getting Started (GUI)

### Installation
To install Tachinidae Analyzer, follow these steps:

1. Clone the repository:
   ```
   git clone [repository URL]
   ```
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```
   streamlit run main_dashboard.py
   ```

### Overview
The sidebar is used to adjust different settings. Here you can choose the image(s) you want to perform inference on. Given the path where you store your models (pt or pth files), you will be able to select the model you want to use.
After performing the inference you will be able to select which files you want to download.

in case you want to perform the prediction for reintegration purposes of new data, you should download the json annotation files. These you can use with the corresponding image file (has to have the same name and should be stored within same folder!) to annotate them with the opensource software "labelme".

### Usage
After starting the application, follow these steps:

1. Upload insect images via the sidebar.
2. Select the model for classification and segmentation.
3. View the segmentation results and analysis in the main dashboard.
4. Download the results and segmented images as needed.

## Getting Started (with programming knowledge)

### Examples
The examples folder contains various python script that can be used as inspiration for your own project (Note: some of them are not mantained anymore!)

### Contributing or Developing
Contributions to Tachinidae Analyzer are welcome. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature.
3. Submit a pull request with a detailed description of your changes.

pip install -e .

### Run without installation

python -m tachinidae_analyzer.perform_inference.inference_yolov8seg

### Create new version of standalone app
1. create package.json file
2. npm install
3. npm run dump tachinidae_analyzer
4. npm run dist

### Examples

1. run ex_yolov8seg_prediction_extraction -> result: df_segments.csv
2. run ex_yolov8seg_segments_analyzer -> df_segments_dataset.csv (df_segments.csv + TachinidaeID_export... .csv)
3. run ex_volume_estimation -> result: df_volumes.csv
4. run analyze_extracted_volumes -> result: df_merged.csv

### Run docker container
docker build -t streamlit .
docker run -p 8501:8501 streamlit
docker container ls --all
docker export 68b0fa2d9905 > streamlit.tar