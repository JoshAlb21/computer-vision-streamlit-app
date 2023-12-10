# tachinidae_analyzer

## Introduction
Tachinidae Analyzer is a Streamlit-based web application designed for classifying and segmenting insects in images. This tool leverages state-of-the-art models to aid entomological research and analysis.

## Features
- Upload and process insect images.
- Perform image segmentation to identify insect parts.
- Display color histograms and area ratios for analyzed images.
- Download results in various formats, including segmented images and analysis data.


## Getting Started (GUI)

Note: The app runs locally on your computer after installation. This means that all data or images that you want to use for your analysis, including the results, are only stored locally and are not uploaded to a cloud (the "Upload" button is therefore somewhat misleading, as it only means uploading to the local application).

If you are interested in using a hosted version of the app, just contact me at the following email address: uvtoc@student.kit.edu. Even a self-hosted app internally in your organisation's network is easily possible due to Docker containers.

### Installation
To install Tachinidae Analyzer, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/JoshAlb21/tachinidae_analyzer
   ```
   OR
   simply download the repo as zip file (unzip after download)

2. Download Docker for your OS: [Docker](https://www.docker.com/get-started/) (makes sure that every user runs the same environment and requirements like python etc. are properly installed despite using a different OS.)
   
3. Setup required tools (by using a shell(.sh) script for MacOS or Linux and bash(.bat) script for Windows):
   
   MacOS and Linux:
   a. make the scripts executable
   ```
   chmod +x install_app.sh
   ```
   and 
   ```
   chmod +x start_app.sh
   ```

   b. Finally you can install the app (will also open it for the first time after installation)
   ```
   ./run_streamlit_app.sh
   ```

   Windows:
   Install the app (opens it at the first time) by
   Double Clicking the install_app.bat file

### Start the app (after succesfull installation)
The application is ready to use as soon as it is succesfully installed.

MacOS/Linux:
open a terminal inside the downloaded folder and type
```
   ./start_app.sh
```

Windows:
Double-Click on file
```
   start_app.bat
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

### Install updates
until now there is no option to install updates withouth re-installing the whole app. Make sure there is a new version of the app by checking the announcement on the github repo.
Then follow the steps described in section "Getting Started (GUI)" / "Installation".

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