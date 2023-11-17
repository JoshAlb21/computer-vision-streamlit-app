# tachinidae_analyzer
Focus on inference mode of DL models. Providing insights

## Run withouth installation

python -m tachinidae_analyzer.perform_inference.inference_yolov8seg

## Developing

pip install -e .

## Installing dependecies

for focus stacking

1. install cython:
brew install cython
pip install Cython

python -m pip install pygco

## Examples

1. run ex_yolov8seg_prediction_extraction -> result: df_segments.csv
2. run ex_yolov8seg_segments_analyzer -> df_segments_dataset.csv (df_segments.csv + TachinidaeID_export... .csv)
3. run ex_volume_estimation -> result: df_volumes.csv
4. run analyze_extracted_volumes -> result: df_merged.csv