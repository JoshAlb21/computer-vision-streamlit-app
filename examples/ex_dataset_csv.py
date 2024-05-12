import computer_vision_streamlit_app as ta

merged_path = "/Users/joshuaalbiez/Documents/python/computer_vision_streamlit_app/data/dataset_csv/merged.csv"
    
# Load csv files and merge them
loaders = [ta.dataset_csv.extract_from_dataset_csv.TachinidCSVLoader(f"/Users/joshuaalbiez/Documents/python/object_detector/data/csv_data/ID_export - CAS00{i}.csv") for i in range(1, 6)]
merged_data = ta.dataset_csv.extract_from_dataset_csv.TachinidCSVMerger.merge_data(loaders)
merged_metadata = ta.dataset_csv.extract_from_dataset_csv.TachinidCSVMerger.merge_metadata(loaders)

merged_data.to_csv(merged_path, index=False)