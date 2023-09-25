import tachinidae_analyzer as ta

merged_path = "/Users/joshuaalbiez/Documents/python/tachinidae_analyzer/data/dataset_csv/merged.csv"
    
# Load csv files and merge them
loaders = [ta.dataset_csv.extract_from_dataset_csv.TachinidCSVLoader(f"/Users/joshuaalbiez/Documents/python/tachinidae_detector/data/csv_data/TachinidaeID_export - CAS00{i}.csv") for i in range(1, 6)]
merged_data = ta.dataset_csv.extract_from_dataset_csv.TachinidCSVMerger.merge_data(loaders)
merged_metadata = ta.dataset_csv.extract_from_dataset_csv.TachinidCSVMerger.merge_metadata(loaders)

merged_data.to_csv(merged_path, index=False)