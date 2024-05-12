'''
This script contains the class that loads the CSV file and extracts the
relevant columns. The class also extracts the metadata from the first two
rows of the CSV file.

Analyze csv files that contain information about the objects.
(for each image ID)
'''

import pandas as pd

class TachinidCSVLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.metadata = self._load_metadata()
        self.data = self._load_csv()
        
    def _load_metadata(self):
        """Load metadata from the first two rows."""
        metadata_df = pd.read_csv(self.file_path, nrows=2, header=None)
        metadata = {}
        for _, row in metadata_df.iterrows():
            key, value = row[0].split(':'), row[1]
            metadata[key[0].strip()] = value.strip() if isinstance(value, str) else value
        return metadata
        
    def _load_csv(self):
        """Load the specified columns starting from the fourth row, if they exist."""
        # Check available columns in the CSV
        available_columns = pd.read_csv(self.file_path, skiprows=3, nrows=0).columns.tolist()
        
        columns_to_extract = [
            "Species", "Age", "MalaiseDate", "SpecimenCode/Filename", 
            "Ventral", "Dorsal", "Lateral(left)", "Lateral(right)"
        ]
        
        # Handle the variation in the 'MalaiseDate' column name
        if "DataMalaise" in available_columns:
            columns_to_extract[columns_to_extract.index("MalaiseDate")] = "DataMalaise"
        
        # Only load columns that exist in the CSV
        columns_to_load = [col for col in columns_to_extract if col in available_columns]
        data = pd.read_csv(self.file_path, skiprows=3, usecols=columns_to_load)
        
        # Rename "DataMalaise" back to "MalaiseDate" for consistency
        if "DataMalaise" in data.columns:
            data.rename(columns={"DataMalaise": "MalaiseDate"}, inplace=True)
        
        return data

    def get_data(self):
        """Return the loaded data."""
        return self.data

    def get_metadata(self):
        """Return the loaded metadata."""
        return self.metadata
    
class TachinidCSVMerger:
    
    @staticmethod
    def merge_data(loaders_list):
        """Merge the data from a list of TachinidCSVLoader instances."""
        dataframes = [loader.get_data() for loader in loaders_list]
        merged_data = pd.concat(dataframes, ignore_index=True)
        return merged_data

    @staticmethod
    def merge_metadata(loaders_list):
        """Merge the metadata from a list of TachinidCSVLoader instances."""
        merged_metadata = {}
        for loader in loaders_list:
            for key, value in loader.get_metadata().items():
                # If the key already exists and has a different value, append the new value
                if key in merged_metadata and merged_metadata[key] != value:
                    merged_metadata[key] = f"{merged_metadata[key]}, {value}"
                else:
                    merged_metadata[key] = value
        return merged_metadata
