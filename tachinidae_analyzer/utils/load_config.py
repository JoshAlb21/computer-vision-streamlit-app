import os
import json

def load_config(config_file: str="config.json") -> dict:

    config_path = os.path.join(config_file)
    with open(config_path) as json_data_file:
        config = json.load(json_data_file)
    return config
