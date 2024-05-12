'''
DEPRECATED
'''

from io import BytesIO
import json

# Assuming 'output_json' is the JSON data you want to convert to bytes
# Let's use a sample JSON data for demonstration
output_json = {
    "shapes": [
        {
            "label": "Example Label",
            "points": [[10, 20], [30, 40]],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
    ]
}

# Convert the JSON data to a string and then to bytes
json_string = json.dumps(output_json)
json_bytes = json_string.encode('utf-8')

# Create a BytesIO buffer with the JSON bytes
bytes_io_buffer = BytesIO(json_bytes)

# Example: Reading from the BytesIO buffer
bytes_io_buffer.seek(0)  # Move to the start of the buffer
read_data = bytes_io_buffer.read()

# Printing the read data to demonstrate it's the same as the original JSON bytes
print(read_data)