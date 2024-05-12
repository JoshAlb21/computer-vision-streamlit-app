"""
VolumeEstimator Class Documentation:

Purpose:
The VolumeEstimator class is designed to estimate the volume of a structure based on orthogonal lines to a central skeleton. By understanding the cross-sectional area at various points along the length (represented by the orthogonal lines), the class approximates the volume of the entire structure.

How It Works:

1. Initialization:
   - The class is initialized with a dictionary of orthogonal lines and a distance value `h`.
   - The dictionary's keys represent the names of body parts (or segments), and the values are lists of orthogonal lines for that segment.
   - The distance `h` represents the distance between two consecutive orthogonal lines.

2. Calculate Cross-sectional Areas:
   - For each orthogonal line, the class calculates the area it represents by assuming the line's endpoints represent the diameter of a circular cross-section.
   - Area = π * (width/2)^2

3. Estimate Volume Between Two Lines:
   - The class computes the volume between two adjacent orthogonal lines by assuming the shape between them is a frustum of a cone.
   - Volume = (h/3) * (Area1 + Area2 + sqrt(Area1 * Area2))

4. Calculate Total Volume:
   - The class computes the total volume by summing up the volumes of all the sections defined by the orthogonal lines.

Usage:

orthogonal_lines = {
    "body_part_1": [line1, line2, ...],   # Actual lines for each body segment
    ...
}
h_value = ...  # Actual distance between the lines

estimator = VolumeEstimator(orthogonal_lines, h_value)
estimated_volume = estimator.calculate_volume()
"""

import numpy as np
from typing import Optional

class VolumeEstimator:
    def __init__(self, orthogonal_lines_dict: dict, h: float, k_conv_factor: Optional[float]):
        """
        Initialize the VolumeEstimator.

        Note: All lengths must be in pixel! The conversion factor k_conv_factor is used to convert from pixels to mm.

        Parameters:
            orthogonal_lines_dict: Dictionary with keys being the name of the object_part and values being lists of lines.
            h: Distance between two adjacent orthogonal lines.
            k_conv_factor: Conversion factor to convert from pixels to mm. [mm/pixel]
        """
        self.orthogonal_lines_dict = orthogonal_lines_dict
        self.h = h
        self.k_conv_factor = k_conv_factor if k_conv_factor is not None else 1

    def _calculate_area(self, line:list):
        """
        Calculate the area represented by the orthogonal line.
        Assuming the cross-sectional shape is roughly elliptical.
        """
        assert len(line) >= 2, f"Line must have at least two points, but got {len(line)} point(s)."
        width = np.linalg.norm(np.array(line[0]) - np.array(line[-1]))  # Distance between the two endpoints of the line
        return np.pi * (width / 2) ** 2

    def _calculate_frustum_volume(self, A1, A2):
        """
        Calculate the volume of a frustum given two cross-sectional areas A1 and A2.
        """
        return (self.h / 3) * (A1 + A2 + np.sqrt(A1 * A2))

    def calculate_volume_in_mm_3(self, round_to) -> tuple[float, dict]:
        """
        Calculate the estimated volume of the structure for each object_part separately and the total volume.
        Units: mm^3
        """
        total_volume = 0
        body_part_volumes = {f'volume_{bp}': 0 for bp in self.orthogonal_lines_dict.keys()}

        for object_part, lines in self.orthogonal_lines_dict.items():
            for i in range(1, len(lines)):
                elements_per_line_i_1 = [len(line) for line in lines[i - 1]]
                elements_per_line_i = [len(line) for line in lines[i]]

                if len(elements_per_line_i) < 2 or len(elements_per_line_i_1) < 2:
                    print(f'Both lines must contain at least two points. Got {len(elements_per_line_i_1)} and {len(elements_per_line_i)} points for segment: {object_part}')
                    print("Skipping...")
                    continue

                A1 = self._calculate_area(lines[i - 1])
                A2 = self._calculate_area(lines[i])

                slice_volume = self.k_conv_factor**3 * self._calculate_frustum_volume(A1, A2)
                
                body_part_volumes[f"volume_{object_part}"] += slice_volume

            body_part_volumes[f"volume_{object_part}"] = round(body_part_volumes[f"volume_{object_part}"], round_to)
            total_volume += body_part_volumes[f"volume_{object_part}"]

        total_volume = round(total_volume, round_to)
        return total_volume, body_part_volumes
