import numpy as np
from tqdm import tqdm

class OrthogonalLinesGenerator:
    def __init__(self, middle_line_points, combined_mask, separate_masks:dict=None):
        self.middle_line_points = middle_line_points
        self.combined_mask = combined_mask
        self.separate_masks = separate_masks if separate_masks is not None else {}
        self.orthogonal_lines = []
        self.orthogonal_lines_w_seg = {k: [] for k in self.separate_masks.keys()} if separate_masks is not None else {}
        self.segmented_lines = {key: [] for key in self.separate_masks}

    def get_gradient(self, p1, p2):
        """Calculate gradient (slope) of the line defined by p1 and p2."""
        return (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-8)  # Avoid division by zero

    def generate_orthogonal_lines(self, num_lines):
        """Generate orthogonal lines to the middle line."""

        if len(self.middle_line_points) < num_lines:
            raise ValueError("Number of orthogonal lines cannot be greater than the number of points in the middle line.")

        # Sample points from the middle line
        sampled_points = np.linspace(0, len(self.middle_line_points) - 1, num_lines, dtype=int)

        # Calculate pairwise distances between consecutive points in middle_line_points
        distances = [np.linalg.norm(np.array(sampled_points[i+1]) - np.array(sampled_points[i])) 
                    for i in range(len(sampled_points)-1)]
        self.h_mean = np.mean(distances)
        
        print("Generate orthogonal lines...")
        for idx in tqdm(sampled_points):
            if idx == 0:
                # Use the next point to calculate the gradient
                gradient = self.get_gradient(self.middle_line_points[idx], self.middle_line_points[idx + 1])
            elif idx == len(self.middle_line_points) - 1:
                # Use the previous point to calculate the gradient
                gradient = self.get_gradient(self.middle_line_points[idx - 1], self.middle_line_points[idx])
            else:
                # Use both previous and next points to calculate the gradient
                gradient = self.get_gradient(self.middle_line_points[idx - 1], self.middle_line_points[idx + 1])

            # Calculate the negative reciprocal for orthogonal slope
            ortho_slope = -1 / gradient
            
            # Generate and trim orthogonal line
            if self.orthogonal_lines_w_seg:
                orthogonal_line, belongs_to_seg = self.generate_trimmed_line(self.middle_line_points[idx], ortho_slope)
                self.orthogonal_lines_w_seg[belongs_to_seg].append(orthogonal_line)
            else:
                orthogonal_line, _ = self.generate_trimmed_line(self.middle_line_points[idx], ortho_slope)
                self.orthogonal_lines.append(orthogonal_line)

        print("Done.")

    def generate_trimmed_line_OLD(self, start_point, slope):
        """Generate orthogonal line and trim based on combined mask."""
        y_vals = np.arange(0, self.combined_mask.shape[0])
        x_vals = start_point[0] + (y_vals - start_point[1]) / slope

        # Filter points based on combined mask and boundary conditions
        valid_points = [(x, y) for x, y in zip(x_vals, y_vals) if 
                        0 <= x < self.combined_mask.shape[1] and 
                        self.combined_mask[int(y), int(x)] == 1]
        
        return valid_points

    def generate_trimmed_line(self, start_point, slope):
        """Generate orthogonal line, trim based on combined mask, and determine its segment."""
        y_vals = np.arange(0, self.combined_mask.shape[0])
        x_vals = start_point[0] + (y_vals - start_point[1]) / slope

        # Filter points based on combined mask and boundary conditions
        valid_points_combined_mask = [(x, y) for x, y in zip(x_vals, y_vals) if 
                                     0 <= x < self.combined_mask.shape[1] and 
                                     self.combined_mask[int(y), int(x)] == 1]

        segmented_points = {key: [] for key in self.separate_masks}

        # Filter points based on each separate mask and store in the segmented_points dictionary
        for key, mask in self.separate_masks.items():
            segmented_points[key] = [(x, y) for x, y in valid_points_combined_mask if mask[int(y), int(x)] == 1]
            
        # Determine the mask with the most points
        if segmented_points:
            dominant_mask_key = max(segmented_points, key=lambda k: len(segmented_points[k]))
            self.segmented_lines[dominant_mask_key].append(segmented_points[dominant_mask_key])
            return segmented_points[dominant_mask_key], dominant_mask_key
        else:
            return valid_points_combined_mask, None

    def get_orthogonal_lines(self):
        """Return generated orthogonal lines."""
        if self.orthogonal_lines_w_seg:
            return self.orthogonal_lines_w_seg
        return self.orthogonal_lines

    def get_h_mean(self):
        """Return the mean distance between consecutive points in the middle line."""
        return self.h_mean

    def lines_intersect(self, line1, line2):
        """Check if two line segments intersect."""
        for i in range(len(line1)-1):
            for j in range(len(line2)-1):
                A, B = line1[i], line1[i+1]
                C, D = line2[j], line2[j+1]
                
                cross1 = np.cross([B[0]-A[0], B[1]-A[1]], [C[0]-A[0], C[1]-A[1]])
                cross2 = np.cross([B[0]-A[0], B[1]-A[1]], [D[0]-A[0], D[1]-A[1]])
                cross3 = np.cross([D[0]-C[0], D[1]-C[1]], [A[0]-C[0], A[1]-C[1]])
                cross4 = np.cross([D[0]-C[0], D[1]-C[1]], [B[0]-C[0], B[1]-C[1]])
                
                if (np.sign(cross1) != np.sign(cross2)) and (np.sign(cross3) != np.sign(cross4)):
                    return True
        return False
    
    def remove_intersecting_lines(self):
        """Remove lines that intersect with their neighbors."""
        to_remove = set()
        print("Remove intersecting lines...")
        for i in tqdm(range(len(self.orthogonal_lines) - 1)):  # Only go up to the second last line
            if self.lines_intersect(self.orthogonal_lines[i], self.orthogonal_lines[i + 1]):
                # Decide which line to remove. In this case, we remove the next line.
                to_remove.add(i + 1)
        print("Done.")
        self.orthogonal_lines = [line for idx, line in enumerate(self.orthogonal_lines) if idx not in to_remove]
        print(f"Total number of deleted lines: {len(to_remove)}")

