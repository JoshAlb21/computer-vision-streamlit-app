import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


class GraphRefiner:
    
    def __init__(self, control_points, scatter_points, alpha=0.1, iterations=100):
        self.control_points = control_points
        self.scatter_points = scatter_points
        self.density_map = self.compute_density_map()
        self.refined_points = self.refine_endpoints(iterations=iterations, alpha=alpha)
    
    def compute_density_map(self, grid_size=100):
        """Compute a 2D density map using gaussian KDE."""
        kde = scipy.stats.gaussian_kde(np.array(self.scatter_points).T)
        x_grid = np.linspace(min(np.array(self.scatter_points)[:,0]), max(np.array(self.scatter_points)[:,0]), grid_size)
        y_grid = np.linspace(min(np.array(self.scatter_points)[:,1]), max(np.array(self.scatter_points)[:,1]), grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = kde.evaluate(np.vstack([X.ravel(), Y.ravel()]))
        return X, Y, Z.reshape(X.shape)
    
    def pull_point_by_density(self, point, alpha=0.1):
        """Pull a point based on the density map."""
        X, Y, Z = self.density_map
        gradient_x = np.gradient(Z, axis=1)
        gradient_y = np.gradient(Z, axis=0)
        
        # Interpolate gradient at the given point
        grad_x_interp = np.interp(point[0], X[0, :], gradient_x[int(len(X) / 2)])
        grad_y_interp = np.interp(point[1], Y[:, 0], gradient_y[:, int(len(Y) / 2)])
        
        # Compute the direction to move
        direction = np.array([grad_x_interp, grad_y_interp])
        direction /= (np.linalg.norm(direction) + 1e-9)  # Normalize
        
        # Move the point
        new_point = np.array(point) + alpha * direction
        return new_point
    
    def refine_endpoints(self, iterations=100, alpha=2):
        """Refine the endpoints of the graph based on the density map."""
        refined_points = self.control_points.copy()
        for _ in range(iterations):
            for i in [0, -1]:
                refined_points[i] = tuple(self.pull_point_by_density(refined_points[i], alpha))
        return refined_points
    
    def get_refined_points(self):
        """Return the refined points after optimization."""
        return self.refined_points

    def trim_by_mask(self, binary_mask):
        """Modify the start and end points of the line to lie within the boundaries defined by the binary mask."""
        refined_points = self.get_refined_points()
        
        refined_points = trim_line(binary_mask, refined_points)

        self.refined_points = refined_points

        return refined_points


def sample_points_from_segments(middle_line_points, n) -> np.ndarray:
    '''
    Samples n points from the middle line segments.
    Other functions require the middle line, to be desribed by a list of points.
    '''

    # Calculate the total length of the piecewise linear connection
    total_length = 0
    for i in range(len(middle_line_points) - 1):
        total_length += np.linalg.norm(np.array(middle_line_points[i+1]) - np.array(middle_line_points[i]))

    # Distance between each sampled point
    distance_between_samples = total_length / (n-1)  # n-1 intervals for n points

    sampled_points = [middle_line_points[0]]  # starting with the first point
    remaining_distance = distance_between_samples

    for i in range(len(middle_line_points) - 1):
        p1 = np.array(middle_line_points[i])
        p2 = np.array(middle_line_points[i+1])
        segment_length = np.linalg.norm(p2 - p1)

        while segment_length >= remaining_distance:
            # Calculate the next sampled point on the current segment
            t = remaining_distance / segment_length
            next_point = (1 - t) * p1 + t * p2
            sampled_points.append(tuple(next_point))
            
            # Move to the next sampling position
            segment_length -= remaining_distance
            remaining_distance = distance_between_samples
            p1 = next_point

        # If we haven't reached the end of the segment, set the remaining distance for the next segment
        if segment_length > 0:
            remaining_distance -= segment_length

    # Ensure the last point is included
    if len(sampled_points) < n:
        sampled_points.append(middle_line_points[-1])

    sampled_points = np.array(sampled_points)

    return sampled_points

def trim_line(binary_mask: np.ndarray, refined_points: list):
    # Create a grid of x and y coordinates for mask
    x_grid, y_grid = np.meshgrid(np.arange(binary_mask.shape[1]), np.arange(binary_mask.shape[0]))
    
    # Function to check if a point lies within the mask boundaries
    def point_in_mask(x, y):
        if 0 <= int(y) < binary_mask.shape[0] and 0 <= int(x) < binary_mask.shape[1]:
            return binary_mask[int(y), int(x)]
        return False
    
    # Trim start segment
    for i in range(len(refined_points)-1):
        x_values = np.linspace(refined_points[i][0], refined_points[i+1][0], 100)
        y_values = np.linspace(refined_points[i][1], refined_points[i+1][1], 100)
        for x, y in zip(x_values, y_values):
            if point_in_mask(x, y):
                refined_points[i] = (x, y)
                break
    
    # Trim end segment
    for i in range(len(refined_points)-1, 0, -1):
        x_values = np.linspace(refined_points[i][0], refined_points[i-1][0], 100)
        y_values = np.linspace(refined_points[i][1], refined_points[i-1][1], 100)
        for x, y in zip(x_values, y_values):
            if point_in_mask(x, y):
                refined_points[i] = (x, y)
                break
    
    return refined_points