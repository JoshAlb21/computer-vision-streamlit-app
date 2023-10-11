import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import cv2

class MaskPointGenerator:
    def __init__(self, masks: list, given_points: np.ndarray = None, given_weights: np.ndarray = None):
        self.masks = masks
        self.combined_mask = self.combine_masks(masks)
        self.points = self.generate_points(300)
        self.model = None
        self.poly = None  # Store the PolynomialFeatures instance
        
        # Combine given points with generated points
        if given_points is not None:
            self.points = self.combine_given_points(given_points, given_weights)

    @staticmethod
    def combine_masks(masks: list) -> np.ndarray:
        """Combine multiple masks into a single binary mask."""
        combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        return combined_mask
    
    def get_combined_mask(self):
        """Get the combined mask."""
        return self.combined_mask

    def get_points(self):
        """Get the generated points."""
        return self.points

    def generate_points(self, n_points: int) -> np.ndarray:
        """Generate n_points randomly within the combined mask."""
        y, x = np.where(self.combined_mask == 1)
        indices = np.random.choice(len(x), size=n_points, replace=False)
        return np.column_stack((x[indices], y[indices]))

    def fit_polynomial(self, degree: int = 6):
        """Fit a polynomial regression model to the generated points."""
        x = self.points[:, 0].reshape(-1, 1)
        y = self.points[:, 1]
        
        self.poly = PolynomialFeatures(degree)  # Store the PolynomialFeatures instance
        x_poly = self.poly.fit_transform(x)
        
        self.model = LinearRegression().fit(x_poly, y)
        return self.model

    def predict(self, x):
        """Predict y values using the fitted polynomial model."""
        if self.model is None or self.poly is None:  # Check that poly is not None
            raise ValueError("Model has not been fitted yet. Call fit_polynomial first.")
        x_poly = self.poly.transform(x.reshape(-1, 1))  # Use the stored poly instance
        return self.model.predict(x_poly)

    def get_fitted_points(self):
        """Get y values for all x coordinates in the combined mask."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit_polynomial first.")
        
        # Get all unique x coordinates in the non-zero regions of the combined mask
        y, x = np.where(self.combined_mask == 1)
        unique_x = np.unique(x)
        
        # Predict y values for each unique x coordinate
        predicted_y = self.predict(unique_x)
        
        # Pair each x coordinate with its corresponding predicted y value
        fitted_points = np.column_stack((unique_x, predicted_y))
        return fitted_points

    def combine_given_points(self, given_points: np.ndarray, given_weights: np.ndarray = None) -> np.ndarray:
        """
        Combine the given points with the randomly generated points.
        If weights are provided, replicate the given points accordingly.
        """
        if given_weights is not None:
            # Validate that given_points and given_weights have compatible shapes
            if len(given_points) != len(given_weights):
                raise ValueError("Given points and weights must have the same length")
            
            # Replicate given points according to their weights
            weighted_points = np.repeat(given_points, given_weights.astype(int), axis=0)
        else:
            weighted_points = given_points
        
        # Combine generated points and weighted given points
        combined_points = np.vstack((self.points, weighted_points))
        return combined_points

if __name__ == '__main__':
    # Usage example:
    masks = [np.random.randint(0, 2, (100, 100)) for _ in range(3)]  # Replace this with your actual masks
    generator = MaskPointGenerator(masks)
    generator.fit_polynomial(degree=6)

    # Get (x, y) pairs for all x coordinates in the combined mask
    fitted_points = generator.get_fitted_points()