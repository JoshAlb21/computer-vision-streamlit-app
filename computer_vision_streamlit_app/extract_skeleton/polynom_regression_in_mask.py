import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import cv2
from scipy.odr import ODR, Model, Data
from scipy.interpolate import interp1d, CubicSpline
import scipy.interpolate


class MaskPointGenerator:
    def __init__(self, masks: list, given_points: np.ndarray = None, given_weights: np.ndarray = None, num_points:int= 500):
        self.masks = masks
        self.combined_mask = self.combine_masks(masks)
        self.points = self.generate_points(num_points)
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

        # Keep only the points that lie inside the combined mask
        inside_mask = [self.combined_mask[int(round(y))][int(round(x))] == 1 for x, y in fitted_points]
        fitted_points = fitted_points[inside_mask]

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

    @staticmethod    
    def poly_func(beta, x):
        '''Define the polynomial function of degree n'''
        # Compute polynomial based on beta's length
        y = 0
        for i, coef in enumerate(beta):
            y += coef * (x ** i)
        return y

    def fit_get_odr(self, degree=6) -> np.ndarray:
        '''
        Fit a polynomial regression model to the generated points via ODR.
        '''

        x = self.points[:, 0].reshape(-1, 1)
        y = self.points[:, 1]
        x = np.squeeze(x)
        y = np.squeeze(y)

        assert len(x) == len(y), "x and y must have the same length, but got {} and {}".format(len(x), len(y))
        model = Model(self.poly_func)
        data = Data(x, y)
        beta_vec = np.ones(degree + 1)
        odr = ODR(data, model, beta0=beta_vec)
        output = odr.run()

        # Print the estimated parameters
        #print('Estimated parameters:', output.beta)
        unique_x = np.unique(x)
        y_fit = self.poly_func(output.beta, unique_x)

        # Pair each x coordinate with its corresponding predicted y value
        fitted_points = np.column_stack((unique_x, y_fit))

        # Count the number of NaN values in fitted_points
        nan_count = np.sum(np.isnan(fitted_points))
        beta_nan_count = np.sum(np.isnan(output.beta))
        print(f"Number of NaN values in fitted_points: {nan_count}")
        print(f"Number of NaN values in beta: {beta_nan_count}")

        # Keep only the points that lie inside the combined mask
        inside_mask = np.array([
            0 <= int(round(x)) < self.combined_mask.shape[1] 
            and 0 <= int(round(y)) < self.combined_mask.shape[0] 
            and self.combined_mask[int(round(y))][int(round(x))] == 1 
            for x, y in fitted_points
        ])
        fitted_points = fitted_points[inside_mask]

        return fitted_points

    def interpolate_points(self, given_points=None, kind='cubic'):
        """
        Interpolate the provided points (or self-given points if not provided).
        
        Args:
            given_points (np.ndarray, optional): Points to interpolate. If None, uses self-given points.
            kind (str): Specifies the kind of interpolation. Can be 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'.
                        Default is 'cubic'.

        Returns:
            np.ndarray: Interpolated points.
        """

        # Use the provided points or the self-given points
        if given_points is None:
            if self.points is not None:
                given_points = self.points
            else:
                raise ValueError("No points provided for interpolation and no given points in the class instance.")

        # Prevent: ValueError: Expect x to not have duplicates
        # Sort the points based on x values
        sorted_indices = np.argsort(given_points[:, 0])
        sorted_points = given_points[sorted_indices]

        # For each unique x value, average the y values
        unique_x, unique_indices = np.unique(sorted_points[:, 0], return_index=True)
        y_values = [np.mean(sorted_points[sorted_points[:, 0] == x_val, 1]) for x_val in unique_x]

        # Create the interpolation function
        f = interp1d(unique_x, y_values, kind=kind, fill_value="extrapolate")

        # Interpolate for the x values
        x_new = np.linspace(unique_x.min(), unique_x.max(), len(unique_x) * 10)  # Create finer x values for interpolation
        y_new = f(x_new)

        return np.column_stack((x_new, y_new))

    def interpolate_points_spline(self, given_points=None):
        """
        Perform spline interpolation on the provided points (or self-given points if not provided).

        Args:
            given_points (np.ndarray, optional): Points to interpolate. If None, uses self-given points.

        Returns:
            np.ndarray: Interpolated points using spline interpolation.
        """

        # Use the provided points or the self-given points
        if given_points is None:
            if self.points is not None:
                given_points = self.points
            else:
                raise ValueError("No points provided for interpolation and no given points in the class instance.")

        # Prevent: ValueError: Expect x to not have duplicates
        # Sort the points based on x values
        sorted_indices = np.argsort(given_points[:, 0])
        sorted_points = given_points[sorted_indices]

        # For each unique x value, average the y values
        unique_x, unique_indices = np.unique(sorted_points[:, 0], return_index=True)
        y_values = [np.mean(sorted_points[sorted_points[:, 0] == x_val, 1]) for x_val in unique_x]

        # Create the spline interpolation function
        cs = CubicSpline(unique_x, y_values)

        # Interpolate for the x values
        x_new = np.linspace(unique_x.min(), unique_x.max(), len(unique_x) * 10)  # Create finer x values for interpolation
        y_new = cs(x_new)

        return np.column_stack((x_new, y_new))

    def interpolate_points_parametric_spline(self, given_points=None):
        """
        Perform parametric spline interpolation on the provided points 
        (or self-given points if not provided).

        Args:
            given_points (np.ndarray, optional): Points to interpolate. If None, uses self-given points.

        Returns:
            np.ndarray: Interpolated points using parametric spline interpolation.
        """

        # Use the provided points or the self-given points
        if given_points is None:
            if self.points is not None:
                given_points = self.points
            else:
                raise ValueError("No points provided for interpolation and no given points in the class instance.")

        # Extract x and y values
        path_x = given_points[:, 0]
        path_y = given_points[:, 1]

        # Define an arbitrary parameter to parameterize the curve
        path_t = np.linspace(0, 1, path_x.size)

        # Create the parametric spline interpolation objects for x and y separately
        spline_x = scipy.interpolate.CubicSpline(path_t, path_x, bc_type='natural')
        spline_y = scipy.interpolate.CubicSpline(path_t, path_y, bc_type='natural')

        # Define values of the arbitrary parameter over which to interpolate
        t = np.linspace(np.min(path_t), np.max(path_t), 100)

        # Interpolate along t
        x_interp = spline_x(t)
        y_interp = spline_y(t)

        #return np.column_stack((x_interp, y_interp))

        # Compute the tangent at t=0 and t=1
        tangent_start = np.array([spline_x.derivative()(path_t[0]), spline_y.derivative()(path_t[0])])
        tangent_end = np.array([spline_x.derivative()(path_t[-1]), spline_y.derivative()(path_t[-1])])

        # Clip the extrapolated points to the boundaries
        start_extension = self._extend_line_from_point(given_points[0], -tangent_start)  # Note the negative sign
        end_extension = self._extend_line_from_point(given_points[-1], tangent_end)

        # Concatenate the extension points to the interpolated curve
        extended_curve = np.vstack(([start_extension], np.column_stack((x_interp, y_interp)), [end_extension]))

        return extended_curve

    def _extend_line_from_point(self, point, direction, x_min=0, y_min=0, x_max=4032, y_max=3040):
        """Extend a line from a given point in a specified direction until it reaches an image boundary."""
        m = direction[1] / direction[0] if direction[0] != 0 else float('inf')
        
        if m == float('inf'):  # vertical line
            return (point[0], y_max if direction[1] > 0 else y_min)
        
        # Find the intersection points with each boundary
        y_at_x_min = m * (x_min - point[0]) + point[1]
        y_at_x_max = m * (x_max - point[0]) + point[1]
        x_at_y_min = point[0] + (y_min - point[1]) / m
        x_at_y_max = point[0] + (y_max - point[1]) / m

        # Check which boundary the line intersects first
        if y_at_x_min >= y_min and y_at_x_min <= y_max and direction[0] < 0:
            return (x_min, y_at_x_min)
        elif y_at_x_max >= y_min and y_at_x_max <= y_max and direction[0] > 0:
            return (x_max, y_at_x_max)
        elif x_at_y_min >= x_min and x_at_y_min <= x_max and direction[1] < 0:
            return (x_at_y_min, y_min)
        else:
            return (x_at_y_max, y_max)

    def parametric_polynomial_regression(self, degree=2, extend_range=0.1):
        """
        Perform a parametric polynomial regression of given degree on the points.

        Args:
            degree (int): Degree of the polynomial regression.
            extend_range (float): Percentage to extend the t-values beyond [0, 1].

        Returns:
            np.ndarray: Fitted points as a result of the parametric regression.
        """

        # If no points are present, raise an error
        if self.points is None:
            raise ValueError("No points available for regression.")

        # Split the points into X and Y components
        t_values = np.linspace(0, 1, len(self.points))
        x = self.points[:, 0]
        y = self.points[:, 1]

        # Fit a polynomial regression to the X and Y components
        poly = PolynomialFeatures(degree)
        t_poly = poly.fit_transform(t_values[:, np.newaxis])
        
        model_x = LinearRegression().fit(t_poly, x)
        model_y = LinearRegression().fit(t_poly, y)

        # Generate fitted points
        t_start = 0 - extend_range
        t_end = 1 + extend_range
        t_new = np.linspace(t_start, t_end, 1000)
        t_poly_new = poly.transform(t_new[:, np.newaxis])
        
        x_fit = model_x.predict(t_poly_new)
        y_fit = model_y.predict(t_poly_new)

        return np.column_stack((x_fit, y_fit))

    def parametric_regression(self, x, y, degree=2):
        """
        Perform a parametric regression on the given 2D data.
        
        Parameters:
        - x, y: 1D arrays of data points.
        - degree: degree of the polynomial for the regression.
        
        Returns:
        - x_pred, y_pred: Predicted values after the regression.
        """
        # The independent parameter t
        t = np.linspace(0, 1, len(x))
        
        # Create polynomial features for t
        polynomial_features = PolynomialFeatures(degree=degree)
        T = polynomial_features.fit_transform(t.reshape(-1, 1))
        
        # Perform linear regressions
        reg_x = LinearRegression().fit(T, x)
        reg_y = LinearRegression().fit(T, y)
        
        # Predict using the trained regressions
        t_pred = np.linspace(0, 1, 1000)
        T_pred = polynomial_features.transform(t_pred.reshape(-1, 1))
        x_pred = reg_x.predict(T_pred)
        y_pred = reg_y.predict(T_pred)
        
        return np.column_stack((x_pred, y_pred))

if __name__ == '__main__':
    # Usage example:
    masks = [np.random.randint(0, 2, (100, 100)) for _ in range(3)]  # Replace this with your actual masks
    generator = MaskPointGenerator(masks)
    generator.fit_polynomial(degree=6)

    # Get (x, y) pairs for all x coordinates in the combined mask
    fitted_points = generator.get_fitted_points()