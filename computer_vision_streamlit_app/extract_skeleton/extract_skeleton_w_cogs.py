import computer_vision_streamlit_app as ta
import numpy as np
from typing import Optional

def method2_connect_cog(ordered_cogs, random_points, combined_mask, alpha=1, iterations=500, n_samples=100) -> Optional[np.ndarray]:
    '''
    1. connect cogs linearly
    2. extend lines on both sides
    3. find intersection points with mask and cut off line
    4. calculate density map of random points, calculate gradient
    5. "push"/"pull" trimmed extension lines along gradient

    Returns
    -------
    fitted_points : None if no points could be fitted (e.g. only one cog present)
        if points could be fitted, return the fitted points
    '''
    try:
        orderer = ta.extract_skeleton.point_orderer.PointOrderer(ordered_cogs)
    except ValueError:
        return None
    reference_points_extended = np.array(orderer.extended_reference_points)
    refiner = ta.extract_skeleton.line_refiner.GraphRefiner(reference_points_extended, random_points, alpha=alpha, iterations=iterations)
    optimized_points = refiner.get_refined_points()
    control_points = refiner.trim_by_mask(combined_mask)
    fitted_points = ta.extract_skeleton.line_refiner.sample_points_from_segments(control_points, n=n_samples)

    return fitted_points