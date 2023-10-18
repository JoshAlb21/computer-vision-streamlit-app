import pytest
import numpy as np

from tachinidae_analyzer.length_estimation.length_estimation import LengthEstimator

def test_total_length():
    points = [(0, 0), (0, 1), (1, 1), (1, 0)]
    masks = {}
    k_conv_factor = 1.0
    estimator = LengthEstimator(points, masks, k_conv_factor)
    assert estimator.calculate_total_length() == 3.0

def test_total_length_vs_body_part_lengths():
    points = [(0, 0), (0, 1), (1, 1), (1, 0)]
    masks = {
        'part1': np.array([[1, 1], [0, 0]]),
        'part2': np.array([[0, 0], [0, 1]])
    }
    k_conv_factor = 1.0
    estimator = LengthEstimator(points, masks, k_conv_factor)
    total_length = estimator.calculate_total_length()
    body_part_lengths = estimator.calculate_lengths()
    assert total_length >= sum(body_part_lengths.values())
