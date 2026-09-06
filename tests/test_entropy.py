import numpy as np
from veritas.spatial import calculate_spatial_entropy


def test_entropy_separates_uniform_and_concentrated_points():
    spread = np.array([[10, 10], [90, 10], [10, 90], [90, 90]], float)
    concentrated = np.repeat([[10.0, 10.0]], 20, axis=0)
    assert calculate_spatial_entropy(spread, (100, 100))["normalized_entropy"] > 0.45
    assert calculate_spatial_entropy(concentrated, (100, 100))["normalized_entropy"] == 0.0
