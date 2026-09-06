import numpy as np

from veritas.geometry import verify_affine
from veritas.verification import select_primary_certificate, transform_agreement


def _certificate(points, translation=(0.0, 0.0)):
    source = np.asarray(points, dtype=float)
    reference = source + np.asarray(translation, dtype=float)
    return verify_affine(source, reference, {"reprojection_threshold": 1.0, "min_inliers": 4})


def test_primary_selection_prefers_distributed_certified_support_over_detector_name():
    clustered = _certificate([(10, 10), (12, 10), (10, 12), (12, 12), (14, 14)])
    distributed = _certificate([(5, 5), (95, 5), (5, 95), (95, 95), (50, 50)])

    name, certificate = select_primary_certificate(
        {"sift": clustered, "orb": distributed}, (100, 100)
    )

    assert name == "orb"
    assert certificate is distributed


def test_transform_agreement_flags_incompatible_certified_maps():
    points = [(5, 5), (95, 5), (5, 95), (95, 95), (50, 50)]
    left = _certificate(points, (3, -2))
    right = _certificate(points, (30, -20))

    result = transform_agreement({"sift": left, "orb": right}, (100, 100))

    assert result["agreement_strength"] == 0.0
    assert result["pairwise"]["orb:sift"]["compatible"] is False
