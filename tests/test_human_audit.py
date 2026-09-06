import csv

import cv2
import numpy as np

from scripts import generate_human_audit as audit


def test_fit_preserves_aspect_ratio():
    image = np.zeros((100, 400, 3), dtype=np.uint8)
    resized = audit.fit(image, 200, 200)
    assert resized.shape[:2] == (50, 200)


def test_sheet_uses_images_without_creating_a_label(tmp_path):
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    cv2.imwrite(str(before), np.full((40, 80, 3), 100, dtype=np.uint8))
    cv2.imwrite(str(after), np.full((80, 40, 3), 150, dtype=np.uint8))
    destination = tmp_path / "case.png"
    assert audit.sheet({"id": "test_001", "before": before, "after": after, "output": None}, {}, destination)
    assert destination.is_file()
