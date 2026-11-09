"""Generate individual BEFORE/AFTER result cards for each pair with metrics."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "human_audit" / "pair_results"
AUDIT = ROOT / "outputs" / "real_world_audit"

# Canvas layout constants.
CANVAS_IMG_W: int = 600
CANVAS_IMG_H: int = 400
CANVAS_PAD_X: int = 60
CANVAS_HEADER_H: int = 60
CANVAS_EXTRA_H: int = 500
METRICS_LINE_H: int = 28


def _load_pair_images(item: dict) -> tuple | None:
    """Resolve and load the before/after images for one audit item.

    Returns ``None`` when either image is missing on disk.
    """
    pair_id = item["pair_id"]
    idx = pair_id.split("_")[1]

    before_path = ROOT / "data" / "raw" / f"ds_{int(idx)}.png"
    if not before_path.exists():
        before_path = ROOT / "data" / "raw" / f"ds_{int(idx)}.jpg"
    after_path = ROOT / "data" / "processed" / f"ds_{int(idx)}.jpg"
    if not after_path.exists():
        after_path = ROOT / "data" / "processed" / f"ds_{int(idx)}.png"

    if not before_path.exists() or not after_path.exists():
        return None

    before = cv2.imread(str(before_path))
    after = cv2.imread(str(after_path))
    return before, after, pair_id


def _build_canvas(before, after, pair_id: str, idx: str) -> np.ndarray:
    """Create the result card canvas with header and placed images."""
    canvas_w = CANVAS_IMG_W * 2 + CANVAS_PAD_X
    canvas_h = CANVAS_IMG_H + CANVAS_EXTRA_H
    canvas = np.full((canvas_h, canvas_w, 3), 245, np.uint8)

    cv2.rectangle(canvas, (0, 0), (canvas_w, CANVAS_HEADER_H), (50, 50, 100), -1)
    cv2.putText(
        canvas,
        f"PAIR {idx} - {pair_id.upper()}",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    before_img = cv2.resize(before, (CANVAS_IMG_W, CANVAS_IMG_H))
    after_img = cv2.resize(after, (CANVAS_IMG_W, CANVAS_IMG_H))
    canvas[70 : 70 + CANVAS_IMG_H, 20 : 20 + CANVAS_IMG_W] = before_img
    canvas[70 : 70 + CANVAS_IMG_H, 40 + CANVAS_IMG_W : 40 + 2 * CANVAS_IMG_W] = after_img

    cv2.putText(
        canvas,
        "BEFORE (RAW)",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "AFTER (PROCESSED)",
        (40 + CANVAS_IMG_W, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (30, 30, 170),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _draw_detector_metrics(canvas: np.ndarray, y: int, item: dict) -> int:
    """Draw SIFT, ORB and AKAZE feature-matching metrics; returns the next y."""
    canvas_w = CANVAS_IMG_W * 2 + CANVAS_PAD_X
    cv2.rectangle(canvas, (20, y), (canvas_w - 20, y + 3), (200, 200, 200), -1)
    y += 30

    cv2.rectangle(canvas, (20, y), (canvas_w - 20, y + 35), (220, 220, 240), -1)
    cv2.putText(
        canvas,
        "FEATURE MATCHING METRICS",
        (30, y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 140),
        2,
        cv2.LINE_AA,
    )
    y += 50

    detectors = item["detectors"]
    inlier_ratio = item["primary_geometry"]["inlier_ratio"]

    sift = detectors["sift"]
    line1 = (
        f"SIFT:  {sift['source_keypoints']} kp | {sift['filtered_matches']} matches | "
        f"{sift['inliers']} inliers | ratio: {inlier_ratio:.4f}"
    )
    cv2.putText(canvas, line1, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    y += METRICS_LINE_H

    orb = detectors["orb"]
    line2 = (
        f"ORB:   {orb['source_keypoints']} kp | {orb['filtered_matches']} matches | "
        f"{orb['inliers']} inliers"
    )
    cv2.putText(canvas, line2, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    y += METRICS_LINE_H

    akaze = detectors["akaze"]
    line3 = (
        f"AKAZE: {akaze['source_keypoints']} kp | {akaze['filtered_matches']} matches | "
        f"{akaze['inliers']} inliers"
    )
    cv2.putText(canvas, line3, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    y += METRICS_LINE_H + 7
    return y


def add_geometry_section(canvas: np.ndarray, y: int, item: dict) -> int:
    """Add geometry and residuals section; returns the next y."""
    canvas_w = CANVAS_IMG_W * 2 + CANVAS_PAD_X
    cv2.rectangle(canvas, (20, y), (canvas_w - 20, y + 3), (200, 200, 200), -1)
    y += 25
    cv2.rectangle(canvas, (20, y), (canvas_w - 20, y + 35), (220, 220, 240), -1)
    cv2.putText(
        canvas,
        "GEOMETRY & RESIDUALS",
        (30, y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 140),
        2,
        cv2.LINE_AA,
    )
    y += 50

    geo = item["primary_geometry"]
    res = geo["residuals"]
    cert = "CERTIFIED" if geo["certified"] else "NOT CERTIFIED"
    cert_color = (0, 120, 0) if geo["certified"] else (180, 0, 0)

    cv2.putText(
        canvas,
        f"Affine Status: {cert}",
        (30, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        cert_color,
        2,
        cv2.LINE_AA,
    )
    y += 28
    line4 = f"RMSE: {res['rmse']:.4f} px | P95: {res['p95']:.4f} px | Max: {res['maximum']:.4f} px"
    cv2.putText(canvas, line4, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    y += 35
    return y


def add_spatial_section(canvas: np.ndarray, y: int, item: dict) -> int:
    """Add spatial distribution section; returns the next y."""
    canvas_w = CANVAS_IMG_W * 2 + CANVAS_PAD_X
    cv2.rectangle(canvas, (20, y), (canvas_w - 20, y + 3), (200, 200, 200), -1)
    y += 25
    cv2.rectangle(canvas, (20, y), (canvas_w - 20, y + 35), (220, 220, 240), -1)
    cv2.putText(
        canvas,
        "SPATIAL DISTRIBUTION",
        (30, y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 140),
        2,
        cv2.LINE_AA,
    )
    y += 50

    spat = item["spatial"]
    line5 = (
        f"Coverage: {spat['coverage_ratio']:.4f} | Entropy: {spat['normalized_entropy']:.4f} | "
        f"Cells: {spat['occupied_cells']}/{spat['total_cells']}"
    )
    cv2.putText(canvas, line5, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    y += 28
    return y


def add_quorum_section(canvas: np.ndarray, y: int, item: dict) -> int:
    """Add quorum and timing section; returns the next y."""
    canvas_w = CANVAS_IMG_W * 2 + CANVAS_PAD_X
    q = item["quorum"]
    line6 = (
        f"Agreement: SIFT={q['detector_results']['sift']} | "
        f"ORB={q['detector_results']['orb']} | "
        f"AKAZE={q['detector_results']['akaze']}"
    )
    cv2.putText(canvas, line6, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    y += 28
    cv2.putText(
        canvas,
        f"Quorum: {q['quorum_state']}",
        (30, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    y += 35

    cv2.rectangle(canvas, (20, y), (canvas_w - 20, y + 3), (200, 200, 200), -1)
    y += 25
    cv2.putText(
        canvas,
        f"Time: {item['timing']['total_seconds']:.3f}s",
        (30, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    return y


def main() -> int:
    """Generate one result card PNG per audit pair."""
    data = json.loads((AUDIT / "results.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    count = 0
    for item in data:
        loaded = _load_pair_images(item)
        if loaded is None:
            continue
        before, after, pair_id = loaded
        idx = pair_id.split("_")[1]

        canvas = _build_canvas(before, after, pair_id, idx)
        y = 70 + CANVAS_IMG_H + 30
        y = _draw_detector_metrics(canvas, y, item)
        y = add_geometry_section(canvas, y, item)
        y = add_spatial_section(canvas, y, item)
        y = add_quorum_section(canvas, y, item)

        output_path = OUT / f"{pair_id}_results.png"
        cv2.imwrite(str(output_path), canvas)
        count += 1

    return count


if __name__ == "__main__":
    generated = main()
    print(f"Done! Generated {generated} result cards.")
