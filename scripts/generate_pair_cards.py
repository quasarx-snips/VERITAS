"""Generate individual BEFORE/AFTER result cards for each pair with metrics."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "human_audit" / "pair_results"
AUDIT = ROOT / "outputs" / "real_world_audit"

def main():
    data = json.loads((AUDIT / "results.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    
    for item in data:
        pair_id = item["pair_id"]
        idx = pair_id.split("_")[1]
        
        # Load images
        before_path = ROOT / "data" / "raw" / f"ds_{int(idx)}.png"
        if not before_path.exists():
            before_path = ROOT / "data" / "raw" / f"ds_{int(idx)}.jpg"
        after_path = ROOT / "data" / "processed" / f"ds_{int(idx)}.jpg"
        if not after_path.exists():
            after_path = ROOT / "data" / "processed" / f"ds_{int(idx)}.png"
        
        if not before_path.exists() or not after_path.exists():
            continue
        
        before = cv2.imread(str(before_path))
        after = cv2.imread(str(after_path))
        
        # Canvas dimensions
        img_w, img_h = 600, 400
        canvas_w = img_w * 2 + 60
        canvas_h = img_h + 500
        canvas = np.full((canvas_h, canvas_w, 3), 245, np.uint8)
        
        # Header
        cv2.rectangle(canvas, (0, 0), (canvas_w, 60), (50, 50, 100), -1)
        cv2.putText(canvas, f"PAIR {idx} - {pair_id.upper()}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Resize and place images
        before_img = cv2.resize(before, (img_w, img_h))
        after_img = cv2.resize(after, (img_w, img_h))
        canvas[70:70+img_h, 20:20+img_w] = before_img
        canvas[70:70+img_h, 40+img_w:40+2*img_w] = after_img
        
        # Image labels
        cv2.putText(canvas, "BEFORE (RAW)", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(canvas, "AFTER (PROCESSED)", (40+img_w, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 170), 2, cv2.LINE_AA)
        
        # Metrics section
        y = 70 + img_h + 30
        cv2.rectangle(canvas, (20, y), (canvas_w-20, y+3), (200, 200, 200), -1)
        y += 30
        
        # Section title
        cv2.rectangle(canvas, (20, y), (canvas_w-20, y+35), (220, 220, 240), -1)
        cv2.putText(canvas, "FEATURE MATCHING METRICS", (30, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 140), 2, cv2.LINE_AA)
        y += 50
        
        # SIFT metrics
        sift = item["detectors"]["sift"]
        sift_ratio = item["primary_geometry"]["inlier_ratio"]
        line1 = f"SIFT:  {sift['source_keypoints']} kp | {sift['filtered_matches']} matches | {sift['inliers']} inliers | ratio: {sift_ratio:.4f}"
        cv2.putText(canvas, line1, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        y += 28
        
        # ORB metrics
        return canvas, y, img_h, item, pair_id, OUT, print


def add_geometry_section(canvas, y, item):
    """Add geometry and residuals section."""
    # Geometry section
    cv2.rectangle(canvas, (20, y), (canvas_w-20, y+3), (200, 200, 200), -1)
    y += 25
    cv2.rectangle(canvas, (20, y), (canvas_w-20, y+35), (220, 220, 240), -1)
    cv2.putText(canvas, "GEOMETRY & RESIDUALS", (30, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 140), 2, cv2.LINE_AA)
    y += 50
    
    geo = item["primary_geometry"]
    res = geo["residuals"]
    cert = "CERTIFIED" if geo["certified"] else "NOT CERTIFIED"
    cert_color = (0, 120, 0) if geo["certified"] else (180, 0, 0)
    
    cv2.putText(canvas, f"Affine Status: {cert}", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cert_color, 2, cv2.LINE_AA)
    y += 28
    line4 = f"RMSE: {res['rmse']:.4f} px | P95: {res['p95']:.4f} px | Max: {res['maximum']:.4f} px"
    cv2.putText(canvas, line4, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    y += 35
    return y, canvas, item, OUT, print


def add_spatial_section(canvas, y, item):
    """Add spatial distribution section."""
    # Spatial section
    cv2.rectangle(canvas, (20, y), (canvas_w-20, y+3), (200, 200, 200), -1)
    y += 25
    cv2.rectangle(canvas, (20, y), (canvas_w-20, y+35), (220, 220, 240), -1)
    cv2.putText(canvas, "SPATIAL DISTRIBUTION", (30, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 140), 2, cv2.LINE_AA)
    y += 50
    
    spat = item["spatial"]
    line5 = f"Coverage: {spat['coverage_ratio']:.4f} | Entropy: {spat['normalized_entropy']:.4f} | Cells: {spat['occupied_cells']}/{spat['total_cells']}"
    cv2.putText(canvas, line5, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    y += 28
    return y, canvas, item, OUT, print


def add_quorum_section(canvas, y, item):
    """Add quorum and timing section."""
    # Quorum
    q = item["quorum"]
    line6 = f"Agreement: SIFT={q['detector_results']['sift']} | ORB={q['detector_results']['orb']} | AKAZE={q['detector_results']['akaze']}"
    cv2.putText(canvas, line6, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    y += 28
    cv2.putText(canvas, f"Quorum: {q['quorum_state']}", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    y += 35
    
    # Timing
    cv2.rectangle(canvas, (20, y), (canvas_w-20, y+3), (200, 200, 200), -1)
    y += 25
    cv2.putText(canvas, f"Time: {item['timing']['total_seconds']:.3f}s", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
    return canvas, y, item, OUT, print


if __name__ == "__main__":
    # Run the main function
    result = main()
    if result:
        canvas, y, img_h, item, pair_id, OUT, print = result
        y, canvas, item, OUT, print = add_geometry_section(canvas, y, item)
        y, canvas, item, OUT, print = add_spatial_section(canvas, y, item)
        canvas, y, item, OUT, print = add_quorum_section(canvas, y, item)
        
        # Save
        output_path = OUT / f"{pair_id}_results.png"
        cv2.imwrite(str(output_path), canvas)
        print(f"Saved: {pair_id}")
    
    print("Done!")

        orb = item["detectors"]["orb"]
        line2 = f"ORB:   {orb['source_keypoints']} kp | {orb['filtered_matches']} matches | {orb['inliers']} inliers"
        cv2.putText(canvas, line2, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        y += 28
        
        # AKAZE metrics
        akaze = item["detectors"]["akaze"]
        line3 = f"AKAZE: {akaze['source_keypoints']} kp | {akaze['filtered_matches']} matches | {akaze['inliers']} inliers"
        cv2.putText(canvas, line3, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        y += 35
        return canvas, y, img_h, item, pair_id, OUT, print
