
import json
import os
from pathlib import Path
import cv2
from veritas.pipeline import verify_evidence_pair

def run_audit():
    """
    Runs the VERITAS Phase 2 audit.
    """
    output_dir = Path("outputs/real_world_audit")
    output_dir.mkdir(exist_ok=True)

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    image_pairs = []
    for i in range(1, 101):
        raw_path_png = raw_dir / f"ds_{i}.png"
        raw_path_jpg = raw_dir / f"ds_{i}.jpg"
        processed_path = processed_dir / f"ds_{i}.jpg"

        raw_path = None
        if raw_path_png.exists():
            raw_path = raw_path_png
        elif raw_path_jpg.exists():
            raw_path = raw_path_jpg

        if raw_path and processed_path.exists():
            image_pairs.append(
                {
                    "pair_id": f"pair_{i:03d}",
                    "image_a": str(raw_path),
                    "image_b": str(processed_path),
                }
            )

    results = []
    for pair in image_pairs:
        print(f"Processing {pair['pair_id']}...")
        image_a = cv2.imread(pair["image_a"])
        image_b = cv2.imread(pair["image_b"])

        evidence = verify_evidence_pair(image_a, image_b)
        
        result_data = {
            "pair_id": pair["pair_id"],
            "image_a": pair["image_a"],
            "image_b": pair["image_b"],
            "evidence": evidence.to_dict(),
        }
        results.append(result_data)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Audit complete. Results saved to {output_dir / 'results.json'}")

if __name__ == "__main__":
    run_audit()