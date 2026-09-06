
import json
import pandas as pd
from pathlib import Path

def analyze_results():
    """
    Analyzes the audit results and generates a CSV and a markdown summary.
    """
    results_path = Path("outputs/real_world_audit/results.json")
    output_dir = results_path.parent

    with open(results_path, "r") as f:
        results = json.load(f)

    # Create a flat list of dictionaries for the CSV
    flat_results = []
    for result in results:
        evidence = result["evidence"]
        geometry = evidence["geometry"]
        spatial = evidence["spatial"]
        quorum = evidence["quorum"]
        counter_evidence = evidence["counter_evidence"]

        flat_results.append(
            {
                "pair_id": result["pair_id"],
                "image_a": result["image_a"],
                "image_b": result["image_b"],
                "sift_matches": evidence["matches"]["sift"]["filtered_count"],
                "orb_matches": evidence["matches"]["orb"]["filtered_count"],
                "akaze_matches": evidence["matches"]["akaze"]["filtered_count"],
                "affine_inliers": geometry["inlier_count"],
                "inlier_ratio": geometry["inlier_ratio"],
                "rmse": geometry["residuals"].get("rmse"),
                "p95": geometry["residuals"].get("p95"),
                "coverage": spatial["coverage_ratio"],
                "entropy": spatial["normalized_entropy"],
                "quorum": quorum["quorum_strength"],
                "concentration": counter_evidence["spatial_concentration"].get("concentration_ratio"),
                "counter_evidence": sum(
                    [
                        len(counter_evidence["residuals"]),
                        len(counter_evidence["feature_disagreement"]),
                        len(counter_evidence["spatial_concentration"]),
                    ]
                ),
                "fusion_summary": evidence.get("evidence_score", "NA"),
            }
        )

    # Create and save the CSV
    df = pd.DataFrame(flat_results)
    df.to_csv(output_dir / "results.csv", index=False)
    print(f"CSV results saved to {output_dir / 'results.csv'}")

    # Create the markdown summary
    summary_md = "# VERITAS Real-World Phase 2 Audit\n\n"
    summary_md += "## 1. Dataset Inventory\n\n"
    summary_md += f"- Number of image pairs evaluated: {len(results)}\n"
    summary_md += "- Images are sourced from `data/raw` and `data/processed`.\n\n"

    summary_md += "## 2. Pair Selection\n\n"
    summary_md += "Pairs were formed by matching `ds_*.png`/`ds_*.jpg` from `data/raw` with `ds_*.jpg` from `data/processed`.\n\n"

    summary_md += "## 3. Overall Results\n\n"
    summary_md += df.to_markdown(index=False)

    with open(output_dir / "summary.md", "w") as f:
        f.write(summary_md)

    print(f"Markdown summary saved to {output_dir / 'summary.md'}")


if __name__ == "__main__":
    analyze_results()