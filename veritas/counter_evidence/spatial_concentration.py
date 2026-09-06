"""Spatial concentration facts derived from coverage and entropy."""
def summarize_spatial_concentration(coverage, entropy):
    return {"occupied_cells": coverage["occupied_cells"], "total_cells": coverage["total_cells"],
            "coverage_ratio": coverage["coverage_ratio"], "normalized_entropy": entropy["normalized_entropy"],
            "concentrated": bool(coverage["occupied_cells"] <= 1)}
