"""Helpers that map concrete Phase 1/2 outputs into evidence contracts."""
from veritas.schemas import GeometryEvidence

def geometry_evidence(certificate):
    stats = certificate.diagnostics.error_stats
    matrix = certificate.transformation.tolist() if certificate.transformation is not None else None
    return GeometryEvidence(certificate.model, certificate.is_valid, certificate.candidate_count, certificate.inlier_count,
                            certificate.inlier_ratio, {"mean": stats.mean, "median": stats.median, "rmse": stats.rmse,
                            "p95": stats.percentile_95, "maximum": stats.max_error}, matrix)
