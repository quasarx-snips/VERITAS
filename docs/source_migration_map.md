# VERITAS — Source Migration Map

Authoritative map of the migration from the reference implementation repository into
the VERITAS architecture. Everything in this document was produced by reading the
complete source files listed below (none were modified).

## §0 Recorded source state (preservation snapshot)

| Field | Value |
|---|---|
| Source repository path | `c:\Users\bijan\Desktop\Veritas\sihcoremodule26166` |
| VERITAS repository path | `c:\Users\bijan\Desktop\Veritas` |
| Source branch | `master` |
| Source HEAD commit | `2f03393295b986aa38879772051d020037515f89` ("Pre release version") |
| Working tree | clean |
| Remote | `https://github.com/quasarx-snips/sihcoremodule26166.git` |
| Top-level tree | `.gitignore`, `README.md`, `requirements.txt`, `run_lunax.py`, `generate_harsh_samples.py`, `run_harsh_test_report.py`, `lunar_samples/`, `lunax/`, `models/`, `outputs/`, `tests/` |
| Baseline tests | `8 passed in 2.62s` (tests/test_geometry.py, test_matching.py, test_pipeline.py; the other 3 test files are empty) |
| Baseline environment | Python 3.14.7, opencv-python 5.0.0, numpy 2.5.2, pytest 9.1.1 (pytest installed separately; not in source `requirements.txt`) |

### §0.1 Migration log (commits)

| Commit | Scope |
|---|---|
| `bf61878` | `migration: add preprocessing primitives` (P0.1) |
| `4b928f2` | `migration: add SIFT feature extraction` (P0.2) |
| `c38ad0e` | `migration: add descriptor matching` (P0.3) |
| `d12f932` | `migration: add affine geometry` (P0.4) |
| `fa761a0` | `migration: add verification metrics` (P0.5) |

P0 is complete (all rows marked migrated below). P1 work (rows marked `P1`) has not started.

## §1 Classification legend (Phase 5)

| Class | Meaning |
|---|---|
| **A** | DIRECTLY REUSABLE — copy / adapt trivially |
| **B** | REUSABLE AFTER ADAPTATION — rename, strip lunar/terrain framing, or re-police policy |
| **C** | DOMAIN-SPECIFIC — DO NOT MIGRATE |
| **D** | REFERENCE ONLY — inspect for ideas; do not copy wholesale |

Priority codes: **P0.1..P0.5** and **P1** follow the mandated migration order.

## §2 Migration table

| Source file | Useful content | VERITAS destination | Priority | Status | Class |
|---|---|---|---|---|---|
| `lunax/preprocessing.py` | `ImagePreprocessor` — grayscale/BGR/BGRA/float load+normalise to finite uint8, CLAHE enhance, `process()` | `veritas/preprocessing/` (`preprocessor.py`, class `Preprocessor`) | P0.1 | migrated (bf61878) | A |
| `lunax/features.py` (part) | `TerrainFeature` dataclass → re-framed `FeatureEvidence`; `SiftDetector` (+ RootSIFT L1/sqrt normalisation); `FeatureStore` JSON/NPY persistence; `TerrainVisualizer` | `veritas/features/` (sift.py, store.py, visualization.py) + `veritas/schemas.py` | P0.2 (SIFT), P1 (visualizer) | migrated SIFT+store (4b928f2); visualizer P1 | A/B |
| `lunax/features.py` (part) | `ONNXCraterDetector`, `CraterDetector` — lunar crater segmentation (ONNX UNet + Hough fallback) | none — domain-specific | — | do not migrate | C |
| `lunax/features.py` (part) | `RidgeDetector`, `TextureGradientDetector` — generic Canny/Hough-Lines and Sobel-local-max structure detectors; re-framed as gradient/line complementary evidence families (drop "ridge"/"texture" lunar semantics) | `veritas/features/` (deferred) | P1 (feature quorum ingredients) | pending | B |
| `lunax/features.py` (part) | `TerrainFeatureExtractor` orchestrator + its CLI | architecture reference → `veritas/features/extractor.py` sketched during P0.2, not copied | P1 | pending | B/D |
| `lunax/matching.py` | `DEFAULT_CONFIG`, `MatchResult`, `_as_descriptors`, `_norm_for`, `match_descriptors` (BF/FLANN + LSH for uint8), `ratio_test` (Lowe), `mutual_consistency_filter`, `_flatten_matches`, `matches_to_correspondences`, `_feature_parts`, `_typed_candidate_matches`, `_unique_train_matches`, `match_feature_sets`, `visualize_matches` | `veritas/matching/` (`core.py`) | P0.3 | migrated (c38ad0e) | A/B |
| `lunax/matching.py` (part) | `visualize_ground_truth_matches`, `_ground_truth_metrics`, `run_sample_14_synthetic_demo`, `run_robustness_benchmark`, `_save_robustness_dashboard` — synthetic benchmark scaffolding | `scripts/` + `tests/benchmark/` (P1). NOTE: `run_robustness_benchmark` passes `descriptor_backend=` into `TerrainFeatureExtractor`, which accepts no such kwarg — latent source bug; rework when migrating | P1 | pending | B/D |
| `lunax/geometry.py` | `GeometricVerificationConfig`, `ErrorStatistics`, `VerificationDiagnostics`, `RegistrationResult`, `_normalize_points` (Hartley), `_fit_affine`, `_are_points_collinear`, `apply_transformation`, `compute_reprojection_errors`, `compute_error_statistics`, `classify_matches`, `_fit_model_by_type` (affine branch), `_ransac_estimator` (deterministic, dynamic iteration bound, refit on inliers), `verify_matches`, `verify_from_match_result` | `veritas/geometry/` (`affine.py`: `AffineVerificationConfig`, `AffineVerificationResult`, `verify_affine`, `verify_affine_from_correspondences`) | P0.4 | migrated (d12f932) | A/B |
| `lunax/geometry.py` (part) | `TransformModel.AUTO`, `HOMOGRAPHY`/`PROJECTIVE`, `estimate_geometry` AUTO branch, `_fit_homography` — automatic model selection + homography | explicitly excluded from VERITAS certification policy; document as reference only in `veritas/geometry/` NOTES | — | do not migrate into certification path | C/D |
| `lunax/geometry.py` (part) | `warp_image`, `register_images`, `visualize_inliers_outliers`, `_draw_line_fallback`, `_to_rgb_uint8` | `veritas/registration/` (planned later; visualisation) | P1+ | pending | B |
| `lunax/geometry.py` (part) | `process_lunar_image` (standalone legacy demo), `generate_synthetic_correspondences` (test fixtures), `run_unit_tests` | `generate_synthetic_correspondences` → tests fixtures (affine/translation/similarity only); demos → reference only | P1 (fixtures) | pending | B/D |
| `lunax/metrics.py` | `EvaluationThresholds`, `calculate_inlier_ratio`, `_matrix`, `calculate_reprojection_errors`, `calculate_rmse`, `calculate_error_statistics`, `calculate_spatial_coverage`, `evaluate_registration` | `veritas/verification/metrics.py` + `veritas/spatial/coverage.py` | P0.5 | migrated (fa761a0) | A |
| `lunax/metrics.py` (part) | `_histogram`, `create_residual_vector_visualization`, `print_registration_report` | the two report renderers migrated as private helpers into `veritas/verification/metrics.py` (fa761a0) because `evaluate_registration` embeds them; `print_registration_report` waits for the P1 audit renderer | P1 (audit renderer) | renderers migrated (fa761a0); print/report P1 | B |
| `lunax/refinement.py` | `RefinementConfig`, `CorrespondenceDiagnostics`, bounded local sub-pixel refinement (quadratic peak interp, correlation gating) | `veritas/refinement/` (planned later; P1+) | P1+ | pending | B |
| `lunax/registration.py` | `register_image`, `warp_image`, overlay/difference/`registration_visualization`, dtype-safe warping | `veritas/registration/` (planned later) | P1+ | pending | B |
| `lunax/pipeline.py` | `PipelineConfig`, `RegistrationResult`, orchestration flow, artifact saving — architecture reference only; DO NOT copy blindly | new `veritas/pipeline.py` written during wiring, after P0.5 | P0–P1 | reference | D/B |
| `lunax/pipeline.py` (part) | `_confidence` — heuristic evidence fusion (0.40·inlier_ratio + 0.25·count + 0.20·coverage + 0.15·error), explicitly NOT a calibrated probability | `veritas/verdict/` (evidence fusion) re-framed as heuristic evidence score | P1 (verdict engine) | pending | B |
| `lunax/augmentation.py` | `HarshVariant`, `create_harsh_variants`, `generate_harsh_sample_set` — deterministic harsh variants | `veritas/benchmark/` or `scripts/` (dataset generation) | P1 (benchmark) | pending | B |
| `generate_harsh_samples.py` | CLI generation script | `scripts/generate_benchmark_variants.py` | P1 | pending | B |
| `run_harsh_test_report.py` | 196-case regression runner + dashboard renderer | `scripts/benchmark_runner.py` (skeleton) | P1 | pending | B/D |
| `run_lunax.py` | CLI skeleton (argparse → pipeline → report → exit code) | `scripts/cli.py` (VERITAS CLI skeleton, wired after P0.5) | P0–P1 | pending | B |
| `lunax/__init__.py` | package facade exports | reference only; `veritas/__init__.py` rewritten during wiring | — | reference | D |
| `tests/test_features.py`, `test_metrics.py`, `test_registration.py` | empty files | — | — | ignore | D |
| `tests/test_geometry.py` | synthetic rotation+scale AUTO test | reworked as `tests/test_geometry_affine.py` — affine-only recovery, outlier rejection, collinearity (AUTO removed) | P0.4 | migrated (d12f932) | B |
| `tests/test_matching.py` | synthetic index-retention, empty/ratio safety, terrain-keypoint indexing, real-image SIFT+FLANN+viz tests | migrated as `tests/test_matching.py` (evidence-family keyed) | P0.3 | migrated (c38ad0e) | B |
| `tests/test_pipeline.py` | preprocessing float/colour, RANSAC outlier rejection, end-to-end known transform | preprocessing part migrated as `tests/test_preprocessing.py` (bf61878); RANSAC + end-to-end reworked as VERITAS per-stage tests once the pipeline chain is wired | P0.1 (done), P0.4 (done per-stage), wiring | partially migrated; chain wiring P1 | B |
| `models/crater_unet.onnx(.data)` | lunar crater segmentation UNet weights | none — domain-specific | — | do not migrate | C |
| `lunar_samples/`, `outputs/` | sample imagery and run artifacts | data fixtures only; do not commit large binaries into VERITAS (`data/raw`, `data/benchmark`) | — | reference | C/D |

## §3 P0 migration order

```
1. preprocessing   -> veritas/preprocessing/            (P0.1)
2. SIFT            -> veritas/features/sift.py          (P0.2)
3. matching        -> veritas/matching/                 (P0.3)
4. affine geometry -> veritas/geometry/                 (P0.4)
5. metrics         -> veritas/verification/ + spatial/  (P0.5)
```

## §4 P1 migration order

```
6.  ORB              -> veritas/features/orb.py      (independent evidence family)
7.  AKAZE            -> veritas/features/akaze.py    (independent evidence family)
8.  feature quorum   -> veritas/verification/         (across evidence families)
9.  spatial entropy  -> veritas/spatial/              (entropy over verified geometry)
10. counter-evidence -> veritas/counter_evidence/
```

## §5 Policy notes recorded during inspection

- VERITAS certification is **AFFINE-ONLY**. The source's `TransformModel.AUTO`,
  homography solver, and projective warping are reference-only material.
  `estimate_geometry`'s automatic model ranking (similarity -> affine -> homography)
  must NOT be migrated into the certification path.
- The source's literature vocabulary ("lunar imagery", "terrain landmarks",
  "crater enrichment", "sample_14 demo") is domain-specific and must not leak
  into VERITAS naming or docstrings.
- The source `_confidence` score is a heuristic evidence score and is already
  documented as "not a calibrated probability"; VERITAS keeps that framing.
- `requirements.txt` of the source (`numpy`, `opencv-python`, `onnxruntime`) —
  onnxruntime is only needed by the domain-specific ONNX crater model (class C)
  and is therefore NOT carried into VERITAS.