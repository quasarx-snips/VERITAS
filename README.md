# VERITAS

**Correspondence Verification and Gating for Geospatial Image Pairs**

VERITAS is a deterministic evidence engine designed to answer a single, critical question before any downstream change detection occurs: Is the geometric relationship between these two images trustworthy enough to act upon? 

It does not guess. It certifies.

---

## The Problem: The False-Positive Crisis

Standard computer vision pipelines for satellite imagery often rely on feature matching and homography to align images. The underlying assumption is simple: if enough feature points match, the images must be aligned. 

In practice, this assumption leads to a massive false-positive problem in geospatial analysis:
- **Repetitive textures** (solar farms, desert dunes, urban grids) generate hundreds of mathematically close but geographically incorrect matches.
- **Over-flexible models** (like homography or auto-selection) will happily warp an image to force these false matches to fit, certifying a misalignment as valid.
- **Partial overlap** (caused by clouds, floods, or sensor footprints) collapses global alignment even when a meaningful region of the image is perfectly valid.

When downstream systems run change detection on this unverified geometry, they flood human analysts with false alarms. At a planetary scale, even a small false-positive rate becomes an unmanageable avalanche of wasted labor.

## The VERITAS Thesis

VERITAS approaches this problem by changing the question. Instead of only looking for evidence that *supports* a match, it actively looks for evidence that *contradicts* it.

It gathers supporting evidence (inliers, spatial distribution, feature agreement) and counter-evidence (high residuals, feature disagreement, spatial clustering) and fuses them into a structured certificate. This certificate determines a verdict, which in turn drives a downstream gate action:

| Verdict | Meaning | Gate Action |
|---------|---------|-------------|
| **STRONG** | Global geometric and spatial support is high; counter-evidence is low. | `ALLOW` |
| **PARTIAL** | Global alignment is weak, but credible local regions exist. | `RESTRICT` or `HUMAN_REVIEW` |
| **WEAK** | Some evidence exists, but contradictions dominate the signal. | `HUMAN_REVIEW` |
| **NONE** | No credible geometric certificate can be formed. | `BLOCK` |

---

## Master Architecture

```
INPUT
  ↓
EVIDENCE EXTRACTION
  ├─ Preprocessing (Normalization / NaN-safety / CLAHE)
  ├─ SIFT (Float descriptors)
  ├─ ORB (Binary descriptors)
  ├─ AKAZE (Binary descriptors)
  ├─ Feature Quorum (Family-level evidence aggregation)
  └─ Descriptor Matching (Ratio / Mutual / Unique-train filtering)
       ↓
GEOMETRIC CERTIFICATION
  ├─ AFFINE-ONLY RANSAC (6-DOF conservative certificate)
  └─ Partial Correspondence (Local region certificates)
       ↓
SPATIAL INTELLIGENCE + COUNTER-EVIDENCE
  ├─ Coverage and Normalized Shannon Entropy
  ├─ Residual Quality (RMSE / P95 / Spread)
  ├─ Feature Disagreement (Cross-family conflict)
  └─ Spatial Concentration (Clustered-support trap detection)
       ↓
DECISION LAYER
  ├─ Evidence Fusion (Auditable structured evidence)
  ├─ Verdict Classifier (STRONG / PARTIAL / WEAK / NONE)
  └─ Downstream Gate (ALLOW / RESTRICT / REVIEW / BLOCK)
       ↓
EXPLAINABILITY
  ├─ Audit Report Generation
  ├─ Zero-Pixel LLM Explainer (SGLang structured JSON)
  └─ ElevenLabs Voice Briefing
```

### The Evidence Philosophy

```
Supporting Evidence                Counter-Evidence
(inliers, coverage,          +      (residuals, disagreement,
 entropy, quorum)                    concentration)
        ↓                                   ↓
        └──────────────┬──────────────┘
                       ↓
                    Verdict
                       ↓
                  Gate Action
```

---

## Core Principles

1. **Affine-Only Certification:** We explicitly restrict geometric certification to 2D affine transformations. We do not use auto-selection or homography fallbacks, as they tend to warp reality to fit false matches in local geospatial patches.

2. **Complementary Evidence Families:** SIFT, ORB, and AKAZE are used as independent implementation paths with different failure modes. Agreement across families is evidence; disagreement is counter-evidence.

3. **Spatial Entropy vs. Coverage:** Coverage measures how much of the scene has matches. Entropy measures how evenly those matches are distributed. A thousand matches concentrated in one corner is a spatial trap, not a success.

4. **Active Counter-Evidence:** The system builds a case against the match as rigorously as it builds the case for it.

5. **Partial Correspondence:** If global alignment fails due to clouds or sensor footprint, the system checks for credible local regions rather than failing outright.

6. **Deterministic Decisions, Zero-Pixel Explainability:** The decision engine relies strictly on structured metrics. The LLM explainer never receives raw pixels; it only reads the final evidence to generate a human-readable summary.

7. **Scientific Language Discipline:** We are careful with terminology. We calculate evidence scores and geometric certificates, not absolute probabilities.

---

## Repository Structure

```
veritas/
├── pipeline.py            # Bounded orchestration
├── schemas.py             # Structured evidence contracts
├── config.py              # Policy configuration (AFFINE-ONLY lock)
├── preprocessing/         # Normalization and enhancement
├── features/              # SIFT, ORB, AKAZE, and quorum logic
├── matching/              # Descriptor matching and filtering
├── geometry/              # Affine RANSAC and certificates
├── spatial/               # Coverage, entropy, and distribution
├── counter_evidence/      # Residuals, disagreement, concentration
├── verification/          # Metrics, evidence fusion, and partial checks
├── verdict/               # Classifier and thresholds
├── gate/                  # Downstream gating actions
├── audit/                 # Audit report generation
└── llm/                   # Zero-pixel explainer (SGLang integration)
tests/                     # Full pytest suite
scripts/                   # CLI and benchmark harness
demo/                      # Review demonstrations
docs/                      # Architecture notes
```

---

## Roadmap

### Phase 1: Core Engine ✓
- [x] Preprocessing (normalization, CLAHE, dtype/NaN safety)
- [x] SIFT extraction (RootSIFT optional)
- [x] Descriptor matching (ratio, mutual consistency, unique-train filtering)
- [x] Affine-only RANSAC with deterministic seeding
- [x] Verification metrics (RMSE, P95, inlier statistics)
- [x] Spatial coverage

### Phase 2: Evidence Intelligence ✓
- ORB + AKAZE complementary families, detector quorum, coverage and entropy
- Counter-evidence, evidence fusion, and partial correspondence

### Phase 3: Decision + Safety Backend ✓

```
Evidence → Partial correspondence → Verdict → Gate → Audit → Explanation payload
```

`STRONG`, `PARTIAL`, `WEAK`, and `NONE` are deterministic verdict classes.
They map by default to `ALLOW`, `RESTRICT`, `HUMAN_REVIEW`, and `BLOCK`.
The gate is binding: a future LLM cannot override it.

Verdict thresholds are provisional operational heuristics, not calibrated
probabilities. Calibration remains future work. The explanation payload is
provider-independent and contains no image pixels; it is ready for future
SGLang and TTS adapters, but this repository includes neither runtime.

---

## Benchmark Philosophy

The benchmark suite is designed to evaluate the decision layer, not merely the matching algorithms. Planned adversarial classes include:

1. Genuine distributed correspondence → Expect `ALLOW`
2. Illumination shift
3. Rotation and combined transforms
4. Partial overlap (flood or cloud cover)
5. Repetitive-texture concentration trap → Expect `BLOCK`
6. Feature-family disagreement
7. High-residual false correspondence

Ablation toggles (SIFT-only vs. quorum, with/without entropy, with/without counter-evidence) demonstrate the value of each architectural layer.

---

## Getting Started

```bash
git clone https://github.com/quasarx-snips/VERITAS.git
cd VERITAS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Run a complete backend decision and write JSON artifacts:

```bash
python -m scripts.run_veritas --before data/raw/ds_1.png --after data/processed/ds_1.jpg
```

The command writes the complete report and a zero-pixel explanation payload to
`outputs/reports/`. No original source repository, TTS provider, SGLang server,
or API key is required.

---
(ignore)
## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
