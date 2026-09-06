# VERITAS

**Trust correspondence before trusting change.**

VERITAS is a deterministic verification and safety-gating layer for image pairs. Before downstream change detection is trusted, it asks whether geometric correspondence is supported by inspectable evidence.

## Run the demo

```bash
python -m scripts.build_prototype_model
```

Then open `outputs/prototype_model/VERITAS_demo_report.html`. It presents real curated runs, including affine-tolerant, partial, disagreement, visually disruptive, and blocked cases. It also creates per-case evidence panels, human-readable report pages with image previews, a contact sheet, decision trace, spatial view, and a false-alarm-defense walkthrough.

To run one image pair directly:

```bash
python -m scripts.run_veritas --before data/raw/ds_1.png --after data/processed/ds_1.jpg
```

The command prints and writes verdict, deterministic gate action, detector/geometry/spatial/counter-evidence, partial correspondence, an audit report with provenance, and a zero-pixel explanation payload. No external LLM, TTS runtime, API key, or source repository is required.

## What it does

```text
IMAGE PAIR → PREPROCESSING → SIFT / ORB / AKAZE → MATCHING
→ AFFINE-ONLY GEOMETRY → SPATIAL + COUNTER-EVIDENCE → EVIDENCE FUSION
→ PARTIAL CORRESPONDENCE → VERDICT → SAFETY GATE → AUDIT → EXPLANATION PAYLOAD
```

| Verdict | Default gate |
|---|---|
| `STRONG` | `ALLOW` |
| `PARTIAL` | `RESTRICT` |
| `WEAK` | `HUMAN_REVIEW` |
| `NONE` | `BLOCK` |

The gate is deterministic and binding. A future explanation adapter cannot override it.

## Engineering principles

- **Affine-only geometry:** no automatic model selection and no homography fallback.
- **Complementary detectors:** SIFT, ORB, and AKAZE statuses are recorded separately; agreement and disagreement both remain visible.
- **Spatial evidence:** coverage and normalized entropy distinguish available matches from scene-wide support.
- **Counter-evidence:** residuals, detector disagreement, and concentration are preserved in the decision record.
- **Auditable contracts:** evidence, verdict, gate, audit, provenance, and explanation payload are serializable outputs.

## Current status

Phase 1, Phase 2, and the major Phase 3 decision/safety backend are implemented. Verdict thresholds are operational heuristics, not calibrated probabilities. The repository does not claim physical-world change ground truth or benchmark accuracy.

Future integrations are intentionally separate: SGLang runtime, TTS runtime, final production UI, benchmark calibration, and scientific hardening. The current explanation payload is provider-independent and contains no raw pixels.

## Development

```bash
pip install -r requirements.txt
pytest -q
```

`outputs/` is generated locally and intentionally excluded from version control. Regenerate the prototype model with `python -m scripts.build_prototype_model`.
