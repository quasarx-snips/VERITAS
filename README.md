# VERITAS

VERITAS is a correspondence verification and gating system for image pairs.

It does **not** re-implement the old `lunax` application architecture. Instead, it
extracts proven implementation primitives from the reference repository
(`sihcoremodule26166`, kept untouched beside this repository) and re-hosts them
under `veritas/` with clean naming, no domain-specific assumptions, and no runtime
dependency on the source repository.

## Core conceptual pipeline

```
IMAGE PAIR
→ preprocessing
→ independent feature evidence
→ descriptor matching
→ geometric verification
→ spatial evidence
→ counter-evidence
→ evidence fusion
→ verdict
→ downstream gate
→ audit report
```

## Geometry certification policy — AFFINE-ONLY

- The only geometric model accepted for VERITAS correspondence **certification** is the **affine** model.
- There is **no automatic model selection** for certification.
- There is **no silent fallback to homography**; homography is **not** an accepted
  VERITAS certification model.
- Homography-related code in the reference repository is treated as **reference-only** material.

Authorization: `veritas/config.py` (`CERTIFICATION_MODELS = ("affine",)`).

## Scientific language

We do not claim strength the evidence does not support:

- SIFT / ORB / AKAZE are treated as **complementary evidence families** and
  **independent implementation paths** with **different descriptor failure modes**,
  not as universally independent statistical measurements.
- The fused evidence score is a **heuristic evidence score**, never a probability
  and never "calibrated confidence" (no calibration has been implemented yet).
- The affine certificate is a **conservative geometric certificate**, not a claim
  that affine geometry perfectly models every scene.

## Repository layout

```
veritas/                 VERITAS python package
  pipeline.py            orchestration skeleton (stages fixed above)
  schemas.py             planned schema contracts
  config.py              policy configuration (AFFINE-ONLY)
  preprocessing/         image loading / normalisation / enhancement
  features/              independent feature-evidence families (SIFT, then ORB/AKAZE)
  matching/              descriptor matching primitives
  geometry/              AFFINE-ONLY geometric verification
  spatial/               spatial evidence (coverage / density / entropy)
  counter_evidence/      counter-evidence checks
  verification/          verification metrics + evidence fusion
  verdict/               verdict decision from fused evidence
  gate/                  downstream gating
  audit/                 audit report generation
  llm/                   LLM-assisted explainability (future work)
docs/source_migration_map.md   authoritative migration map against the source repo
tests/                   VERITAS tests (run with pytest)
scripts/                 CLI / benchmark scripts
data/                    raw / processed / benchmark data
outputs/                 run artifacts
demo/                    runnable demos
```

## Current status

**P0 migration complete** (5 migration commits):

| Commit | Scope |
|---|---|
| `bf61878` | `migration: add preprocessing primitives` |
| `4b928f2` | `migration: add SIFT feature extraction` |
| `c38ad0e` | `migration: add descriptor matching` |
| `d12f932` | `migration: add affine geometry` |
| `fa761a0` | `migration: add verification metrics` |

Implemented: `veritas/preprocessing` (ImagePreprocessor → normalization/CLAHE
split with `PreprocessedImage` result), `veritas/features` (SIFT +
RootSIFT + FeatureStore, `FeatureEvidence` in `veritas.schemas`), `veritas/matching`
(correspondences), `veritas/geometry` (AFFINE-ONLY `verify_affine`), `veritas/verification`
+ `veritas/spatial` (metrics + coverage). Matching stops at descriptor
correspondences; only the affine certificate is accepted.

Remaining migration order:

- **P1 (6–10):** ORB → AKAZE → feature quorum → spatial entropy → counter-evidence
- **After P1:** pipeline wiring, verdict engine, downstream gate, audit report,
  refinement/registration/visualization modules, benchmark scripts.

## Running tests

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -v
```

### Bytecode-cache policy (one single cache tree)

All Python bytecode caches are kept in **one** location: `.pycache/` (git-ignored).
The root `conftest.py` sets `sys.pycache_prefix`, so pytest runs never scatter
`__pycache__/` directories across the package tree. For runs outside pytest,
point the interpreter at the same tree once per shell:

```powershell
$env:PYTHONPYCACHEPREFIX = "$PWD\.pycache"
```

(`.pytest_cache/` is pytest's own bookkeeping cache, also git-ignored; it is not
Python bytecode.)