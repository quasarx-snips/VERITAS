# VERITAS Backend — Frontend Developer Handover

Welcome to the **VERITAS** backend distribution for frontend integration.

This folder contains the core deterministic verification engine and REST API server. All unneeded test outputs, bloated datasets, and unit test suites have been removed to keep this workspace ultra-clean (**~4 MB**).

For full endpoint specifications, response payloads, React/Vue grid code snippets, and TTS voice integration, see:
👉 **[FRONTEND_HANDOVER.md](FRONTEND_HANDOVER.md)**

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python -m veritas.api --port 8000
```
- Server URL: `http://localhost:8000`
- Health check: `GET http://localhost:8000/api/health`
- Verification API: `POST http://localhost:8000/api/verify` (CORS enabled)

### 3. Test Verification with Included Samples
```bash
python -m scripts.run_veritas --before samples/sample_before.png --after samples/sample_after.jpg
```

---

## Directory Structure

```text
Veritas - Copy/
├── FRONTEND_HANDOVER.md    # Complete frontend API & UI integration specification
├── README.md               # Quickstart guide
├── requirements.txt        # Minimal python dependencies
├── configs/                # Default threshold and pipeline configurations
├── pages/                  # KompilerZ UI prototype design reference
├── samples/                # Sample image pair for immediate testing
│   ├── sample_before.png
│   └── sample_after.jpg
├── scripts/                # CLI runner scripts
│   └── run_veritas.py
└── veritas/                # Core backend verification package
    ├── api.py              # Zero-dependency CORS REST API server
    ├── pipeline.py         # verify_evidence_pair() entrypoint
    ├── features/           # SIFT, ORB, AKAZE detectors
    ├── geometry/           # Affine certification and RANSAC
    ├── matching/           # Descriptor matching & ratio filtering
    ├── spatial/            # Coverage and entropy analysis
    ├── verdict/            # Verdict classifier & heuristics
    ├── gate/               # Deterministic safety gate
    ├── counter_evidence/   # Residuals & disagreement tracking
    ├── llm/                # Zero-pixel explanation payload schemas
    └── audit/              # Run ID and provenance tracking
```
