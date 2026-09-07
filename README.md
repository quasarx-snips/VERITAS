# VERITAS — Automated Verification & Forensic Counsel Platform

**VERITAS** (*Verifiable Evidence Reasoning with Independent Topological Alignment Systems*) is a zero-pixel deterministic dual-image correspondence verification and forensic intelligence engine. [CHECK]

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Requirements: `numpy`, `opencv-contrib-python>=4.8,<5`, and `pyyaml`)*

### 2. Configure LLM API Key (Optional but Recommended)
Set your `GROQ_API_KEY` for ultra-fast, in-depth forensic reasoning (or `GEMINI_API_KEY`):

```bash
# In your terminal or in a local .env file:
export GROQ_API_KEY="gsk_..."
```

### 3. Launch the Platform Server
```bash
python -m veritas.api --port 8000
```
- **Server Address**: `http://localhost:8000`
- **Landing Page**: [http://localhost:8000/landing_page.html](http://localhost:8000/landing_page.html)
- **Counsel Agent**: [http://localhost:8000/counsel_agent.html](http://localhost:8000/counsel_agent.html)
- **Interactive Verification**: [http://localhost:8000/analysis_page.html](http://localhost:8000/analysis_page.html)
- **Health Check**: `GET http://localhost:8000/api/health`

---

## 🏛️ Platform Architecture

VERITAS is architected around two synchronized engines:

### 1. Deterministic Verification Pipeline (`veritas/`)
- **Zero-Pixel Privacy**: Operates strictly on extracted keypoint coordinates and descriptor correspondences without inspecting semantic image contents.
- **Multi-Detector Quorum**: Triangulates correspondences using three complementary detector paradigms:
  - **SIFT**: Scale-space continuous Gaussian gradients.
  - **ORB**: High-speed binary FAST corners and rotated BRIEF descriptors.
  - **AKAZE**: Non-linear diffusion edge-preserving filtering.
- **Affine Certificate Engine**: Enforces rigid/affine geometric invariants via RANSAC with strict condition number, determinant, and sub-pixel reprojection residual bounds.
- **Spatial Coverage & Entropy**: Evaluates 2D Shannon spatial entropy and grid distribution to prevent false positives from localized feature clustering.
- **Safety Gate Directives**: Assigns deterministic policy actions: `ALLOW`, `RESTRICT`, `HUMAN_REVIEW`, and `BLOCK`.

### 2. Forensic Counsel Agent (`pages/counsel_agent.html`)
- **Teacher-Student Forensic Masterclass**: Delivers in-depth, pedagogical evaluations that deconstruct reason codes, explain mathematical anomalies, and provide concrete operator checklists.
- **Groq Llama 3.3 70B Versatile**: Sub-second high-throughput conversational reasoning grounded in mathematical verification dossiers.
- **Dual-Layer Rich Markdown Engine**: Renders tables, section headers, code spans, alert callouts (`> [!IMPORTANT]`, `> [!NOTE]`), and clean mathematical notation in Midnight Sakura (dark) and light modes.
- **Client Cache Synchronization**: Automatically syncs verification telemetry and detector keypoints between Flow 1 and the Counsel Agent via `veritas_store.js`.

---

## 📡 API Endpoints

### 1. Health Check
```http
GET /api/health
```
```json
{
  "service": "VERITAS Backend Verification Engine",
  "status": "healthy",
  "version": "1.0.0",
  "detectors": ["SIFT", "ORB", "AKAZE"],
  "gate_actions": ["ALLOW", "RESTRICT", "HUMAN_REVIEW", "BLOCK"]
}
```

### 2. Image Verification
```http
POST /api/verify
Content-Type: application/json
```
```json
{
  "before_base64": "data:image/png;base64,...",
  "after_base64": "data:image/jpeg;base64,..."
}
```

### 3. Forensic Counsel Chat
```http
POST /api/chat
Content-Type: application/json
```
```json
{
  "prompt": "Why did VERITAS reach this verdict? Walk me through the reason codes.",
  "payload": { ... },
  "messages": [
    { "role": "user", "content": "Why did VERITAS reach this verdict?" }
  ]
}
```

Counsel only responds to geospatial-imagery topics (remote sensing, satellite/aerial imagery, GIS, map analysis, and VERITAS image verification). Chat requests are limited to one per client every 5 seconds (12/minute) and return `429` plus `Retry-After` when exceeded. Set `VERITAS_CHAT_COOLDOWN_SECONDS` in `.env` to a stricter value if your provider quota requires it.

---

## 📁 Directory Structure

```text
VERITAS/
├── FRONTEND_HANDOVER.md    # Complete frontend API & UI integration specification
├── README.md               # Quickstart & platform guide
├── requirements.txt        # Minimal python dependencies
├── configs/                # Default threshold and pipeline configurations
├── pages/                  # Web user interface
│   ├── landing_page.html   # Main platform landing & mode toggle
│   ├── upload_image.html   # Dual-image upload flow
│   ├── analysis_page.html  # Interactive visual verification dashboard
│   ├── counsel_agent.html  # Forensic Counsel conversational interface
│   └── veritas_store.js    # IndexedDB telemetry & image cache engine
├── samples/                # Sample image pair for testing
│   ├── sample_before.png
│   └── sample_after.jpg
├── scripts/                # Verification CLI utilities
│   └── run_veritas.py
├── tests/                  # Unit and integration test suite
│   ├── test_api.py
│   └── test_runtime_adapters.py
└── veritas/                # Core verification engine
    ├── api.py              # Zero-dependency CORS REST API server
    ├── pipeline.py         # verify_evidence_pair() entrypoint
    ├── features/           # SIFT, ORB, AKAZE detectors
    ├── geometry/           # Affine certification and RANSAC
    ├── matching/           # Descriptor matching & ratio filtering
    ├── spatial/            # Coverage and entropy analysis
    ├── verdict/            # Verdict classifier & heuristics
    ├── gate/               # Deterministic safety gate
    ├── counter_evidence/   # Residuals & disagreement tracking
    ├── llm/                # Groq and Gemini client adapters
    └── audit/              # Run ID and provenance tracking
```

---

## 🧪 Testing

Execute the test suite:
```bash
pytest tests/
```
All tests verify pipeline determinism, adapter fallback, context construction, and rich markdown formatting.
