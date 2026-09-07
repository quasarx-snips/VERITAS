# VERITAS — Frontend Developer Handover & Integration Guide

Welcome! This package is the clean, stripped-down backend distribution of **VERITAS** (Verifiable Evidence Reasoning with Independent Topological Alignment Systems). 

All heavy test outputs, bulky training datasets, and unit test suites have been removed. This package contains only the **production core backend engine**, the **REST API server**, and lightweight sample images for immediate testing.

---

## 1. Quickstart (Start the Backend Server)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
*(Requirements are minimal: `numpy`, `opencv-contrib-python>=4.8,<5`, and `pyyaml`)*

### Step 2: Launch the API Server
```bash
python -m veritas.api --port 8000
```
Server runs by default on `http://localhost:8000` with **CORS enabled** (`*`), allowing direct fetch/axios calls from `localhost:3000`, `localhost:5173`, or any frontend dev server.

---

## 2. API Endpoints Specification

### Endpoint 1: Health Check
- **URL**: `GET /api/health`
- **Response**:
```json
{
  "service": "VERITAS Backend Verification Engine",
  "status": "healthy",
  "version": "1.0.0",
  "detectors": ["SIFT", "ORB", "AKAZE"],
  "gate_actions": ["ALLOW", "RESTRICT", "HUMAN_REVIEW", "BLOCK"]
}
```

---

### Endpoint 2: Image Verification & Safety Gate
- **URL**: `POST /api/verify`
- **Content-Type**: `application/json`

#### Option A: Sending File Paths (e.g. Local Server Files)
```json
{
  "before_path": "samples/sample_before.png",
  "after_path": "samples/sample_after.jpg"
}
```

#### Option B: Sending Browser-Uploaded Images (Base64)
Ideal when the user selects files in the browser via `<input type="file">`:
```json
{
  "before_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "after_base64": "/9j/4AAQSkZJRgABAQAAAQ..."
}
```

---

## 3. Response Schema (Frontend Contract)

The backend returns a comprehensive JSON object:

```json
{
  "status": "success",
  "verdict": {
    "verdict": "PARTIAL",
    "rationale_codes": ["AFFINE_CERTIFIED", "HIGH_RESIDUAL", "PARTIAL_OVERLAP"],
    "threshold_profile": "default"
  },
  "gate": {
    "action": "RESTRICT",
    "reasons": ["AFFINE_CERTIFIED", "HIGH_RESIDUAL", "PARTIAL_OVERLAP"]
  },
  "partial_correspondence": {
    "status": "PARTIAL",
    "rationale": "Supported regions meet threshold criteria",
    "confidence": 0.85
  },
  "explanation": {
    "headline": "VERITAS verdict: PARTIAL; gate: RESTRICT.",
    "key_reasons": [
      "The affine certificate was verified.",
      "Reprojection residuals are high.",
      "Local regions provide partial correspondence support."
    ],
    "recommended_next_action": "Use only supported regions or apply the configured restricted workflow.",
    "prohibited_claims": [
      "Do not claim geographic identity without evidence.",
      "Do not call heuristic scores probabilities.",
      "Do not claim semantic change from correspondence evidence.",
      "Do not override the deterministic gate."
    ],
    "gate_action": "RESTRICT",
    "verdict": "PARTIAL",
    "partial_correspondence": {
      "coverage_fraction": 0.75,
      "supported_regions": [[1, 0], [1, 1], [1, 2], [1, 3], [2, 0], [2, 1], [2, 2], [2, 3], [3, 0], [3, 1], [3, 2], [3, 3]],
      "unsupported_regions": [[0, 0], [0, 1], [0, 2], [0, 3]],
      "region_evidence": [ ...16 grid cells with inlier & residual metrics... ]
    }
  },
  "audit": {
    "run_id": "51b8a62c-85e7-4c74-b63b-f0c56b2d412f",
    "timestamp": "2026-09-07T04:10:59.123456"
  }
}
```

---

## 4. UI Components & Display Guide

### A. Verdict Badges & Color Coding
| Verdict | Safety Gate Action | Color / Status | Meaning |
| :--- | :--- | :--- | :--- |
| `STRONG` | `ALLOW` | **Green (#10B981)** | Full alignment confirmed; downstream change detection is safe. |
| `PARTIAL` | `RESTRICT` | **Yellow/Orange (#F59E0B)** | Alignment valid only in specific regions; restrict inspection. |
| `WEAK` | `HUMAN_REVIEW` | **Purple (#8B5CF6)** | Low confidence; requires human operator intervention. |
| `NONE` | `BLOCK` | **Red (#EF4444)** | No valid correspondence; completely block comparison. |

### B. Rendering the 4x4 Spatial Correspondence Grid
The image is divided into a **4-row by 4-column grid** (16 cells total).
- Look at `data.explanation.partial_correspondence.supported_regions` and `unsupported_regions`.
- Each entry is `[row, col]` where `row ∈ [0..3]` and `col ∈ [0..3]`.
- **Supported regions**: Render with a subtle green tint (`rgba(16, 185, 129, 0.2)`).
- **Unsupported regions**: Render with a subtle red/crosshatch tint (`rgba(239, 68, 68, 0.3)`).

#### React / JSX Example:
```jsx
function SpatialGridOverlay({ supportedRegions, unsupportedRegions }) {
  const isSupported = (r, c) => 
    supportedRegions.some(([row, col]) => row === r && col === c);

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(4, 1fr)',
      gridTemplateRows: 'repeat(4, 1fr)',
      width: '100%',
      height: '100%',
      position: 'absolute',
      top: 0,
      left: 0,
      pointerEvents: 'none'
    }}>
      {Array.from({ length: 16 }).map((_, i) => {
        const row = Math.floor(i / 4);
        const col = i % 4;
        const active = isSupported(row, col);
        return (
          <div
            key={i}
            style={{
              border: '1px solid rgba(255,255,255,0.15)',
              backgroundColor: active ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.35)'
            }}
          />
        );
      })}
    </div>
  );
}
```

---

## 5. Integrating with Text-To-Speech (TTS) & Voice Assistants

The `explanation` payload is designed specifically for **zero-pixel text-to-speech**:
- **Introduction**: `data.explanation.headline`
- **Details**: `data.explanation.key_reasons.join(". ")`
- **Recommendation**: `data.explanation.recommended_next_action`

#### Frontend Web Speech API (One-Liner):
```javascript
function speakVerdict(explanation) {
  const speechText = `${explanation.headline}. ${explanation.key_reasons.join('. ')}. Recommended action: ${explanation.recommended_next_action}`;
  const utterance = new SpeechSynthesisUtterance(speechText);
  utterance.rate = 1.0;
  window.speechSynthesis.speak(utterance);
}
```

---

## 6. Python Direct Import (Alternative to HTTP)

If the frontend uses a Python backend framework (e.g. Next.js with Python API, Django, Flask, FastAPI), you can call the engine directly without running the separate server:

```python
from veritas.pipeline import verify_evidence_pair

result = verify_evidence_pair("samples/sample_before.png", "samples/sample_after.jpg")

# Direct access to all metrics:
print("Verdict:", result.verdict.verdict.value)
print("Gate Action:", result.gate.action.value)

# Convert to JSON serializable dictionaries:
summary_dict = result.to_dict()
explanation_dict = result.explanation_payload.to_dict()
```

---

## 7. Sample Files Included
In the `samples/` directory:
- `samples/sample_before.png`: Reference image
- `samples/sample_after.jpg`: Comparative image
You can immediately test the API with these samples:
```bash
curl -X POST http://localhost:8000/api/verify \
  -H "Content-Type: application/json" \
  -d "{\"before_path\": \"samples/sample_before.png\", \"after_path\": \"samples/sample_after.jpg\"}"
```
