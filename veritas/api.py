"""Lightweight REST API bridge connecting VERITAS backend to frontend clients.

Provides a zero-dependency HTTP server with CORS support and JSON endpoints.

Usage:
    python -m veritas.api --port 8000
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from veritas.pipeline import verify_evidence_pair
from veritas.features import SiftDetector, OrbDetector, AkazeDetector
from veritas.preprocessing import ImagePreprocessor
from veritas.llm.runtime import explain
from veritas.runtime_config import GeminiSettings
from veritas.tts import synthesize

# Max payload size limit: 50MB
MAX_BODY_SIZE = 50 * 1024 * 1024
BASE_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("veritas.api")


def process_verification(before_path: str, after_path: str, *, enable_llm: bool = False,
                         enable_tts: bool = False) -> Dict[str, Any]:
    """Execute VERITAS deterministic verification pipeline on an image pair."""
    result = verify_evidence_pair(before_path, after_path)
    explanation = result.explanation_payload.to_dict()
    tts: Dict[str, Any] | None = None
    if enable_llm or enable_tts:
        explanation = explain(result.explanation_payload)
    if enable_tts:
        tts = synthesize(explanation["text"])

    # Extract SIFT, ORB, and AKAZE keypoint samples for visual inspection
    detector_keypoints: Dict[str, Any] = {
        "sift": {"before_count": 0, "after_count": 0, "before_keypoints": [], "after_keypoints": []},
        "orb": {"before_count": 0, "after_count": 0, "before_keypoints": [], "after_keypoints": []},
        "akaze": {"before_count": 0, "after_count": 0, "before_keypoints": [], "after_keypoints": []},
    }
    try:
        detector_instances = {"sift": SiftDetector(), "orb": OrbDetector(), "akaze": AkazeDetector()}
        preprocessor = ImagePreprocessor()
        src_proc, ref_proc = preprocessor.process(after_path), preprocessor.process(before_path)

        for name, det in detector_instances.items():
            try:
                ref_feats, _ = det.detect(ref_proc.enhanced)
                src_feats, _ = det.detect(src_proc.enhanced)
                ref_h, ref_w = ref_proc.source_shape[:2] if hasattr(ref_proc, "source_shape") else (480, 640)
                src_h, src_w = src_proc.source_shape[:2] if hasattr(src_proc, "source_shape") else (480, 640)
                detector_keypoints[name] = {
                    "before_count": len(ref_feats),
                    "after_count": len(src_feats),
                    "before_width": ref_w,
                    "before_height": ref_h,
                    "after_width": src_w,
                    "after_height": src_h,
                    "before_keypoints": [[round(f.x, 1), round(f.y, 1), round(f.scale, 1)] for f in ref_feats[:300]],
                    "after_keypoints": [[round(f.x, 1), round(f.y, 1), round(f.scale, 1)] for f in src_feats[:300]],
                }
            except Exception as det_err:
                logger.warning("Detector %s extraction error: %s", name, det_err)
    except Exception as proc_err:
        logger.warning("Preprocessing for detector keypoints skipped or failed: %s", proc_err)

    response = {
        "status": "success",
        "verdict": {
            "verdict": result.verdict.verdict.value,
            "rationale_codes": result.verdict.rationale_codes,
            "threshold_profile": getattr(result.verdict, "threshold_profile", "default"),
        },
        "gate": {
            "action": result.gate.action.value,
            "reasons": result.gate.reasons,
        },
        "partial_correspondence": {
            "status": result.partial_correspondence.status.value,
            "rationale": getattr(result.partial_correspondence, "rationale", "See regional correspondence evidence."),
            "confidence": getattr(result.partial_correspondence, "confidence", getattr(result.partial_correspondence, "coverage_fraction", 0.0)),
        },
        "audit": {
            "run_id": result.audit.run_id,
            "timestamp": getattr(result.audit, "timestamp", result.audit.provenance.get("timestamp") if hasattr(result.audit, "provenance") else None),
        },
        "detectors": detector_keypoints,
        "explanation": explanation,
        "summary": result.to_dict(),
    }
    if tts is not None:
        response["tts"] = tts
    return response


def runtime_health() -> Dict[str, Any]:
    """Report configuration only; this endpoint never exposes credentials."""
    gemini = GeminiSettings.from_env()
    return {
        "gemini": {"configured": gemini.configured, "reachable": None,
                   "model": gemini.model},
    }


class VeritasAPIHandler(BaseHTTPRequestHandler):
    """Multi-threaded HTTP request handler supporting CORS, static files, and verification."""

    def _set_cors_headers(self, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self) -> None:
        self._set_cors_headers(204)
        self.end_headers()

    def do_GET(self) -> None:
        clean_path = self.path.split("?")[0]

        if clean_path in ("/", "/api/health", "/health", "/api/runtime"):
            self._set_cors_headers(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = runtime_health() if clean_path == "/api/runtime" else {
                "service": "VERITAS Backend Verification Engine",
                "status": "healthy",
                "version": "1.0.0",
                "detectors": ["SIFT", "ORB", "AKAZE"],
                "gate_actions": ["ALLOW", "RESTRICT", "HUMAN_REVIEW", "BLOCK"],
            }
            self.wfile.write(json.dumps(response, indent=2).encode("utf-8"))
            return

        # Static file serving from pages/ and samples/
        candidate_file = None
        if clean_path.startswith("/pages/") or clean_path.startswith("/samples/"):
            candidate_file = (BASE_DIR / clean_path.lstrip("/")).resolve()
        elif clean_path in ("/page_zero.html", "/landing_page.html", "/upload_image.html", "/analysis_page.html", "/counsel_agent.html", "/veritas_store.js"):
            candidate_file = (BASE_DIR / "pages" / clean_path.lstrip("/")).resolve()

        if candidate_file and candidate_file.is_file():
            # Security sandbox: must be inside BASE_DIR
            try:
                candidate_file.relative_to(BASE_DIR)
            except ValueError:
                self._set_cors_headers(403)
                self.end_headers()
                self.wfile.write(b"Access Denied")
                return

            mime_type, _ = mimetypes.guess_type(str(candidate_file))
            if not mime_type:
                mime_type = "application/octet-stream"
            self._set_cors_headers(200)
            self.send_header("Content-Type", mime_type)
            self.end_headers()
            with open(candidate_file, "rb") as f:
                self.wfile.write(f.read())
            return

        self._set_cors_headers(404)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def do_POST(self) -> None:
        if self.path not in ("/api/verify", "/verify"):
            self._set_cors_headers(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))
            return

        content_type = self.headers.get("Content-Type", "")
        tmp_before = None
        tmp_after = None

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > MAX_BODY_SIZE:
                self._set_cors_headers(413)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Payload exceeds 50MB limit"}).encode("utf-8"))
                return

            body = self.rfile.read(content_length)

            if "application/json" in content_type:
                payload = json.loads(body.decode("utf-8"))

                before_path = payload.get("before_path")
                after_path = payload.get("after_path")
                enable_llm = bool(payload.get("enable_llm", False))
                enable_tts = bool(payload.get("enable_tts", False))

                # Support base64 image input for frontend clients
                before_b64 = payload.get("before_base64")
                after_b64 = payload.get("after_base64")

                if before_b64 and after_b64:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1:
                        tmp_before = f1.name
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2:
                        tmp_after = f2.name

                    with open(tmp_before, "wb") as f1:
                        f1.write(base64.b64decode(before_b64))
                    with open(tmp_after, "wb") as f2:
                        f2.write(base64.b64decode(after_b64))

                    before_path = tmp_before
                    after_path = tmp_after
                elif before_path and after_path:
                    # Sandboxing check for filesystem paths
                    bp = (BASE_DIR / before_path).resolve()
                    ap = (BASE_DIR / after_path).resolve()
                    if not bp.is_file() or not ap.is_file():
                        self._set_cors_headers(400)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": "Provided file paths could not be located"}).encode("utf-8"))
                        return
                    before_path = str(bp)
                    after_path = str(ap)

                if not before_path or not after_path:
                    self._set_cors_headers(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"error": "Provide either ('before_path', 'after_path') or ('before_base64', 'after_base64')"}).encode("utf-8")
                    )
                    return

            else:
                self._set_cors_headers(415)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": "Unsupported Content-Type. Please send application/json"}).encode("utf-8")
                )
                return

            result = process_verification(before_path, after_path, enable_llm=enable_llm, enable_tts=enable_tts)

            self._set_cors_headers(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result, indent=2).encode("utf-8"))

        except Exception as exc:
            logger.exception("Verification request error: %s", exc)
            self._set_cors_headers(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Internal verification engine error"}).encode("utf-8"))

        finally:
            if tmp_before and os.path.exists(tmp_before):
                try:
                    os.unlink(tmp_before)
                except OSError:
                    pass
            if tmp_after and os.path.exists(tmp_after):
                try:
                    os.unlink(tmp_after)
                except OSError:
                    pass


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, VeritasAPIHandler)
    print(f"VERITAS Backend API running on http://{host}:{port}")
    print(f"  - Health check: http://{host}:{port}/api/health")
    print(f"  - Verification: POST http://{host}:{port}/api/verify")
    print(f"  - Static Frontend: http://{host}:{port}/pages/page_zero.html")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping VERITAS API server...")
        httpd.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Start VERITAS Backend API server for frontend.")
    parser.add_argument("--host", default="127.0.0.1", help="Host binding address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())

