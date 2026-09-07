"""Lightweight REST API bridge connecting VERITAS backend to frontend clients.

Provides a zero-dependency HTTP server with CORS support and JSON endpoints.

Usage:
    python -m veritas.api --port 8000
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict

from veritas.pipeline import verify_evidence_pair
from veritas.llm.runtime import explain
from veritas.runtime_config import GeminiSettings
from veritas.tts import synthesize


def process_verification(before_path: str, after_path: str, *, enable_llm: bool = False,
                         enable_tts: bool = False) -> Dict[str, Any]:
    """Execute VERITAS deterministic verification pipeline on an image pair."""
    result = verify_evidence_pair(before_path, after_path)
    explanation = result.explanation_payload.to_dict()
    tts: Dict[str, Any] | None = None
    if enable_llm or enable_tts:
        # Voice mode always requests the human-readable explanation first; both
        # providers remain strictly downstream of the immutable pipeline result.
        explanation = explain(result.explanation_payload)
    if enable_tts:
        tts = synthesize(explanation["text"])
    response = {
        "status": "success",
        "verdict": {
            "verdict": result.verdict.verdict.value,
            "rationale_codes": result.verdict.rationale_codes,
            # Older callers expect this label although VerdictResult no longer
            # stores it as a field.
            "threshold_profile": getattr(result.verdict, "threshold_profile", "default"),
        },
        "gate": {
            "action": result.gate.action.value,
            "reasons": result.gate.reasons,
        },
        "partial_correspondence": {
            "status": result.partial_correspondence.status.value,
            # Keep the legacy frontend fields while the canonical object exposes
            # coverage and per-region evidence instead of a confidence claim.
            "rationale": getattr(result.partial_correspondence, "rationale", "See regional correspondence evidence."),
            "confidence": getattr(result.partial_correspondence, "confidence", getattr(result.partial_correspondence, "coverage_fraction", 0.0)),
        },
        "audit": {
            "run_id": result.audit.run_id,
            "timestamp": getattr(result.audit, "timestamp", result.audit.provenance.get("timestamp") if hasattr(result.audit, "provenance") else None),
        },
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
    """Zero-dependency HTTP request handler supporting CORS and JSON responses."""

    def _set_cors_headers(self, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self) -> None:
        self._set_cors_headers(204)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in ("/", "/api/health", "/health", "/api/runtime"):
            self._set_cors_headers(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = runtime_health() if self.path == "/api/runtime" else {
                "service": "VERITAS Backend Verification Engine",
                "status": "healthy",
                "version": "1.0.0",
                "detectors": ["SIFT", "ORB", "AKAZE"],
                "gate_actions": ["ALLOW", "RESTRICT", "HUMAN_REVIEW", "BLOCK"],
            }
            self.wfile.write(json.dumps(response, indent=2).encode("utf-8"))
        else:
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
                        f1.write(base64.b64decode(before_b64))
                        tmp_before = f1.name
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2:
                        f2.write(base64.b64decode(after_b64))
                        tmp_after = f2.name
                    before_path = tmp_before
                    after_path = tmp_after

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
            self._set_cors_headers(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))

        finally:
            if tmp_before and os.path.exists(tmp_before):
                os.unlink(tmp_before)
            if tmp_after and os.path.exists(tmp_after):
                os.unlink(tmp_after)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    server_address = (host, port)
    httpd = HTTPServer(server_address, VeritasAPIHandler)
    print(f"VERITAS Backend API running on http://{host}:{port}")
    print(f"  - Health check: http://localhost:{port}/api/health")
    print(f"  - Verification: POST http://localhost:{port}/api/verify")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping VERITAS API server...")
        httpd.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Start VERITAS Backend API server for frontend.")
    parser.add_argument("--host", default="0.0.0.0", help="Host binding address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
