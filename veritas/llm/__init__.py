"""Zero-pixel deterministic explanation payloads and optional runtime."""

from .explainer import build_explanation_payload
from .schema import ExplanationPayload
from .runtime import explain

__all__ = ["ExplanationPayload", "build_explanation_payload", "explain"]
