"""Provider-independent input contracts for future LLM/TTS adapters."""

from .explainer import build_explanation_payload
from .schema import ExplanationPayload

__all__ = ["ExplanationPayload", "build_explanation_payload"]
