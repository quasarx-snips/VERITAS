"""Safe explanation runtime; it consumes evidence but never alters decisions."""

from __future__ import annotations

import json
from typing import Any

from .schema import ExplanationPayload
from .gemini_client import GeminiClient, LLMRuntimeResult

SYSTEM_PROMPT = """You are the VERITAS explanation layer. You explain deterministic verification results. You do not make or modify decisions. Use only supplied structured evidence. Do not invent measurements, detectors, probabilities, ground truth, semantic events, or claims of disaster detection. Do not output JSON, a verdict, or a gate action. State limitations concisely."""


def deterministic_explanation(payload: ExplanationPayload) -> str:
    reasons = " ".join(payload.key_reasons)
    limitations = " ".join(payload.limitations or payload.warnings)
    return f"{payload.headline} {reasons} Recommended action: {payload.recommended_next_action} Limitations: {limitations}".strip()


def build_prompt(payload: ExplanationPayload) -> str:
    """Serialize only the pre-built zero-pixel contract."""
    return "Explain this VERITAS payload in concise plain language:\n" + json.dumps(payload.to_dict(), sort_keys=True, default=str)


def explain(payload: ExplanationPayload, client: GeminiClient | None = None) -> dict[str, Any]:
    fallback = deterministic_explanation(payload)
    runtime: LLMRuntimeResult = (client or GeminiClient()).complete(SYSTEM_PROMPT, build_prompt(payload))
    text = runtime.text if runtime.success else fallback
    return {"source": "gemini" if runtime.success else "deterministic", "text": text,
            "structured": payload.to_dict(), "runtime": runtime.to_dict()}
