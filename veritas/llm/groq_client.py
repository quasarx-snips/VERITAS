"""Groq API adapter for ultra-fast, high-throughput VERITAS conversational reasoning."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from veritas.llm.gemini_client import LLMRuntimeResult
from veritas.runtime_config import GroqSettings


class GroqClient:
    """Zero-dependency Groq REST API client for fast forensic explanation and multi-turn chat."""

    def __init__(self, settings: GroqSettings | None = None) -> None:
        self.settings = settings or GroqSettings.from_env()

    @property
    def configured(self) -> bool:
        return bool(self.settings.api_key and self.settings.model)

    def complete(self, system: str, user: str) -> LLMRuntimeResult:
        """Single-turn completion."""
        return self.chat(system, [{"role": "user", "content": user}])

    def chat(self, system: str, messages: list[dict[str, str]]) -> LLMRuntimeResult:
        """Multi-turn conversation with Groq OpenAI-compatible chat completion API."""
        if not self.configured:
            return LLMRuntimeResult("groq", self.settings.model, False, error="Groq API key is not configured")

        groq_messages = []
        if system:
            groq_messages.append({"role": "system", "content": system})

        for msg in messages:
            role = "user" if msg.get("role") == "user" else "assistant"
            groq_messages.append({"role": role, "content": msg.get("content", "")})

        # List of candidate models to try in order
        candidate_models = [self.settings.model]
        for fallback in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "groq/compound"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        endpoint = "https://api.groq.com/openai/v1/chat/completions"
        started = time.perf_counter()
        last_error = ""

        for model_name in candidate_models:
            body = {
                "model": model_name,
                "messages": groq_messages,
                "temperature": self.settings.temperature,
                "max_tokens": self.settings.max_tokens,
            }

            request = Request(
                endpoint,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.settings.api_key.strip()}",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) VERITAS-Console/1.0",
                },
                method="POST",
            )

            try:
                with urlopen(request, timeout=self.settings.timeout_seconds) as response:  # nosec B310
                    payload: Mapping[str, Any] = json.loads(response.read().decode("utf-8"))

                choices = payload.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    if text and text.strip():
                        latency_ms = round((time.perf_counter() - started) * 1000, 2)
                        return LLMRuntimeResult("groq", model_name, True, text.strip(), latency_ms=latency_ms)
            except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
                error_msg = str(exc)
                if isinstance(exc, HTTPError):
                    try:
                        error_detail = exc.read().decode("utf-8")
                        parsed = json.loads(error_detail)
                        error_msg = parsed.get("error", {}).get("message", error_msg)
                    except Exception:
                        pass
                last_error = error_msg
                # If it's a model not found / unauthorized error, continue to next candidate model
                if "does not exist" in error_msg.lower() or "not have access" in error_msg.lower() or "model_not_found" in error_msg.lower():
                    continue
                break

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return LLMRuntimeResult("groq", self.settings.model, False, latency_ms=latency_ms, error=last_error or "Groq generation failed")
