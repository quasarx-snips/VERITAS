"""Local runtime configuration for optional explanation providers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_local_env(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE entries without replacing exported environment values."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class GroqSettings:
    api_key: str = ""
    model: str = "openai/gpt-oss-120b"
    timeout_seconds: float = 60.0
    temperature: float = 0.7
    max_tokens: int = 8192
    chat_cooldown_seconds: float = 5.0

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)

    @classmethod
    def from_env(cls) -> "GroqSettings":
        load_local_env()
        return cls(
            api_key=os.getenv("GROQ_API_KEY", ""),
            model=os.getenv("VERITAS_GROQ_MODEL", os.getenv("GROQ_MODEL", cls.model)),
            timeout_seconds=float(os.getenv("VERITAS_LLM_TIMEOUT_SECONDS", "60")),
            temperature=float(os.getenv("VERITAS_LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("VERITAS_LLM_MAX_TOKENS", "8192")),
            chat_cooldown_seconds=float(os.getenv("VERITAS_CHAT_COOLDOWN_SECONDS", "5")),
        )


@dataclass(frozen=True)
class GeminiSettings:
    api_key: str = ""
    # The stable alias follows the currently available Flash-Lite release.
    model: str = "gemini-flash-lite-latest"
    timeout_seconds: float = 60.0
    temperature: float = 0.7
    max_tokens: int = 8192
    tts_model: str = "gemini-2.5-flash-preview-tts"
    tts_voice: str = "Puck"
    tts_timeout_seconds: float = 60.0
    tts_speed: float = 1.25
    chat_cooldown_seconds: float = 5.0

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)

    @classmethod
    def from_env(cls) -> "GeminiSettings":
        load_local_env()
        return cls(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model=os.getenv("VERITAS_GEMINI_MODEL", cls.model),
            timeout_seconds=float(os.getenv("VERITAS_LLM_TIMEOUT_SECONDS", "60")),
            temperature=float(os.getenv("VERITAS_LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("VERITAS_LLM_MAX_TOKENS", "8192")),
            tts_model=os.getenv("VERITAS_GEMINI_TTS_MODEL", cls.tts_model),
            tts_voice=os.getenv("VERITAS_GEMINI_TTS_VOICE", cls.tts_voice),
            tts_timeout_seconds=float(os.getenv("VERITAS_TTS_TIMEOUT_SECONDS", "60")),
            tts_speed=float(os.getenv("VERITAS_TTS_SPEED", "1.25")),
            chat_cooldown_seconds=float(os.getenv("VERITAS_CHAT_COOLDOWN_SECONDS", "5")),
        )
