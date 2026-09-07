"""Gemini TTS adapter that persists returned PCM audio as a WAV artifact."""

from __future__ import annotations

import base64
import json
import time
import wave
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

from veritas.runtime_config import GeminiSettings


def synthesize(text: str, settings: GeminiSettings | None = None,
               output_dir: str | Path = "outputs/audio") -> dict[str, Any]:
    settings = settings or GeminiSettings.from_env()
    result: dict[str, Any] = {"enabled": True, "provider": "gemini", "model": settings.tts_model,
                              "voice": settings.tts_voice, "success": False, "audio_format": "wav",
                              "audio_path": None, "latency_ms": None, "error": None}
    if not settings.configured:
        result["error"] = "Gemini API key is not configured"
        return result
    if not text.strip():
        result["error"] = "No text provided for synthesis"
        return result
    body = {"contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": settings.tts_voice}}}}}
    endpoint = ("https://generativelanguage.googleapis.com/v1beta/models/"
                f"{quote(settings.tts_model, safe='')}:generateContent?key={quote(settings.api_key, safe='')}")
    request = Request(endpoint, data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=settings.tts_timeout_seconds) as response:  # nosec B310: fixed Gemini API host
            payload: Mapping[str, Any] = json.loads(response.read().decode("utf-8"))
        inline = payload["candidates"][0]["content"]["parts"][0]["inlineData"]
        pcm = base64.b64decode(inline["data"])
        if not pcm:
            raise ValueError("Gemini returned empty audio")
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"veritas-{uuid4().hex}.wav"
        with wave.open(str(destination), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(pcm)
        result.update(success=True, audio_path=str(destination),
                      latency_ms=round((time.perf_counter() - started) * 1000, 2))
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError, json.JSONDecodeError, OSError) as exc:
        result.update(error=str(exc), latency_ms=round((time.perf_counter() - started) * 1000, 2))
    return result
