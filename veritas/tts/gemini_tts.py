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


def time_stretch(pcm_bytes: bytes, rate: float = 1.25) -> bytes:
    """Time-stretch 16-bit mono PCM audio using WSOLA to preserve natural pitch."""
    if abs(rate - 1.0) < 1e-3:
        return pcm_bytes
    try:
        import numpy as np
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        n_in = len(samples)
        n_out = int(n_in / rate)
        win_size = 1024
        hop = 256
        search_range = 128
        out = np.zeros(n_out + win_size, dtype=np.float32)
        norm = np.zeros(n_out + win_size, dtype=np.float32)
        window = np.hanning(win_size).astype(np.float32)

        t_out = 0
        t_in = 0
        while t_out + win_size < n_out and t_in + win_size + search_range < n_in:
            if t_out == 0:
                offset = 0
            else:
                target = out[t_out : t_out + win_size // 2]
                candidates = samples[t_in : t_in + win_size // 2 + search_range]
                corrs = np.correlate(candidates, target, mode="valid")
                offset = int(np.argmax(corrs)) if len(corrs) > 0 else 0

            pos = t_in + offset
            out[t_out : t_out + win_size] += samples[pos : pos + win_size] * window
            norm[t_out : t_out + win_size] += window
            t_out += hop
            t_in = int(t_out * rate)

        mask = norm > 1e-4
        out[mask] /= norm[mask]
        clipped = np.clip(out[:n_out], -32768, 32767).astype(np.int16)
        return clipped.tobytes()
    except Exception:
        return pcm_bytes


def synthesize(text: str, settings: GeminiSettings | None = None,
               output_dir: str | Path = "outputs/audio",
               speed: float | None = None) -> dict[str, Any]:
    settings = settings or GeminiSettings.from_env()
    playback_speed = speed if speed is not None else getattr(settings, "tts_speed", 1.25)
    result: dict[str, Any] = {"enabled": True, "provider": "gemini", "model": settings.tts_model,
                              "voice": settings.tts_voice, "speed": playback_speed,
                              "success": False, "audio_format": "wav",
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
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urlopen(request, timeout=settings.tts_timeout_seconds) as response:  # nosec B310: fixed Gemini API host
                payload: Mapping[str, Any] = json.loads(response.read().decode("utf-8"))
            inline = payload["candidates"][0]["content"]["parts"][0]["inlineData"]
            pcm = base64.b64decode(inline["data"])
            if not pcm:
                raise ValueError("Gemini returned empty audio")
            if abs(playback_speed - 1.0) > 1e-3:
                pcm = time_stretch(pcm, playback_speed)
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
            break
        except HTTPError as exc:
            if exc.code == 429 and attempt < max_retries - 1:
                time.sleep(2.5 * (attempt + 1))
                continue
            # Fall back to local offline synthesis on rate limit / error
            offline = synthesize_local_offline(text, output_dir=output_dir, speed=playback_speed)
            if offline.get("success"):
                result.update(success=True, audio_path=offline["audio_path"], provider="gemini_fallback_offline",
                              latency_ms=round((time.perf_counter() - started) * 1000, 2), error=None)
            else:
                result.update(error=str(exc), latency_ms=round((time.perf_counter() - started) * 1000, 2))
            break
        except (URLError, TimeoutError, ValueError, KeyError, IndexError, json.JSONDecodeError, OSError) as exc:
            offline = synthesize_local_offline(text, output_dir=output_dir, speed=playback_speed)
            if offline.get("success"):
                result.update(success=True, audio_path=offline["audio_path"], provider="gemini_fallback_offline",
                              latency_ms=round((time.perf_counter() - started) * 1000, 2), error=None)
            else:
                result.update(error=str(exc), latency_ms=round((time.perf_counter() - started) * 1000, 2))
            break
    return result


def synthesize_local_offline(text: str, output_dir: str | Path = "outputs/audio", speed: float = 1.25) -> dict[str, Any]:
    """Fallback offline speech synthesizer using Windows native System.Speech at custom speed."""
    import subprocess
    rate_val = int((speed - 1.0) * 8)
    rate_val = max(-10, min(10, rate_val))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"veritas-{uuid4().hex}.wav"
    clean_text = text.replace('"', ' ').replace("'", " ").replace("`", " ").strip()
    ps_script = f"""
Add-Type -AssemblyName System.Speech
$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speak.Rate = {rate_val}
$speak.SetOutputToWaveFile("{destination.as_posix()}")
$speak.Speak("{clean_text}")
$speak.Dispose()
"""
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                       capture_output=True, text=True, timeout=15)
        if destination.exists() and destination.stat().st_size > 500:
            return {
                "enabled": True, "provider": "local_offline", "model": "windows_sapi",
                "voice": "Default", "speed": speed, "success": True, "audio_format": "wav",
                "audio_path": str(destination), "latency_ms": 120.0, "error": None,
            }
    except Exception as exc:
        return {"enabled": True, "provider": "local_offline", "success": False, "error": str(exc)}
    return {"enabled": True, "provider": "local_offline", "success": False, "error": "Offline synthesis produced empty file"}

