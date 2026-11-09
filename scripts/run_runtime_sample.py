"""Run the bundled sample pair through VERITAS, Gemini explanation, and Gemini TTS."""

from __future__ import annotations

import json
from pathlib import Path

from veritas.api import process_verification


def main() -> int:
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    result = process_verification("samples/sample_before.png", "samples/sample_after.jpg",
                                  enable_llm=True, enable_tts=True)
    destination = output / "sample_runtime_result.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Result: {destination}")
    print(f"Explanation: {result['explanation']['source']}")
    print(f"TTS: {'PASS' if result['tts']['success'] else 'FAIL'}")
    if result["tts"]["audio_path"]:
        print(f"Audio: {result['tts']['audio_path']}")
    return 0 if result["tts"]["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
