"""Optional live runtime smoke test. It never reports a provider PASS unless contacted."""

from __future__ import annotations

from veritas.llm.runtime import SYSTEM_PROMPT
from veritas.llm.gemini_client import GeminiClient


def main() -> int:
    gemini = GeminiClient()
    if not gemini.configured:
        print("Gemini: NOT CONFIGURED")
    else:
        result = gemini.complete(SYSTEM_PROMPT, "Reply with one concise explanation sentence.")
        print("Gemini:", "PASS" if result.success else f"FAIL ({result.error})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
