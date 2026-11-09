import json
import types
from unittest.mock import patch

from veritas.llm.runtime import build_prompt, explain
from veritas.llm.schema import ExplanationPayload
from veritas.llm.gemini_client import GeminiClient, LLMRuntimeResult
from veritas.runtime_config import GeminiSettings
from veritas import api


def payload():
    return ExplanationPayload("test", "explain", "VERITAS verdict: WEAK; gate: HUMAN_REVIEW.", "WEAK", "HUMAN_REVIEW",
                              ["Spatial support is limited."], {}, {}, {}, ["Review required."],
                              "Route this pair to a human reviewer before downstream action.", {}, ["Evidence is limited."], [])


def test_prompt_contains_structured_evidence_and_no_pixels():
    prompt = build_prompt(payload())
    assert "WEAK" in prompt and "HUMAN_REVIEW" in prompt
    assert "image_data" not in prompt and "base64" not in prompt


def test_gemini_unconfigured_and_fallback_preserve_authority():
    result = explain(payload(), GeminiClient(GeminiSettings(api_key="")))
    assert result["source"] == "deterministic"
    assert result["structured"]["verdict"] == "WEAK"
    assert result["structured"]["gate_action"] == "HUMAN_REVIEW"


def test_gemini_bad_response_is_failure():
    settings = GeminiSettings(api_key="test-key", timeout_seconds=1)
    with patch("veritas.llm.gemini_client.urlopen") as open_url:
        response = open_url.return_value.__enter__.return_value
        response.read.return_value = json.dumps({"candidates": []}).encode()
        result = GeminiClient(settings).complete("system", "user")
    assert not result.success and result.error


def test_gemini_response_is_parsed():
    settings = GeminiSettings(api_key="test-key", model="gemini-2.5-flash-lite")
    with patch("veritas.llm.gemini_client.urlopen") as open_url:
        response = open_url.return_value.__enter__.return_value
        response.read.return_value = json.dumps({"candidates": [{"content": {"parts": [{"text": "Safe explanation."}]}}]}).encode()
        result = GeminiClient(settings).complete("system", "user")
    assert result.success and result.provider == "gemini" and result.text == "Safe explanation."


def test_llm_text_cannot_change_structured_verdict_or_gate():
    class Client:
        def complete(self, *_):
            return LLMRuntimeResult("gemini", "test", True, "Verdict: STRONG. Gate: ALLOW.")
    result = explain(payload(), Client())
    assert result["text"] == "Verdict: STRONG. Gate: ALLOW."
    assert result["structured"]["verdict"] == "WEAK"
    assert result["structured"]["gate_action"] == "HUMAN_REVIEW"


def test_api_opt_in_runtime_does_not_change_deterministic_fields():
    class Value:
        def __init__(self, value): self.value = value
    class Result:
        explanation_payload = payload()
        verdict = types.SimpleNamespace(verdict=Value("WEAK"), rationale_codes=["LOW_SPATIAL_COVERAGE"], threshold_profile="default")
        gate = types.SimpleNamespace(action=Value("HUMAN_REVIEW"), reasons=["LOW_SPATIAL_COVERAGE"])
        partial_correspondence = types.SimpleNamespace(status=Value("NONE"), rationale="none", confidence=0.0)
        audit = types.SimpleNamespace(run_id="run", timestamp="now")
        def to_dict(self): return {"verdict": {"verdict": "WEAK"}, "gate": {"action": "HUMAN_REVIEW"}}
    with patch("veritas.api.verify_evidence_pair", return_value=Result()), \
         patch("veritas.api.explain", return_value={"source": "deterministic", "text": "fallback", "structured": payload().to_dict(), "runtime": {}}), \
         patch("veritas.api.synthesize", return_value={"enabled": True, "success": False, "error": "mocked"}):
        response = api.process_verification("before", "after", enable_llm=True, enable_tts=True)
    assert response["verdict"]["verdict"] == "WEAK"
    assert response["gate"]["action"] == "HUMAN_REVIEW"
    assert response["explanation"]["source"] == "deterministic"
    assert response["tts"]["enabled"] is True


def test_gemini_client_chat():
    settings = GeminiSettings(api_key="test-key", model="gemini-2.5-flash-lite")
    with patch("veritas.llm.gemini_client.urlopen") as open_url:
        response = open_url.return_value.__enter__.return_value
        response.read.return_value = json.dumps({"candidates": [{"content": {"parts": [{"text": "Chat answer."}]}}]}).encode()
        result = GeminiClient(settings).chat("system instruction", [{"role": "user", "content": "hello"}])
    assert result.success is True
    assert result.text == "Chat answer."


def test_chat_domain_guardrail_rejects_unrelated_questions_without_calling_llm():
    class Client:
        def chat(self, *_):
            raise AssertionError("Out-of-scope requests must not reach the LLM")

    result = api.process_chat(
        [{"role": "user", "content": "Write a recipe for chocolate cake."}], client=Client()
    )

    assert result["source"] == "guardrail"
    assert result["text"] == api.OUT_OF_SCOPE_REPLY


def test_chat_domain_guardrail_allows_geospatial_imagery_questions():
    class Client:
        def chat(self, *_):
            return LLMRuntimeResult("test", "test-model", True, "NDVI is useful for vegetation analysis.")

    result = api.process_chat(
        [{"role": "user", "content": "How is NDVI calculated from satellite imagery?"}], client=Client()
    )

    assert result["source"] == "test"
    assert "NDVI" in result["text"]


def test_groq_client_complete_and_chat():
    from veritas.llm.groq_client import GroqClient
    from veritas.runtime_config import GroqSettings

    # Unconfigured case
    unconfigured = GroqClient(GroqSettings(api_key=""))
    res_unconf = unconfigured.complete("sys", "user")
    assert res_unconf.success is False
    assert "not configured" in res_unconf.error

    # Configured mock case
    settings = GroqSettings(api_key="gsk_mock_key_test", model="llama-3.3-70b-versatile")
    client = GroqClient(settings)
    mock_body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Groq verified evidence: Affine certificate valid."
                }
            }
        ]
    }
    with patch("veritas.llm.groq_client.urlopen") as open_url:
        response = open_url.return_value.__enter__.return_value
        response.read.return_value = json.dumps(mock_body).encode("utf-8")
        result = client.complete("system", "user")

    assert result.success is True
    assert result.provider == "groq"
    assert result.model == "llama-3.3-70b-versatile"
    assert "Groq verified" in result.text

