"""Run the complete deterministic VERITAS backend for one image pair.

Features:
  - Generates comprehensive metrics text report (<run_id>.metrics.txt and latest_metrics.txt)
  - Exports complete audit JSON (<run_id>.json) and explanation JSON (<run_id>.explanation.json)
  - Synthesizes audio explanation to outputs/audio/
  - Interactive CLI chat loop with Gemini LLM to answer questions about evidence and metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from veritas.pipeline import verify_evidence_pair, VeritasPipelineResult
from veritas.llm.runtime import explain
from veritas.llm.gemini_client import GeminiClient
from veritas.tts import synthesize


def format_metrics_report(result: VeritasPipelineResult, before_path: str, after_path: str,
                           explanation: Dict[str, Any], audio_result: Dict[str, Any] | None) -> str:
    """Format all verification metrics into a clean, human-readable text document."""
    lines = []
    w = 80
    lines.append("=" * w)
    lines.append("VERITAS VERIFICATION METRICS REPORT".center(w))
    lines.append("=" * w)

    # 1. Metadata
    lines.append("\n[1] EXECUTION METADATA")
    lines.append("-" * w)
    lines.append(f"Run ID:             {result.audit.run_id}")
    provenance = getattr(result.audit, "provenance", {})
    timestamp = getattr(result.audit, "timestamp", provenance.get("timestamp") if isinstance(provenance, dict) else "N/A")
    lines.append(f"Timestamp:          {timestamp}")
    lines.append(f"Source (Before):    {before_path}")
    lines.append(f"Reference (After):  {after_path}")

    # 2. Decision Summary
    lines.append("\n[2] DECISION & GATING SUMMARY")
    lines.append("-" * w)
    lines.append(f"Verdict:            {result.verdict.verdict.value}")
    lines.append(f"Verdict Rationales: {', '.join(result.verdict.rationale_codes) if result.verdict.rationale_codes else 'None'}")
    lines.append(f"Threshold Profile:  {getattr(result.verdict, 'threshold_profile', 'default')}")
    lines.append(f"Safety Gate Action: {result.gate.action.value}")
    lines.append(f"Gate Reasons:       {', '.join(result.gate.reasons) if result.gate.reasons else 'None'}")
    lines.append(f"Gate Policy:        {getattr(result.gate, 'policy', 'default')}")
    lines.append(f"Partial Status:     {result.partial_correspondence.status.value}")
    lines.append(f"Partial Coverage:   {getattr(result.partial_correspondence, 'coverage_fraction', 0.0) * 100:.1f}%")
    lines.append(f"Partial Rationale:  {getattr(result.partial_correspondence, 'rationale', 'None')}")

    # 3. Multi-Detector Evidence & Quorum
    lines.append("\n[3] DETECTOR & MATCHING EVIDENCE")
    lines.append("-" * w)
    quorum = result.evidence.quorum
    lines.append(f"Quorum Strength:    {quorum.quorum_strength:.2f}")
    lines.append(f"Supporting:         {', '.join(quorum.supporting_detectors) if quorum.supporting_detectors else 'None'}")
    lines.append(f"Disagreeing:        {', '.join(quorum.disagreeing_detectors) if quorum.disagreeing_detectors else 'None'}")
    lines.append(f"Available:          {', '.join(quorum.available_detectors) if quorum.available_detectors else 'None'}")

    lines.append("\nPer-Detector Breakdown:")
    for det_name in ("sift", "orb", "akaze"):
        det_ev = result.evidence.features.get(det_name)
        match_ev = result.evidence.matches.get(det_name)
        if det_ev and match_ev:
            ret_pct = (match_ev.filtered_count / match_ev.candidate_count * 100) if match_ev.candidate_count > 0 else 0.0
            lines.append(
                f"  - {det_name.upper():<6} | Keypoints: {det_ev.keypoint_count} | "
                f"Matches: {match_ev.candidate_count} raw -> {match_ev.filtered_count} filtered ({ret_pct:.1f}%)"
            )

    # 4. Geometric Affine Certificate & Residuals
    lines.append("\n[4] GEOMETRIC AFFINE CERTIFICATE & RESIDUALS")
    lines.append("-" * w)
    geom = result.evidence.geometry
    lines.append(f"Model:              {geom.model}")
    lines.append(f"Certified:          {geom.certified}")
    lines.append(f"Inliers:            {geom.inlier_count} / {geom.candidate_count} ({geom.inlier_ratio * 100:.1f}%)")
    if geom.affine_matrix:
        lines.append(f"Affine Matrix:      {geom.affine_matrix}")

    counter = result.evidence.counter_evidence
    res = counter.residuals
    mean_err = res.get("mean") or geom.residuals.get("mean") or 0.0
    max_err = res.get("maximum") or geom.residuals.get("maximum") or 0.0
    rms_err = geom.residuals.get("rmse") or 0.0
    lines.append(f"Reprojection Error: Mean = {mean_err:.3f} px | Max = {max_err:.3f} px | RMS = {rms_err:.3f} px")
    exceed = res.get("threshold_exceedance_count")
    if exceed is not None:
        lines.append(f"High Residuals:     {exceed} matches above threshold")

    # 5. Spatial Distribution
    lines.append("\n[5] SPATIAL COVERAGE & ENTROPY")
    lines.append("-" * w)
    spatial = result.evidence.spatial
    total_cells = spatial.grid_rows * spatial.grid_columns
    lines.append(f"Spatial Grid:       {spatial.grid_rows} x {spatial.grid_columns} ({total_cells} cells)")
    lines.append(f"Occupied Cells:     {spatial.occupied_cells} / {total_cells}")
    lines.append(f"Coverage Ratio:     {spatial.coverage_ratio * 100:.1f}%")
    entropy_str = f"{spatial.normalized_entropy:.3f}" if spatial.normalized_entropy is not None else "N/A"
    lines.append(f"Spatial Entropy:    {entropy_str}")

    disagree = counter.feature_disagreement
    bidir = disagree.get("bidirectional_coverage", {})
    if bidir:
        lines.append(f"Bidirectional Cov:  Source = {bidir.get('source', 0.0) * 100:.1f}% | Ref = {bidir.get('reference', 0.0) * 100:.1f}% | Conservative = {bidir.get('conservative', 0.0) * 100:.1f}%")

    # 6. Audio & AI Explanation
    lines.append("\n[6] EXPLANATION & AUDIO")
    lines.append("-" * w)
    lines.append(f"Explanation Source: {explanation.get('source', 'deterministic')}")
    if audio_result and audio_result.get("success"):
        lines.append(f"Audio File:         {audio_result.get('audio_path')} ({audio_result.get('audio_format')}, {audio_result.get('latency_ms')} ms)")
    else:
        err = audio_result.get("error") if audio_result else "Disabled"
        lines.append(f"Audio File:         Not generated ({err})")

    lines.append("\nExplanation Text:")
    text_content = explanation.get("text", "")
    for paragraph in text_content.splitlines():
        lines.append(f"  {paragraph}")

    lines.append("\n" + "=" * w)
    lines.append("END OF VERITAS METRICS REPORT".center(w))
    lines.append("=" * w)
    return "\n".join(lines)


def generate_spoken_narration(result: VeritasPipelineResult, explanation_text: str, client: GeminiClient | None = None) -> str:
    """Generate an articulate, engaging, conversational narration specifically crafted for audio playback.

    The tone is that of a friendly, knowledgeable engineer who makes technical concepts easy to grasp.
    Contains zero markdown artifacts (no asterisks, bullet marks, headers) so it sounds completely natural.
    """
    client = client or GeminiClient()
    if not client.configured:
        clean = explanation_text.replace("*", "").replace("#", "").replace("`", "").strip()
        return clean

    evidence_dict = result.explanation_payload.to_dict()
    prompt = (
        "You are an engaging, friendly computer vision engineer and educator speaking to a teammate "
        "to help them intuitively understand this verification result.\n\n"
        "Guidelines for spoken audio:\n"
        "1. Persona: You are a sharp, articulate, relatable guy who explains technical stuff simply and clearly.\n"
        "2. Format: Output ONLY natural spoken prose. NEVER use markdown, bullet points, asterisks, hashes, or numbered lists.\n"
        "3. Narrative flow:\n"
        "   - Give a warm, natural opener with the verdict and what the gate action means in plain terms.\n"
        "   - Walk through the detectors (SIFT, ORB, AKAZE) and explain their agreement.\n"
        "   - Explain the inlier ratio and geometric alignment intuitively.\n"
        "   - Clarify the spatial breakdown (which regions matched well and which specific cells didn't).\n"
        "   - Finish with a clear, practical recommendation for what to do next.\n"
        "4. Length: 100 to 140 words (around 45 to 60 seconds spoken).\n\n"
        f"Verification Evidence:\n{json.dumps(evidence_dict, default=str)}"
    )
    res = client.complete("You are an expert audio narrator and computer vision educator.", prompt)
    if res.success and res.text:
        clean_spoken = res.text.replace("*", "").replace("#", "").replace("`", "").strip()
        return clean_spoken
    return explanation_text.replace("*", "").replace("#", "").replace("`", "").strip()


def start_interactive_chat(result: VeritasPipelineResult, explanation_text: str) -> None:
    """Run an interactive conversation loop in the terminal allowing user to query the LLM with strict railguards."""
    client = GeminiClient()
    if not client.configured:
        print("\n[Chat Note] Gemini API key not configured in .env. Interactive chat is disabled.")
        return

    # Feed complete, comprehensive evidence
    evidence_json = json.dumps(result.to_dict(), indent=2, default=str)
    system_prompt = (
        "You are the VERITAS Verification Assistant. You are conducting an interactive session with an engineer "
        "reviewing the correspondence analysis of this specific image pair.\n\n"
        "STRICT SCOPE GUARDRAILS (CRITICAL):\n"
        "1. ALLOWED TOPICS: You may ONLY discuss:\n"
        "   - The provided image pair, its verification results, verdict, and safety gate actions.\n"
        "   - The computer vision metrics, detectors (SIFT, ORB, AKAZE), inliers, inlier ratios, candidate matches, and residuals.\n"
        "   - Geometric transformations (affine certificates, 2x3 matrix, translation, rotation, scale).\n"
        "   - Spatial correspondence (4x4 grid cells, cell occupancy, spatial entropy, supported vs unsupported regions).\n"
        "   - Computer vision principles and math directly relevant to interpreting this evidence.\n"
        "2. FORBIDDEN TOPICS & MANDATORY REFUSAL:\n"
        "   - If the user asks ANY question outside of this scope (including math puzzles like '1+2', trivia like 'capital of France', general coding, general world knowledge, weather, philosophy, or casual chit-chat):\n"
        "     YOU MUST STRICTLY REFUSE TO ANSWER.\n"
        "     DO NOT answer the math question or trivia question.\n"
        "     State politely that you are strictly scoped to this image pair's verification results, and suggest a valid verification question they can ask (e.g. asking about the inlier ratio, the reprojection residuals, or why column 0 was marked unsupported).\n"
        "3. GROUNDING:\n"
        "   - All your answers about this pair must be strictly grounded in the evidence provided below. Do not invent measurements or ground truth claims.\n\n"
        f"VERITAS Complete Verification Evidence and Metrics:\n{evidence_json}\n\n"
        f"Initial Plain-Language Explanation:\n{explanation_text}"
    )

    print("\n" + "=" * 80)
    print("💬  VERITAS INTERACTIVE CHAT (Gemini LLM)")
    print("=" * 80)
    print("You can ask questions about the metrics, evidence, verdict, or gate action.")
    print("Commands: Type 'exit', 'quit', or 'q' to end the chat session.")
    print("-" * 80)

    history: list[dict[str, str]] = []
    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat session.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Session ended.")
            break

        history.append({"role": "user", "content": user_input})
        print("VERITAS is thinking...", end="\r", flush=True)
        resp = client.chat(system_prompt, history)
        # Clear "thinking" line
        print(" " * 30, end="\r", flush=True)

        if resp.success:
            print(f"VERITAS > {resp.text}")
            history.append({"role": "model", "content": resp.text})
        else:
            print(f"VERITAS > [Error: {resp.error}]")
            history.pop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete VERITAS backend with metrics report, audio, and CLI chat.")
    parser.add_argument("--before", default="samples/sample_before.png",
                        help="Source/before image path (default: samples/sample_before.png)")
    parser.add_argument("--after", default="samples/sample_after.jpg",
                        help="Reference/after image path (default: samples/sample_after.jpg)")
    parser.add_argument("--output-dir", default="outputs/reports",
                        help="Directory for reports and artifacts (default: outputs/reports)")
    parser.add_argument("--audio-dir", default="outputs/audio",
                        help="Directory for audio synthesis outputs (default: outputs/audio)")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio synthesis")
    parser.add_argument("--no-chat", action="store_true", help="Disable interactive CLI chat loop")
    args = parser.parse_args()

    before_path = Path(args.before)
    after_path = Path(args.after)

    if not before_path.exists():
        print(f"Error: Before image not found: {before_path}", file=sys.stderr)
        return 1
    if not after_path.exists():
        print(f"Error: After image not found: {after_path}", file=sys.stderr)
        return 1

    print(f"Running VERITAS Verification Chain...")
    print(f"  Before: {before_path}")
    print(f"  After:  {after_path}\n")

    result = verify_evidence_pair(str(before_path), str(after_path))

    # 1. AI Explanation
    print("Generating zero-pixel explanation...")
    explanation = explain(result.explanation_payload)

    # 2. Audio Synthesis with Conversational Voice Persona
    audio_result = None
    if not args.no_audio:
        print("Preparing spoken audio narration...")
        spoken_script = generate_spoken_narration(result, explanation["text"])
        print("Synthesizing audio with Puck voice...")
        audio_result = synthesize(spoken_script, output_dir=args.audio_dir)
        if audio_result.get("success"):
            print(f"  Audio generated: {audio_result.get('audio_path')}")
        else:
            print(f"  Audio note: {audio_result.get('error')}")

    # 3. Save JSON and Text Artifacts
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = result.audit.run_id
    json_path = output_dir / f"{run_id}.json"
    latest_json_path = output_dir / "latest.json"
    explanation_path = output_dir / f"{run_id}.explanation.json"
    metrics_txt_path = output_dir / f"{run_id}.metrics.txt"
    latest_txt_path = output_dir / "latest_metrics.txt"

    # Full JSON report
    full_data = result.to_dict()
    full_data["explanation"] = explanation
    if audio_result:
        full_data["tts"] = audio_result

    json_content = json.dumps(full_data, indent=2, sort_keys=True)
    json_path.write_text(json_content, encoding="utf-8")
    latest_json_path.write_text(json_content, encoding="utf-8")

    explanation_content = json.dumps(result.explanation_payload.to_dict(), indent=2, sort_keys=True)
    explanation_path.write_text(explanation_content, encoding="utf-8")

    # Format and save text report
    metrics_report = format_metrics_report(result, str(before_path), str(after_path), explanation, audio_result)
    metrics_txt_path.write_text(metrics_report, encoding="utf-8")
    latest_txt_path.write_text(metrics_report, encoding="utf-8")

    # Print summary to console
    print("\n" + metrics_report)

    print("\nGenerated Artifacts:")
    print(f"  - Metrics TXT:   {metrics_txt_path}")
    print(f"  - Latest TXT:    {latest_txt_path}")
    print(f"  - Audit JSON:    {json_path}")
    print(f"  - Latest JSON:   {latest_json_path}")
    print(f"  - Explanation:   {explanation_path}")
    if audio_result and audio_result.get("audio_path"):
        print(f"  - Audio WAV:     {audio_result.get('audio_path')}")

    # 4. Interactive Chat Loop
    if not args.no_chat and sys.stdin.isatty():
        start_interactive_chat(result, explanation["text"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
