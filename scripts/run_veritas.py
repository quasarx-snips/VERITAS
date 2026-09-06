"""Run the complete deterministic VERITAS backend for one image pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from veritas.pipeline import verify_evidence_pair


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete VERITAS backend.")
    parser.add_argument("--before", required=True, help="Source/before image path")
    parser.add_argument("--after", required=True, help="Reference/after image path")
    parser.add_argument("--output-dir", default="outputs/reports", help="Directory for the JSON report")
    args = parser.parse_args()
    result = verify_evidence_pair(args.before, args.after)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{result.audit.run_id}.json"
    payload_path = output_dir / f"{result.audit.run_id}.explanation.json"
    report_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    payload_path.write_text(json.dumps(result.explanation_payload.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print("VERITAS RESULT\n==============")
    print(f"\nVerdict: {result.verdict.verdict.value}\nGate: {result.gate.action.value}")
    print("\nKey evidence:\n  " + ", ".join(result.verdict.rationale_codes or ["none"]))
    print("\nCounter-evidence:\n  " + ", ".join(result.gate.reasons or ["none"]))
    print(f"\nPartial correspondence:\n  {result.partial_correspondence.status.value}")
    print("\nDecision reasons:\n  " + ", ".join(result.gate.reasons))
    print(f"\nAudit:\n  {report_path}\n\nExplanation payload:\n  {payload_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
