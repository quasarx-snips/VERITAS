"""Build reproducible, judge-facing artifacts from the VERITAS backend.

This module is deliberately a presentation consumer: it reads the public
pipeline result and existing diagnostic artifacts, and never changes detector,
matching, geometry, verdict, or gate behavior.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from veritas.pipeline import verify_evidence_pair


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "judge_demo"
CASES = (
    ("strong_distributed", 55, "Strong distributed correspondence", "A real run with broad spatial support and all detector families supporting the affine certificate."),
    ("partial_restricted", 1, "Partial correspondence", "A real run with an affine certificate, but partial correspondence leads to a restricted downstream action."),
    ("detector_disagreement", 42, "Detector disagreement", "A real run where one detector family disagrees; the verdict remains cautious and requires human review."),
    ("sparse_support", 38, "Candidate false-correspondence case", "A real run with sparse spatial support. This is not labeled as a false alarm; its correspondence evidence requires human review."),
    ("insufficient_support", 16, "Insufficient correspondence", "A real run with limited geometric and spatial support; the gate blocks downstream use."),
)

PALETTE = {"STRONG": (43, 161, 102), "PARTIAL": (29, 151, 206), "WEAK": (36, 149, 239), "NONE": (67, 78, 94)}


def image_for(folder: str, index: int) -> Path:
    candidates = sorted((ROOT / folder).glob(f"ds_{index}.*"))
    if not candidates:
        raise FileNotFoundError(f"No ds_{index} image in {folder}")
    return candidates[0]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    return cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def place(canvas: np.ndarray, image: np.ndarray, x: int, y: int, width: int, height: int) -> None:
    item = fit(image, width, height)
    ox, oy = x + (width - item.shape[1]) // 2, y + (height - item.shape[0]) // 2
    canvas[oy:oy + item.shape[0], ox:ox + item.shape[1]] = item


def text(canvas: np.ndarray, value: str, at: tuple[int, int], scale: float = .55, color=(30, 41, 59), thickness: int = 1) -> None:
    cv2.putText(canvas, value, at, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def evidence_lines(report: dict[str, Any]) -> list[str]:
    evidence, gate = report["evidence"], report["gate"]
    geometry, spatial, quorum = evidence["geometry"], evidence["spatial"], evidence["quorum"]
    lines = []
    lines.append("Affine geometry certified" if geometry["certified"] else "Affine geometry certificate failed")
    lines.append(f"Detector quorum: {len(quorum['supporting_detectors'])}/3 supporting")
    lines.append(f"Spatial coverage: {spatial['occupied_cells']}/{spatial['grid_rows'] * spatial['grid_columns']} cells")
    if evidence["counter_evidence"]["spatial_concentration"].get("concentrated"):
        lines.append("Counter-evidence: spatial concentration detected")
    if quorum["disagreeing_detectors"]:
        lines.append("Counter-evidence: detector disagreement detected")
    for reason in gate["reasons"]:
        if reason == "AFFINE_CERTIFIED":
            continue
        readable = reason.replace("_", " ").title()
        lines.append(readable)
    return lines[:6]


def hero_visual(case: dict[str, Any], report: dict[str, Any], destination: Path) -> None:
    before, after = cv2.imread(str(case["before"])), cv2.imread(str(case["after"]))
    if before is None or after is None:
        raise RuntimeError(f"Unreadable demo image for {case['case_id']}")
    verdict, gate = report["verdict"]["verdict"], report["gate"]["action"]
    accent = PALETTE[verdict]
    canvas = np.full((1320, 2200, 3), (246, 248, 252), np.uint8)
    cv2.rectangle(canvas, (0, 0), (2200, 150), (15, 23, 42), -1)
    text(canvas, "VERITAS", (70, 65), 1.25, (255, 255, 255), 3)
    text(canvas, "Correspondence Verification", (70, 112), .62, (203, 213, 225), 1)
    text(canvas, f"CASE  {case['case_id']}", (1640, 84), .72, (255, 255, 255), 2)
    text(canvas, "BEFORE", (70, 205), .85, (15, 23, 42), 2)
    text(canvas, "AFTER", (980, 205), .85, (15, 23, 42), 2)
    place(canvas, before, 70, 235, 830, 625)
    place(canvas, after, 980, 235, 830, 625)
    cv2.arrowedLine(canvas, (875, 545), (950, 545), (29, 151, 206), 6, cv2.LINE_AA, tipLength=.25)
    cv2.rectangle(canvas, (70, 920), (2130, 1250), (255, 255, 255), -1)
    cv2.rectangle(canvas, (70, 920), (2130, 1250), (226, 232, 240), 2)
    text(canvas, "VERITAS DECISION", (110, 978), .72, (15, 23, 42), 2)
    text(canvas, f"VERDICT  {verdict}", (110, 1050), .85, accent, 2)
    text(canvas, f"GATE  {gate}", (110, 1115), .72, accent, 2)
    text(canvas, "WHY", (780, 978), .64, (15, 23, 42), 2)
    for i, line in enumerate(evidence_lines(report)[:5]):
        marker = "+" if not line.startswith("Counter-evidence") and "failed" not in line.lower() else "!"
        text(canvas, f"{marker}  {line}", (780, 1030 + 43 * i), .54, (30, 41, 59), 1)
    text(canvas, "Evidence is distinct from the verdict; the verdict is distinct from the gate.", (110, 1300), .48, (71, 85, 105), 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), canvas)


def contact_sheet(cases: list[dict[str, Any]], destination: Path) -> None:
    canvas = np.full((1650, 2100, 3), (246, 248, 252), np.uint8)
    cv2.rectangle(canvas, (0, 0), (2100, 135), (15, 23, 42), -1)
    text(canvas, "VERITAS  |  JUDGE DEMO CASES", (55, 80), 1.0, (255, 255, 255), 2)
    for i, case in enumerate(cases):
        report = case["report"]
        x, y = 45 + (i % 2) * 1025, 180 + (i // 2) * 470
        cv2.rectangle(canvas, (x, y), (x + 980, y + 420), (255, 255, 255), -1)
        cv2.rectangle(canvas, (x, y), (x + 980, y + 420), (226, 232, 240), 2)
        visual = cv2.imread(str(case["visual_path"]))
        place(canvas, visual, x + 18, y + 55, 590, 340)
        verdict, gate = report["verdict"]["verdict"], report["gate"]["action"]
        text(canvas, case["case_id"].upper(), (x + 20, y + 35), .55, (15, 23, 42), 2)
        text(canvas, verdict, (x + 635, y + 105), .72, PALETTE[verdict], 2)
        text(canvas, gate, (x + 635, y + 150), .58, PALETTE[verdict], 2)
        for j, line in enumerate(evidence_lines(report)[:3]):
            text(canvas, line[:41], (x + 635, y + 215 + j * 37), .43, (30, 41, 59), 1)
    cv2.imwrite(str(destination), canvas)


def architecture_visual(destination: Path) -> None:
    canvas = np.full((1250, 1800, 3), (246, 248, 252), np.uint8)
    cv2.rectangle(canvas, (0, 0), (1800, 135), (15, 23, 42), -1)
    text(canvas, "VERITAS EVIDENCE STACK", (60, 82), 1.0, (255, 255, 255), 2)
    boxes = [("SIFT", 120), ("ORB", 350), ("AKAZE", 580)]
    for label, x in boxes:
        cv2.rectangle(canvas, (x, 215), (x + 170, 285), (219, 234, 254), -1); text(canvas, label, (x + 35, 260), .65, (15, 23, 42), 2)
        cv2.line(canvas, (x + 85, 285), (900, 380), (29, 151, 206), 2)
    stages = [("DETECTOR QUORUM", 900, 380), ("MATCH EVIDENCE", 900, 545), ("AFFINE CERTIFICATE", 900, 710), ("COVERAGE  +  ENTROPY  +  COUNTER-EVIDENCE", 900, 875), ("EVIDENCE FUSION", 900, 1040), ("VERDICT  →  SAFETY GATE", 900, 1175)]
    last_y = 0
    for label, x, y in stages:
        if last_y: cv2.arrowedLine(canvas, (x, last_y + 35), (x, y - 40), (29, 151, 206), 3, cv2.LINE_AA, tipLength=.2)
        cv2.rectangle(canvas, (x - 310, y - 38), (x + 310, y + 38), (255, 255, 255), -1); cv2.rectangle(canvas, (x - 310, y - 38), (x + 310, y + 38), (203, 213, 225), 2)
        text(canvas, label, (x - min(280, len(label) * 9), y + 8), .58, (15, 23, 42), 2); last_y = y
    cv2.imwrite(str(destination), canvas)


def detector_visual(case: dict[str, Any], report: dict[str, Any], destination: Path) -> None:
    canvas = np.full((1060, 1800, 3), (246, 248, 252), np.uint8)
    cv2.rectangle(canvas, (0, 0), (1800, 130), (15, 23, 42), -1)
    text(canvas, f"MULTI-DETECTOR EVIDENCE  |  {case['case_id']}", (50, 78), .88, (255, 255, 255), 2)
    audit = ROOT / "outputs" / "real_world_audit" / "matches"
    number = case["index"]
    for i, detector in enumerate(("sift", "orb", "akaze")):
        x = 45 + i * 585
        data = report["evidence"]["matches"][detector]
        state = report["evidence"]["quorum"]["detector_results"][detector]
        text(canvas, detector.upper(), (x, 195), .82, (15, 23, 42), 2)
        text(canvas, f"candidate {data['candidate_count']}  →  filtered {data['filtered_count']}", (x, 235), .45, (71, 85, 105), 1)
        text(canvas, f"quorum: {state}", (x, 268), .45, (71, 85, 105), 1)
        source = audit / f"pair_{number:03d}_{detector}_affine_inliers.png"
        if source.exists(): place(canvas, cv2.imread(str(source)), x, 300, 530, 600)
        else: text(canvas, "Existing inlier diagnostic unavailable", (x, 500), .45, (100, 116, 139), 1)
    text(canvas, "Three complementary detector families are checked; quorum is evidence, not a probability.", (50, 1000), .52, (71, 85, 105), 1)
    cv2.imwrite(str(destination), canvas)


def spatial_visual(case: dict[str, Any], report: dict[str, Any], destination: Path) -> None:
    canvas = np.full((1030, 1800, 3), (246, 248, 252), np.uint8)
    cv2.rectangle(canvas, (0, 0), (1800, 130), (15, 23, 42), -1)
    text(canvas, f"SPATIAL EVIDENCE  |  {case['case_id']}", (50, 78), .88, (255, 255, 255), 2)
    source = ROOT / "outputs" / "real_world_audit" / "spatial" / f"pair_{case['index']:03d}_sift_inlier_grid.png"
    if source.exists(): place(canvas, cv2.imread(str(source)), 50, 185, 1060, 760)
    spatial = report["evidence"]["spatial"]
    concentration = report["evidence"]["counter_evidence"]["spatial_concentration"]
    x = 1190
    text(canvas, "MATCHES MUST BE DISTRIBUTED", (x, 230), .58, (15, 23, 42), 2)
    rows = [("GRID", f"{spatial['grid_rows']} x {spatial['grid_columns']}"), ("OCCUPIED", f"{spatial['occupied_cells']} cells"), ("COVERAGE", f"{spatial['coverage_ratio']:.3f}"), ("ENTROPY", f"{spatial['normalized_entropy']:.3f}"), ("CONCENTRATED", str(concentration.get('concentrated')).upper())]
    for i, (label, value) in enumerate(rows):
        text(canvas, label, (x, 330 + i * 110), .48, (71, 85, 105), 1); text(canvas, value, (x, 370 + i * 110), .78, (15, 23, 42), 2)
    text(canvas, "Coverage asks where support exists.", (50, 990), .5, (71, 85, 105), 1)
    text(canvas, "Entropy asks how evenly support is distributed.", (50, 1018), .5, (71, 85, 105), 1)
    cv2.imwrite(str(destination), canvas)


def trace_visual(case: dict[str, Any], report: dict[str, Any], destination: Path) -> None:
    evidence = report["evidence"]
    stages = [
        ("INPUT", f"{case['before'].name} → {case['after'].name}"),
        ("3 DETECTORS", " / ".join(f"{n.upper()}: {evidence['features'][n]['keypoint_count']} kp" for n in ('sift', 'orb', 'akaze'))),
        ("MATCHING", f"SIFT filtered: {evidence['matches']['sift']['filtered_count']}"),
        ("AFFINE RANSAC", "certified" if evidence['geometry']['certified'] else "not certified"),
        ("SPATIAL EVIDENCE", f"coverage {evidence['spatial']['coverage_ratio']:.3f}; entropy {evidence['spatial']['normalized_entropy']:.3f}"),
        ("COUNTER-EVIDENCE", ", ".join(report['gate']['reasons']) or "none"),
        ("VERDICT", report['verdict']['verdict']),
        ("SAFETY GATE", report['gate']['action']),
    ]
    canvas = np.full((1510, 1600, 3), (246, 248, 252), np.uint8)
    cv2.rectangle(canvas, (0, 0), (1600, 130), (15, 23, 42), -1); text(canvas, f"DECISION TRACE  |  {case['case_id']}", (50, 78), .88, (255, 255, 255), 2)
    for i, (name, value) in enumerate(stages):
        y = 180 + i * 155
        cv2.rectangle(canvas, (190, y), (1410, y + 92), (255, 255, 255), -1); cv2.rectangle(canvas, (190, y), (1410, y + 92), (203, 213, 225), 2)
        text(canvas, name, (225, y + 38), .58, (29, 151, 206), 2); text(canvas, value[:100], (225, y + 72), .45, (30, 41, 59), 1)
        if i < len(stages) - 1: cv2.arrowedLine(canvas, (800, y + 95), (800, y + 145), (29, 151, 206), 3, cv2.LINE_AA, tipLength=.2)
    cv2.imwrite(str(destination), canvas)


def json_block(data: Any) -> str:
    return html.escape(json.dumps(data, indent=2, sort_keys=True))


def report_html(case: dict[str, Any], report: dict[str, Any], destination: Path) -> None:
    audit = report["audit"]
    verdict, gate = report["verdict"]["verdict"], report["gate"]["action"]
    sections = [("INPUT", audit["input"]), ("EVIDENCE", report["evidence"]), ("GEOMETRY", audit["geometry"]), ("SPATIAL SUPPORT", audit["spatial"]), ("COUNTER-EVIDENCE", audit["counter_evidence"]), ("PARTIAL CORRESPONDENCE", audit["partial_correspondence"]), ("VERDICT", audit["verdict"]), ("SAFETY GATE", audit["gate"]), ("DECISION TRACE", audit["decision_trace"]), ("PROVENANCE", audit["provenance"]), ("EXPLANATION PAYLOAD", report["explanation_payload"])]
    cards = "".join(f"<li>{html.escape(line)}</li>" for line in evidence_lines(report))
    details = "".join(f"<details {'open' if title in ('EVIDENCE', 'GEOMETRY', 'SPATIAL SUPPORT', 'SAFETY GATE') else ''}><summary>{title}</summary><pre>{json_block(value)}</pre></details>" for title, value in sections)
    page = f"""<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>VERITAS | {case['case_id']}</title><style>
body{{margin:0;background:#f6f8fc;color:#1e293b;font:16px system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.5}}main{{max-width:1080px;margin:auto;padding:42px 24px 80px}}header{{background:#0f172a;color:white;padding:40px;border-radius:20px}}h1{{margin:0;font-size:2.3rem}}.eyebrow{{color:#93c5fd;letter-spacing:.12em;font-weight:700;font-size:.82rem}}.result{{display:flex;gap:18px;flex-wrap:wrap;margin:30px 0}}.pill{{padding:15px 22px;border-radius:12px;background:white;border:1px solid #e2e8f0;font-weight:800}}.pill b{{color:#0e7490}}section{{background:white;border:1px solid #e2e8f0;border-radius:16px;padding:24px;margin:20px 0}}details{{border-top:1px solid #e2e8f0;padding:16px 0}}summary{{cursor:pointer;font-weight:800}}pre{{background:#0f172a;color:#dbeafe;padding:18px;border-radius:10px;overflow:auto;font-size:.78rem}}a{{color:#0369a1}}.note{{color:#475569}}</style><main><header><div class='eyebrow'>VERITAS · CORRESPONDENCE VERIFICATION</div><h1>{html.escape(case['case_id'])}</h1><p>{html.escape(case['description'])}</p></header><div class='result'><div class='pill'>VERDICT<br><b>{verdict}</b></div><div class='pill'>SAFETY GATE<br><b>{gate}</b></div><div class='pill'>MODEL<br><b>AFFINE-ONLY</b></div></div><section><h2>Why this decision</h2><ul>{cards}</ul><p class='note'>This page renders the machine-readable pipeline result. It does not calculate a second decision.</p></section>{details}</main></html>"""
    destination.write_text(page, encoding="utf-8")


def overview_html(cases: list[dict[str, Any]], destination: Path) -> None:
    cards = "".join(f"<article><img src='cases/{c['case_id']}.png' alt='VERITAS case {c['case_id']}'><h2>{html.escape(c['title'])}</h2><p><b>{c['report']['verdict']['verdict']}</b> → <b>{c['report']['gate']['action']}</b></p><p>{html.escape(c['description'])}</p><a href='reports/{c['case_id']}.html'>Open audit report →</a></article>" for c in cases)
    page = f"""<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>VERITAS Judge Demo</title><style>body{{margin:0;background:#f6f8fc;color:#1e293b;font:16px system-ui,sans-serif}}main{{max-width:1300px;margin:auto;padding:42px 24px}}header{{background:#0f172a;color:#fff;padding:50px;border-radius:22px}}h1{{font-size:3rem;margin:0}}.lead{{font-size:1.25rem;max-width:800px;color:#dbeafe}}.flow{{background:white;border-radius:16px;padding:25px;margin:22px 0;border:1px solid #e2e8f0;font-weight:700;color:#0e7490}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:22px}}article{{background:white;border:1px solid #e2e8f0;border-radius:16px;padding:18px}}article img{{width:100%;border-radius:10px}}a{{color:#0369a1;font-weight:700}}.note{{color:#475569}}</style><main><header><p>VERITAS · JUDGE DEMONSTRATION</p><h1>Trust correspondence before trusting change.</h1><p class='lead'>VERITAS is an evidence-first verification layer for image pairs. It assembles detector, affine, spatial, and counter-evidence before issuing a deterministic verdict and downstream safety action.</p></header><div class='flow'>IMAGE PAIR → 3 DETECTORS → AFFINE CERTIFICATE → SPATIAL + COUNTER-EVIDENCE → VERDICT → SAFETY GATE → AUDIT</div><h2>Real curated cases</h2><div class='grid'>{cards}</div><h2>The safety boundary</h2><p><b>STRONG → ALLOW</b> · <b>PARTIAL → RESTRICT</b> · <b>WEAK → HUMAN REVIEW</b> · <b>NONE → BLOCK</b></p><p class='note'>Future SGLang and TTS adapters receive structured, zero-pixel explanation data. They cannot alter the verdict or gate. Benchmark calibration and production UI remain future work.</p><p><a href='false_alarm_defense.html'>Open false-alarm defense →</a> · <a href='JUDGE_FLOW.md'>90-second judge story →</a> · <a href='TECHNICAL_DEPTH.md'>Technical depth →</a></p></main></html>"""
    destination.write_text(page, encoding="utf-8")


def false_alarm_html(case: dict[str, Any], destination: Path) -> None:
    report = case["report"]
    page = f"""<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>VERITAS | False-alarm defense</title><style>body{{margin:0;background:#f6f8fc;color:#1e293b;font:17px system-ui,sans-serif;line-height:1.55}}main{{max-width:1100px;margin:auto;padding:42px 24px}}header,.step{{background:white;border:1px solid #e2e8f0;border-radius:16px;padding:26px;margin:18px 0}}header{{background:#0f172a;color:#fff}}img{{max-width:100%;border-radius:12px}}.arrow{{text-align:center;font-size:2rem;color:#0284c7}}.warning{{border-left:5px solid #f59e0b;padding-left:16px}}</style><main><header><h1>False-alarm defense</h1><p>Do not trust visual difference until correspondence has been verified.</p></header><div class='step'><h2>Before → After</h2><img src='cases/{case['case_id']}.png' alt='Candidate false-correspondence case'></div><div class='arrow'>↓</div><div class='step'><h2>What a naive change detector could see</h2><p>Visual differences alone are insufficient to establish a real-world change. This repository has no ground-truth change label for this pair.</p></div><div class='arrow'>↓</div><div class='step'><h2>VERITAS check</h2><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in evidence_lines(report))}</ul></div><div class='arrow'>↓</div><div class='step'><h2>VERITAS result</h2><p><b>{report['verdict']['verdict']} → {report['gate']['action']}</b></p><p class='warning'>This is a candidate false-correspondence case, not a claim that any downstream alarm is false. The gate acts on the available correspondence evidence.</p><p><a href='reports/{case['case_id']}.html'>Inspect the complete machine-readable audit rendering →</a></p></div></main></html>"""
    destination.write_text(page, encoding="utf-8")


def documentation() -> tuple[str, str, str, str]:
    readme = """# VERITAS judge demo\n\nRegenerate every judge-facing artifact with `python -m scripts.build_judge_demo`. Open `VERITAS_demo_report.html` in a browser. The builder runs or reuses real deterministic backend reports for its curated cases and then renders presentation assets from those contracts.\n\nThe demo intentionally includes constrained and failure cases. Source paths in the manifest are repository-relative.\n"""
    flow = """# 90-second judge story\n\n1. **The problem** — visual difference is unreliable if the images are not corresponded.\n2. **Failure mode** — sparse, repeated, or detector-specific matches can look persuasive.\n3. **VERITAS intervention** — verify correspondence before downstream change analysis.\n4. **Multi-detector evidence** — SIFT, ORB, and AKAZE are recorded separately.\n5. **Affine certificate** — geometry is affine-only; there is no automatic model choice or homography fallback.\n6. **Spatial distribution** — coverage and entropy distinguish distributed support from a local cluster.\n7. **Counter-evidence** — residuals, disagreement, and concentration stay visible.\n8. **Verdict** — evidence is classified as STRONG, PARTIAL, WEAK, or NONE.\n9. **Safety gate** — the deterministic gate controls downstream action.\n10. **Auditability** — every result has a JSON report, trace, provenance, and rendered human view.\n11. **Future SGLang/TTS** — adapters may explain structured evidence; they cannot override the gate.\n"""
    depth = """# Technical depth\n\n- **Why affine-only:** a bounded affine model avoids silently escalating to a more flexible fit.\n- **Why three detectors:** SIFT, ORB, and AKAZE provide complementary implementation paths; their statuses remain inspectable.\n- **Why quorum:** agreement is evidence; disagreement remains counter-evidence.\n- **Coverage vs. entropy:** coverage measures occupied grid cells; entropy measures the distribution of support over those cells.\n- **Counter-evidence:** residual quality, detector disagreement, and spatial concentration are preserved rather than hidden.\n- **Partial correspondence:** credible local support can be represented without calling the entire pair globally aligned.\n- **Evidence vs. verdict:** evidence is measured input; verdict is deterministic classification.\n- **Verdict vs. gate:** a gate action controls downstream handling and is a separate contract.\n- **Zero-pixel explanation:** future language or speech adapters receive structured output rather than raw image pixels, and have no decision authority.\n- **Provenance:** reports preserve input identifiers, configuration, decision trace, and schema version.\n\nThis is architectural integration and safety framing, not a claim of benchmark calibration or physical-world change ground truth.\n"""
    source = """# VERITAS judge-demo source report\n\n**Audience:** project judges  \n**Scope:** current repository implementation and generated local evidence  \n**Date:** 2026-09-07\n\n## Executive answer\n\nVERITAS currently provides a deterministic image-pair correspondence verification chain: detector evidence, affine-only geometry, spatial and counter-evidence, partial correspondence, verdict, safety gate, audit/provenance, and a zero-pixel explanation payload. The judge demo is a presentation layer over that chain; it does not alter it.\n\n## Evidence basis\n\n- `veritas/pipeline.py` is the source of the end-to-end contract.\n- `veritas/verdict/` and `veritas/gate/` define separate verdict and deterministic safety-gate contracts.\n- `veritas/audit/` and `veritas/llm/` define the audit/provenance and zero-pixel explanation payloads.\n- `outputs/real_world_audit/results.json` and the five regenerated pipeline reports provide the curated, real-case measurements.\n- `pytest -q` passed 101 tests during final verification.\n\n## Limitations\n\nNo physical-world change ground truth, benchmark calibration, SGLang runtime, TTS runtime, or final production UI is claimed. The sparse-support example is intentionally framed as a candidate false-correspondence case rather than a proven false alarm.\n"""
    return readme, flow, depth, source


def build(force: bool = False) -> list[dict[str, Any]]:
    (OUT / "cases").mkdir(parents=True, exist_ok=True); (OUT / "reports").mkdir(parents=True, exist_ok=True)
    curated = []
    for case_id, index, title, description in CASES:
        before, after = image_for("data/raw", index), image_for("data/processed", index)
        report_path = OUT / "reports" / f"{case_id}.json"
        if report_path.exists() and not force:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = verify_evidence_pair(str(before), str(after), run_id=f"judge-demo-{case_id}").to_dict()
            write_json(report_path, report)
            write_json(OUT / "reports" / f"{case_id}.explanation.json", report["explanation_payload"])
        case = {"case_id": case_id, "index": index, "title": title, "description": description, "before": before, "after": after, "report": report, "visual_path": OUT / "cases" / f"{case_id}.png"}
        hero_visual(case, report, case["visual_path"]); report_html(case, report, OUT / "reports" / f"{case_id}.html")
        curated.append(case)
    manifest = {"schema_version": "1.0", "purpose": "Curated judge-facing cases generated from real VERITAS backend runs.", "cases": [{"case_id": c["case_id"], "before_image": c["before"].relative_to(ROOT).as_posix(), "after_image": c["after"].relative_to(ROOT).as_posix(), "description": c["description"], "actual_backend_result": {"verdict": c["report"]["verdict"]["verdict"], "gate": c["report"]["gate"]["action"], "partial_correspondence": c["report"]["partial_correspondence"]["status"]}, "report_path": f"outputs/judge_demo/reports/{c['case_id']}.json", "rendered_report_path": f"outputs/judge_demo/reports/{c['case_id']}.html"} for c in curated], "unavailable_categories": ["Semantic change with trustworthy correspondence: no supplied change ground truth or downstream change outputs."]}
    write_json(OUT / "demo_manifest.json", manifest)
    contact_sheet(curated, OUT / "demo_contact_sheet.png"); architecture_visual(OUT / "evidence_architecture.png")
    detector_visual(next(c for c in curated if c["case_id"] == "detector_disagreement"), next(c for c in curated if c["case_id"] == "detector_disagreement")["report"], OUT / "detector_comparison.png")
    spatial_case = next(c for c in curated if c["case_id"] == "sparse_support"); spatial_visual(spatial_case, spatial_case["report"], OUT / "spatial_evidence.png")
    trace_case = next(c for c in curated if c["case_id"] == "strong_distributed"); trace_visual(trace_case, trace_case["report"], OUT / "decision_trace.png")
    false_alarm_html(spatial_case, OUT / "false_alarm_defense.html"); overview_html(curated, OUT / "VERITAS_demo_report.html")
    readme, flow, depth, source = documentation(); (OUT / "README.md").write_text(readme, encoding="utf-8"); (OUT / "JUDGE_FLOW.md").write_text(flow, encoding="utf-8"); (OUT / "TECHNICAL_DEPTH.md").write_text(depth, encoding="utf-8"); (OUT / "report-source.md").write_text(source, encoding="utf-8")
    return curated


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate judge-facing VERITAS presentation artifacts.")
    parser.add_argument("--force", action="store_true", help="Re-run curated backend cases even when cached reports exist.")
    args = parser.parse_args()
    cases = build(args.force)
    print(f"Judge demo built: {len(cases)} cases in {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
