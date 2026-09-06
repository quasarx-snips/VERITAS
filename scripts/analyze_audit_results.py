"""Write an honest human-readable summary of audit_real_images output."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'/'real_world_audit'
def mean(values): return sum(values)/len(values) if values else 0
def main():
    rows=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    counts={n:[x['detectors'][n]['source_keypoints'] for x in rows] for n in ('sift','orb','akaze')}
    best=max(rows,key=lambda x:x['primary_geometry']['inlier_ratio']); worst=min(rows,key=lambda x:x['primary_geometry']['inlier_ratio'])
    lines=['# VERITAS REAL-WORLD PHASE 2 VALIDATION','', '## Dataset','',f'- Focused audit pairs evaluated: {len(rows)} (filename-paired; semantic labels remain UNKNOWN).',f'- Images discovered by the audit: 100 raw and 100 processed filename matches.','', '## Feature Extraction','']
    for n,v in counts.items(): lines.append(f'- {n.upper()}: source keypoints min/mean/max = {min(v)}/{mean(v):.1f}/{max(v)}; configured target: SIFT 6000, ORB 2000, AKAZE threshold-driven.')
    lines += ['', '## Visual Inspection','',f"- Strongest focused correspondence: {best['pair_id']} — SIFT {best['primary_geometry']['inlier_ratio']:.3f} inlier ratio, {best['primary_geometry']['inlier_count']} inliers.",f"- Weakest/failure case: {worst['pair_id']} — certified={worst['primary_geometry']['certified']}, SIFT {worst['primary_geometry']['inlier_count']} inliers from {worst['primary_geometry']['candidate_count']} filtered matches.",' - Semantic-change, translation, and ground-truth false-correspondence cases: UNKNOWN / not established from filenames alone.','', '## Detector Agreement','']
    for x in rows: lines.append(f"- {x['pair_id']}: {x['quorum']['detector_results']}.")
    lines += ['', '## Spatial Evidence','']
    for x in rows: lines.append(f"- {x['pair_id']}: coverage {x['spatial']['coverage_ratio']:.3f}, entropy {x['spatial']['normalized_entropy']:.3f}, dominant-cell fraction {x['counter_evidence']['spatial_concentration']['dominant_cell_fraction']}.")
    lines += ['', '## Bugs Found and Fixes Applied','', '- The original audit only serialized a pipeline wrapper; it had no visualizations, timing, dimensions, per-detector geometry, CSV schema, or dashboard. Replaced with an inspectable diagnostic audit.', '- Spatial concentration omitted dominant-cell share; it now reports dominant count/fraction and flags a single cell or >=80% dominance.', '- Direct `python scripts/audit_real_images.py` lacks the package root; use `python -m scripts.audit_real_images` (no `sys.path` mutation).','', '## Remaining Limitations','', '- This focused run has no supplied semantic/translation ground truth; it must not label those cases from appearance or filenames.', '- SIFT’s 6000 cap and ORB’s 2000 cap are practical inspection populations; AKAZE is threshold-driven and can yield substantially more.', '- A complete all-100 visual audit is supported via `python -m scripts.audit_real_images --visualize-all`, but was not substituted for this focused diagnostic run.','', '## Phase 3 Readiness','', '**NOT READY** — focused visual validation is complete, but dataset ground truth and an all-pair audit are still needed before the requested Phase 3 gate.']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__': main()
