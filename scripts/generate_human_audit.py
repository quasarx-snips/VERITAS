"""Create presentation-ready, human-labelled BEFORE -> AFTER audit sheets.

This is visualization only.  It never creates detections or assigns a label.
"""
from __future__ import annotations
import argparse, csv, json, math, time
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'/'human_audit'

def paired_cases():
    """Use explicit test outputs when present; otherwise paired dataset images."""
    tests=sorted((ROOT/'outputs').glob('test_*'))
    if tests:
        result=[]
        for d in tests:
            imgs=[p for p in d.rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg')]
            result.append({'id':d.name,'before':imgs[0] if imgs else None,'after':imgs[1] if len(imgs)>1 else None,'output':d})
        return result
    raw,processed=ROOT/'data/raw',ROOT/'data/processed'; result=[]
    for i in range(1,1001):
        before=next((raw/f'ds_{i}{x}' for x in ('.png','.jpg','.jpeg') if (raw/f'ds_{i}{x}').exists()),None); after=next((processed/f'ds_{i}{x}' for x in ('.png','.jpg','.jpeg') if (processed/f'ds_{i}{x}').exists()),None)
        if before and after: result.append({'id':f'test_{i:03d}','before':before,'after':after,'output':None})
    return result

def fit(image,w,h):
    scale=min(w/image.shape[1],h/image.shape[0]); size=(round(image.shape[1]*scale),round(image.shape[0]*scale)); return cv2.resize(image,size,interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
def text(canvas,value,at,scale=.65,color=(35,35,35),thickness=1): cv2.putText(canvas,value,at,cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)
def put_center(canvas,image,x,y,w,h):
    image=fit(image,w,h); ox=x+(w-image.shape[1])//2; oy=y+(h-image.shape[0])//2; canvas[oy:oy+image.shape[0],ox:ox+image.shape[1]]=image
def arrow(canvas,x,y):
    cv2.arrowedLine(canvas,(x-48,y),(x+48,y),(205,110,0),8,cv2.LINE_AA,tipLength=.25)

def load_evidence():
    p=ROOT/'outputs/real_world_audit/results.json'
    if not p.exists(): return {}
    return {x['pair_id']:x for x in json.loads(p.read_text(encoding='utf-8'))}

def sheet(case,evidence,path):
    # One self-contained, high-resolution sheet per case.  Contact sheets are
    # secondary navigation only and never replace these individual artifacts.
    canvas=np.full((2450,2200,3),250,np.uint8); text(canvas,case['id'].upper(),(70,75),1.3,(20,20,20),2); text(canvas,'HUMAN VISUAL VERIFICATION — UNLABELED',(70,120),.65,(120,80,10),2)
    before=cv2.imread(str(case['before'])) if case['before'] else None; after=cv2.imread(str(case['after'])) if case['after'] else None
    if before is not None: put_center(canvas,before,70,190,930,700)
    if after is not None: put_center(canvas,after,1200,190,930,700)
    text(canvas,'BEFORE',(70,165),.95,(20,20,20),2); text(canvas,'AFTER',(1200,165),.95,(20,80,170),2); arrow(canvas,1100,615)
    text(canvas,f"File: {case['before'].name if case['before'] else 'N/A'}",(70,930),.52); text(canvas,f"File: {case['after'].name if case['after'] else 'N/A'}",(1200,930),.52)
    pair='pair_'+case['id'].split('_')[-1]
    match_path=ROOT/'outputs'/'real_world_audit'/'matches'/f'{pair}_sift_affine_inliers.png'
    match_image=cv2.imread(str(match_path)) if match_path.exists() else None
    cv2.rectangle(canvas,(70,970),(2130,1335),(225,225,225),2); text(canvas,'AFFINE INLIER CORRESPONDENCES — GREEN LINES ARE ACTUAL SIFT RANSAC INLIERS',(95,1010),.58,(0,110,0),2)
    if match_image is not None: put_center(canvas,match_image,95,1030,2010,285)
    else: text(canvas,'N/A: correspondence artifact unavailable.',(95,1170),.6,(80,80,150),1)
    cv2.rectangle(canvas,(70,1375),(2130,2365),(225,225,225),2); text(canvas,'MEASURED VERITAS RESULTS — THIS PAIR ONLY',(95,1425),.7,(20,20,20),2)
    row=evidence.get(pair)
    if row:
        d=row['detectors']; g=row['primary_geometry']; s=row['spatial']; q=row['quorum']
        y=1480
        for name in ('sift','orb','akaze'):
            x=d[name]; text(canvas,f"{name.upper()}: A/B keypoints {x['source_keypoints']}/{x['reference_keypoints']} | descriptors {x['source_descriptors']}/{x['reference_descriptors']} ({x['descriptor_type']}) | candidate/filtered {x['candidate_matches']}/{x['filtered_matches']} | affine inliers/outliers {x['inliers']}/{x['outliers']} | {q['detector_results'][name]}",(95,y),.48); y+=43
        residual=g['residuals']; text(canvas,f"PRIMARY AFFINE (SIFT): certified={g['certified']} | inlier ratio={g['inlier_ratio']:.4f} | matrix={g['affine_matrix']}",(95,y),.46); y+=43
        text(canvas,f"RESIDUALS: mean={residual.get('mean')} | median={residual.get('median')} | RMSE={residual.get('rmse')} | P95={residual.get('p95')} | maximum={residual.get('maximum')}",(95,y),.48); y+=43
        conc=row['counter_evidence']['spatial_concentration']; text(canvas,f"SPATIAL: {s['occupied_cells']}/{s['total_cells']} cells | coverage={s['coverage_ratio']:.4f} | normalized entropy={s['normalized_entropy']:.4f} | dominant cell={conc.get('dominant_cell_fraction')} | concentrated={conc['concentrated']}",(95,y),.48); y+=43
        text(canvas,f"COUNTER-EVIDENCE: {row['counter_evidence']['residuals']} | detector disagreement={q['disagreeing_detectors'] or 'none'}",(95,y),.44); y+=43
        text(canvas,f"TIMING: total={row['timing']['total_seconds']:.3f}s | preprocessing={row['timing']['preprocessing_seconds']:.3f}s | spatial={row['timing']['spatial_seconds']:.3f}s",(95,y),.48); y+=43
        text(canvas,'EXISTING CHANGE DETECTION: N/A (no masks, boxes, contours, or output/test_* artifacts in this repository).',(95,y),.5,(80,80,150),1)
    else: text(canvas,'CORRESPONDENCE EVIDENCE: N/A (audit result not yet available). EXISTING CHANGE DETECTION: N/A.',(95,1480),.58)
    text(canvas,'HUMAN VERIFICATION    [ ] TRUE ALARM    [ ] FALSE ALARM    [ ] UNCERTAIN     Notes: ____________________________________________',(95,2315),.55,(25,25,25),1)
    cv2.imwrite(str(path),canvas)
    return before is not None and after is not None

def contact_sheet(cases,case_dir):
    pages=[]
    for page,start in enumerate(range(0,len(cases),20),1):
        subset=cases[start:start+20]; canvas=np.full((2200,2000,3),250,np.uint8)
        for j,case in enumerate(subset):
            img=cv2.imread(str(case_dir/f"{case['id']}.png")); thumb=fit(img,470,390); x=25+(j%4)*495; y=30+(j//4)*435; canvas[y:y+thumb.shape[0],x:x+thumb.shape[1]]=thumb; text(canvas,case['id'].upper(),(x,y+410),.6,(20,20,20),2)
        dest=OUT/f'contact_sheet_{page:02d}.png'; cv2.imwrite(str(dest),canvas); pages.append(dest.name)
    return pages

def main():
    global OUT
    parser=argparse.ArgumentParser(); parser.add_argument('--all',action='store_true'); parser.add_argument('--test',nargs='+'); parser.add_argument('--output-dir',type=Path,default=OUT); args=parser.parse_args(); OUT=args.output_dir
    started=time.perf_counter(); cases=paired_cases(); cases=[x for x in cases if not args.test or x['id'].split('_')[-1] in args.test]; case_dir=OUT/'cases'; case_dir.mkdir(parents=True,exist_ok=True); evidence=load_evidence(); successes=[]; failures=[]
    for case in cases:
        try:
            if sheet(case,evidence,case_dir/f"{case['id']}.png"): successes.append(case)
            else: failures.append(case['id'])
        except Exception: failures.append(case['id'])
    with (OUT/'human_labels.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=['test_id','before_image','after_image','detection_output','detected_regions','detection_area','human_label','reviewer_notes']); w.writeheader(); [w.writerow({'test_id':x['id'],'before_image':str(x['before'].relative_to(ROOT)) if x['before'] else 'N/A','after_image':str(x['after'].relative_to(ROOT)) if x['after'] else 'N/A','detection_output':str(x['output'].relative_to(ROOT)) if x['output'] else 'N/A','detected_regions':'N/A','detection_area':'N/A','human_label':'UNLABELED','reviewer_notes':''}) for x in cases]
    pages=contact_sheet(successes,case_dir) if successes else []; elapsed=time.perf_counter()-started
    (OUT/'summary.md').write_text(f"# Human audit generation\n\n- Cases discovered: {len(cases)}\n- Visualized: {len(successes)}\n- Failures: {len(failures)} ({', '.join(failures) or 'none'})\n- Existing change-detection outputs: N/A; no `outputs/test_*` directories exist.\n- Contact sheets: {', '.join(pages) or 'none'}\n- Runtime: {elapsed:.2f}s ({elapsed/max(1,len(cases)):.2f}s/case)\n- Labels are intentionally all `UNLABELED`.\n",encoding='utf-8')
    (OUT/'README.md').write_text('Run `python -m scripts.generate_human_audit --all`. Sheets preserve source aspect ratio and show only existing evidence; N/A means no existing change-detection output was found.\n',encoding='utf-8')
    print(f'Human audit: {len(successes)}/{len(cases)} visualized; {len(failures)} failures; {elapsed:.2f}s')
if __name__=='__main__': main()
