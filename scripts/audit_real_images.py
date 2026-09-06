"""Diagnostic-only, inspectable Phase 1/2 audit; no verdict or gate logic."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from veritas.features import AkazeDetector, OrbDetector, SiftDetector, assess_quorum
from veritas.geometry import compute_reprojection_errors, verify_affine_from_correspondences
from veritas.matching import match_feature_sets
from veritas.preprocessing import ImagePreprocessor
from veritas.spatial import calculate_spatial_coverage, calculate_spatial_entropy
from veritas.counter_evidence import summarize_feature_disagreement, summarize_residuals, summarize_spatial_concentration
from veritas.verification import geometry_evidence

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "outputs" / "real_world_audit"
DETECTORS = {"sift": SiftDetector, "orb": OrbDetector, "akaze": AkazeDetector}
# Filename pairings/categories are inferences only: no ground truth was supplied.
IMPORTANT = (1, 13, 16, 18, 19, 25, 50, 75, 100)

def image_path(folder, i):
    return next((folder / f"ds_{i}{s}" for s in (".png", ".jpg", ".jpeg") if (folder / f"ds_{i}{s}").exists()), None)

def pairs():
    raw, processed = ROOT / "data/raw", ROOT / "data/processed"
    return [{"pair_id": f"pair_{i:03d}", "index": i, "image_a": a, "image_b": b,
             "category": "UNKNOWN (filename-paired; no supplied ground truth)"}
            for i in range(1,1001) if (a:=image_path(raw,i)) and (b:=image_path(processed,i))]

def bgr(img): return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
def serial(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,dict): return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [serial(v) for v in x]
    return x

def keypoints(img, features, path, title):
    base=bgr(img); overlay=base.copy()
    for f in features: cv2.circle(overlay,(round(f.x),round(f.y)),2,(0,230,255),-1,cv2.LINE_AA)
    canvas=cv2.addWeighted(overlay,.58,base,.42,0)
    cv2.putText(canvas,title,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(canvas,title,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA); cv2.imwrite(str(path),canvas)

def matches_image(a,b,src,dst,path,title,limit=300):
    left,right=bgr(a),bgr(b); canvas=np.zeros((max(left.shape[0],right.shape[0]),left.shape[1]+right.shape[1],3),np.uint8)
    canvas[:left.shape[0],:left.shape[1]]=left; canvas[:right.shape[0],left.shape[1]:]=right
    if len(src)>limit: pick=np.linspace(0,len(src)-1,limit,dtype=int); src,dst=src[pick],dst[pick]
    for p,q in zip(src,dst):
        # Fixed green is intentional: a reviewer can trace real correspondence
        # lines across every diagnostic without interpreting a colour legend.
        color=(0,255,0); p=(round(p[0]),round(p[1])); q=(round(q[0]+left.shape[1]),round(q[1]))
        cv2.line(canvas,p,q,color,1,cv2.LINE_AA); cv2.circle(canvas,p,2,color,-1,cv2.LINE_AA); cv2.circle(canvas,q,2,color,-1,cv2.LINE_AA)
    cv2.putText(canvas,title,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA); cv2.imwrite(str(path),canvas)

def spatial_image(img, coverage, path, title):
    canvas=bgr(img); h,w=canvas.shape[:2]; rows,cols=coverage["grid_shape"]
    for r in range(rows):
        for c in range(cols):
            x0,y0,x1,y1=round(c*w/cols),round(r*h/rows),round((c+1)*w/cols),round((r+1)*h/rows); n=int(coverage["points_per_cell"][r,c])
            if n:
                overlay=canvas.copy(); cv2.rectangle(overlay,(x0,y0),(x1,y1),(0,min(230,35+n),255),-1); canvas=cv2.addWeighted(overlay,.22,canvas,.78,0)
            cv2.rectangle(canvas,(x0,y0),(x1,y1),(255,255,255),1); cv2.putText(canvas,str(n),(x0+5,y0+20),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),3); cv2.putText(canvas,str(n),(x0+5,y0+20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    cv2.putText(canvas,title,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2); cv2.imwrite(str(path),canvas)

def dashboard(result, path):
    g,s,q,c=result["primary_geometry"],result["spatial"],result["quorum"],result["counter_evidence"]
    rows=[f"# {result['pair_id']} diagnostic","",f"- Image A: `{result['image_a']}`",f"- Image B: `{result['image_b']}`",f"- Category: {result['category']}","","## Detector evidence","","| Detector | A/B keypoints | filtered | inliers | state |","|---|---:|---:|---:|---|"]
    rows += [f"| {n} | {d['source_keypoints']}/{d['reference_keypoints']} | {d['filtered_matches']} | {d['inliers']} | {q['detector_results'][n]} |" for n,d in result['detectors'].items()]
    rows += ["","## Geometry (SIFT primary certificate)","",f"- Certified: `{g['certified']}`; inlier ratio: `{g['inlier_ratio']:.4f}`",f"- Affine matrix: `{g['affine_matrix']}`",f"- Residuals: `{g['residuals']}`","","## Spatial and counter-evidence","",f"- Coverage: `{s['coverage_ratio']:.4f}` ({s['occupied_cells']}/{s['total_cells']}); entropy: `{s['normalized_entropy']:.4f}`",f"- Counter-evidence: `{c}`",f"- Quorum: `{q}`","","## Fusion evidence","","None: this diagnostic layer implements no verdict, score, or gate action."]
    path.write_text("\n".join(rows),encoding="utf-8")

def audit(pair, visuals):
    a,b=cv2.imread(str(pair['image_a'])),cv2.imread(str(pair['image_b']))
    if a is None or b is None: raise RuntimeError(f"Unreadable image in {pair['pair_id']}")
    start=time.perf_counter(); t=time.perf_counter(); prep=ImagePreprocessor(); source,ref=prep.process(a),prep.process(b); timings={"preprocessing_seconds":time.perf_counter()-t}; channels={}
    for name, factory in DETECTORS.items():
        t=time.perf_counter(); detector=factory(); sf,sd=detector.detect(source.enhanced); rf,rd=detector.detect(ref.enhanced); feature=time.perf_counter()-t
        t=time.perf_counter(); m=match_feature_sets((sf,sd),(rf,rd)); matching=time.perf_counter()-t
        t=time.perf_counter(); cert=verify_affine_from_correspondences(m); geometry=time.perf_counter()-t
        channels[name]={"sf":sf,"rf":rf,"sd":sd,"rd":rd,"m":m,"cert":cert}; timings.update({f"{name}_feature_seconds":feature,f"{name}_matching_seconds":matching,f"{name}_geometry_seconds":geometry})
    primary=channels['sift']['cert']; t=time.perf_counter(); coverage=calculate_spatial_coverage(primary.source_inliers,source.enhanced.shape); entropy=calculate_spatial_entropy(primary.source_inliers,source.enhanced.shape); timings['spatial_seconds']=time.perf_counter()-t
    geometries={n:geometry_evidence(ch['cert']) for n,ch in channels.items()}; quorum=assess_quorum(geometries)
    residuals=compute_reprojection_errors(primary.source_inliers,primary.reference_inliers,primary.transformation) if primary.transformation is not None else np.empty(0)
    counter={"residuals":summarize_residuals(residuals,primary.threshold),"feature_disagreement":summarize_feature_disagreement(quorum),"spatial_concentration":summarize_spatial_concentration(coverage,entropy)}
    detectors={n:{"source_keypoints":len(ch['sf']),"reference_keypoints":len(ch['rf']),"source_descriptors":len(ch['sd']),"reference_descriptors":len(ch['rd']),"descriptor_type":str(ch['sd'].dtype),"candidate_matches":ch['m'].candidate_count,"filtered_matches":ch['m'].accepted_count,"inliers":ch['cert'].diagnostics.inlier_count,"outliers":ch['cert'].diagnostics.outlier_count,"certified":ch['cert'].is_valid,"affine_matrix":ch['cert'].transformation,"residuals":geometries[n].residuals,"filter_diagnostics":ch['m'].filter_diagnostics} for n,ch in channels.items()}
    timings['total_seconds']=time.perf_counter()-start
    result={"pair_id":pair['pair_id'],"image_a":str(pair['image_a'].relative_to(ROOT)),"image_b":str(pair['image_b'].relative_to(ROOT)),"category":pair['category'],"image_a_metadata":{"width":a.shape[1],"height":a.shape[0],"channels":a.shape[2],"format":pair['image_a'].suffix},"image_b_metadata":{"width":b.shape[1],"height":b.shape[0],"channels":b.shape[2],"format":pair['image_b'].suffix},"detectors":detectors,"primary_geometry":geometries['sift'].to_dict(),"spatial":{**serial(coverage),**serial(entropy)},"quorum":quorum.to_dict(),"counter_evidence":serial(counter),"fusion_evidence":None,"timing":timings}
    if visuals:
        for d in (OUT/'features',OUT/'matches',OUT/'spatial',OUT/'dashboard'): d.mkdir(parents=True,exist_ok=True)
        for n,ch in channels.items():
            keypoints(a,ch['sf'],OUT/'features'/f"{pair['pair_id']}_a_{n}_keypoints.png",f"{pair['pair_id']} A {n}: {len(ch['sf'])} actual keypoints"); keypoints(b,ch['rf'],OUT/'features'/f"{pair['pair_id']}_b_{n}_keypoints.png",f"{pair['pair_id']} B {n}: {len(ch['rf'])} actual keypoints")
            m,cert=ch['m'],ch['cert']; cs=np.array([(ch['sf'][x.queryIdx].x,ch['sf'][x.queryIdx].y) for x in m.candidate_matches]).reshape(-1,2); cd=np.array([(ch['rf'][x.trainIdx].x,ch['rf'][x.trainIdx].y) for x in m.candidate_matches]).reshape(-1,2)
            matches_image(a,b,cs,cd,OUT/'matches'/f"{pair['pair_id']}_{n}_candidate_matches.png",f"{pair['pair_id']} {n}: candidates ({len(cs)}, <=300 shown)"); matches_image(a,b,m.source_points,m.reference_points,OUT/'matches'/f"{pair['pair_id']}_{n}_filtered_matches.png",f"{pair['pair_id']} {n}: filtered ({m.accepted_count}, <=300 shown)"); matches_image(a,b,m.source_points[cert.inlier_mask],m.reference_points[cert.inlier_mask],OUT/'matches'/f"{pair['pair_id']}_{n}_affine_inliers.png",f"{pair['pair_id']} {n}: affine inliers ({cert.diagnostics.inlier_count})"); matches_image(a,b,m.source_points[~cert.inlier_mask],m.reference_points[~cert.inlier_mask],OUT/'matches'/f"{pair['pair_id']}_{n}_affine_outliers.png",f"{pair['pair_id']} {n}: affine outliers ({cert.diagnostics.outlier_count})")
        spatial_image(a,coverage,OUT/'spatial'/f"{pair['pair_id']}_sift_inlier_grid.png",f"{pair['pair_id']} SIFT inlier grid: {coverage['occupied_cells']}/{coverage['total_cells']} cells"); dashboard(result,OUT/'dashboard'/f"{pair['pair_id']}.md")
    return serial(result)

def write(results):
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    fields='pair_id image_a image_b category sift_keypoints orb_keypoints akaze_keypoints sift_matches orb_matches akaze_matches sift_inliers orb_inliers akaze_inliers affine_status affine_matrix inlier_ratio rmse p95_residual coverage occupied_cells normalized_entropy quorum_state counter_evidence fusion_evidence total_seconds'.split()
    with (OUT/'results.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for x in results:
            d,g,s=x['detectors'],x['primary_geometry'],x['spatial']; w.writerow({'pair_id':x['pair_id'],'image_a':x['image_a'],'image_b':x['image_b'],'category':x['category'],**{f'{n}_keypoints':d[n]['source_keypoints'] for n in DETECTORS},**{f'{n}_matches':d[n]['filtered_matches'] for n in DETECTORS},**{f'{n}_inliers':d[n]['inliers'] for n in DETECTORS},'affine_status':g['certified'],'affine_matrix':json.dumps(g['affine_matrix']),'inlier_ratio':g['inlier_ratio'],'rmse':g['residuals'].get('rmse'),'p95_residual':g['residuals'].get('p95'),'coverage':s['coverage_ratio'],'occupied_cells':s['occupied_cells'],'normalized_entropy':s['normalized_entropy'],'quorum_state':json.dumps(x['quorum']['detector_results']),'counter_evidence':json.dumps(x['counter_evidence']),'fusion_evidence':None,'total_seconds':x['timing']['total_seconds']})

def main():
    p=argparse.ArgumentParser(); p.add_argument('--limit',type=int); p.add_argument('--pair-ids',nargs='+',type=int,help='Specific numeric ds IDs; useful for focused inspection.'); p.add_argument('--visualize-all',action='store_true'); p.add_argument('--append',action='store_true',help='Preserve previously written focused-audit rows.'); args=p.parse_args(); found=pairs(); found=[x for x in found if x['index'] in args.pair_ids] if args.pair_ids else found; found=found[:args.limit] if args.limit else found; results=[]
    if args.append and (OUT/'results.json').exists():
        results=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
        done={x['pair_id'] for x in results}; found=[x for x in found if x['pair_id'] not in done]
    for pair in found: print(f"Auditing {pair['pair_id']}..."); results.append(audit(pair,args.visualize_all or pair['index'] in IMPORTANT)); write(results)
    print(f"Audit complete: {len(results)} pairs in {OUT}")
if __name__=='__main__': main()
