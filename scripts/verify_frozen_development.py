"""Recompute development F1 from frozen OOF decisions; never opens test artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['anger','disgust','fear','joy','sadness','surprise']
FILES = {
    'CT1': 'bertweetbr_weighted_focal_thresholded_oof.parquet',
    'CT2': 'bertweetbr_asl_thresholded_oof.parquet',
    'CT3': 'bertimbau_large_bce_thresholded_oof.parquet',
    'EN1': 'ensemble_t2_t3_equal_weight_oof.parquet',
    'EN2': 'ensemble_t2_t3_tuned_weight_oof.parquet',
}

def verify():
    import pandas as pd
    from sklearn.metrics import f1_score
    summary=json.loads((ROOT/'metadata/brighter_ptbr/experiments/ensemble_summary.json').read_text(encoding='utf-8'))
    reference={x['system']:x for x in summary['main_results']}
    output=[]
    for system,name in FILES.items():
        frame=pd.read_parquet(ROOT/'artifacts/predictions'/name)
        assert len(frame)==2426 and frame.id.nunique()==2426
        assert frame.id.str.startswith(('ptbr_train_track_a_','ptbr_dev_track_a_')).all()
        true=frame[[x+'_true' for x in LABELS]].to_numpy()
        pred=frame[[x+'_pred' for x in LABELS]].to_numpy()
        macro=float(f1_score(true,pred,average='macro',zero_division=0))
        micro=float(f1_score(true,pred,average='micro',zero_division=0))
        key={'CT2':'T2','CT3':'T3'}.get(system,system)
        if key in reference:
            assert abs(macro-reference[key]['macro_f1'])<1e-12
            assert abs(micro-reference[key]['micro_f1'])<1e-12
        else:
            assert round(macro,4)==0.5152
        output.append({'system':system,'rows':len(frame),'macro_f1':macro,'micro_f1':micro})
    return {'scope':'Frozen development OOF decisions only; no model execution or parameter selection',
            'official_test_opened':False,'results':output}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output:
        path=args.output.resolve()
        if path.is_relative_to(ROOT) or path.exists():
            parser.error('Output must be new and outside the evidence checkout.')
    result=verify()
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output: args.output.write_text(text,encoding='utf-8')
    print(text)

if __name__=='__main__': main()
