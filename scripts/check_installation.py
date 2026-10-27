"""Check imports, frozen configurations and fold metadata without data/models."""
from __future__ import annotations
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def check():
    modules=[]
    for p in sorted((ROOT/'src/brighter').glob('*.py')):
        if p.stem=='__init__': continue
        importlib.import_module('brighter.'+p.stem)
        modules.append(p.stem)
    import yaml
    from brighter.experiments import load_config
    configs=[]
    for p in sorted((ROOT/'configs').glob('*.yaml')):
        if p.name.startswith('final_'):
            obj=yaml.safe_load(p.read_text(encoding='utf-8'))
            assert obj['frozen'] is True and obj['development_examples']==2426
            assert obj['dataset_revision']=='419566ac8f46f951b51584be36c13e5363008144'
            assert obj['label_order']==['anger','disgust','fear','joy','sadness','surprise']
            if p.name=='final_en2.yaml':
                assert obj['alpha_t2']==0.7
                assert list(obj['thresholds'].values())==[.67,.36,.43,.65,.58,.49]
        else: load_config(p)
        configs.append(p.name)
    assert len(configs)==15
    folds=json.loads((ROOT/'metadata/brighter_ptbr/experiments/cv_folds.json').read_text(encoding='utf-8'))
    outer={r['id']:r['fold'] for r in folds['assignments']}
    assert len(outer)==2426 and set(outer.values())==set(range(5))
    inner_count=0
    for p in (ROOT/'metadata/brighter_ptbr/experiments/threshold_inner_folds').rglob('outer_*.json'):
        block=json.loads(p.read_text(encoding='utf-8'))
        inside={r['id']:r['inner_fold'] for r in block['assignments']}
        expected={i for i,f in outer.items() if f!=block['outer_fold']}
        assert set(inside)==expected
        assert set(inside.values())=={0,1,2}
        inner_count+=1
    assert inner_count==15
    packages=['torch','transformers','tokenizers','datasets','huggingface-hub','numpy','pandas','pyarrow','scikit-learn','iterative-stratification','matplotlib','pytest']
    return {'python':platform.python_version(),'platform':platform.system(),
            'package_versions':{p:importlib.metadata.version(p) for p in packages},
            'modules_imported':modules,'configs_validated':configs,
            'outer_assignment_count':len(outer),'inner_manifests_validated':inner_count,
            'dataset_read':False,'models_loaded':False,'official_test_evaluated':False}

if __name__=='__main__':
    print(json.dumps(check(),ensure_ascii=False,indent=2))
