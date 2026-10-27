"""Optional inference on user-supplied, unlabelled texts using verified final weights.

Added for distribution; not the official test entry point. Requires the external
final model directories and CUDA used by the original prediction function.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def check_model(directory,system):
    ref=ROOT/'artifacts/models/final'/('t2_bertweetbr_asl' if system=='T2' else 't3_bertimbau_large_bce')/'final_training_manifest.json'
    manifest=json.loads(ref.read_text(encoding='utf-8'))
    for name,expected in manifest['file_hashes'].items():
        p=directory/name
        if not p.is_file() or digest(p)!=expected:
            raise RuntimeError(f'{system} missing or different frozen file: {name}')

def load_texts(path):
    rows=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        row=json.loads(line)
        if set(row)!={'id','text'} or not isinstance(row['text'],str) or not isinstance(row['id'],str):
            raise ValueError('Each JSONL row must contain only string id and text; no gold labels.')
        if row['id'].startswith('ptbr_test_'):
            raise ValueError('Official test IDs are excluded from this interface.')
        rows.append(row)
    if not rows or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Input must contain unique IDs and at least one unlabelled text.')
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--t2-directory',type=Path,required=True)
    parser.add_argument('--t3-directory',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() or output.is_relative_to(ROOT):
        parser.error('Output must be new and outside the frozen evidence checkout.')
    if not output.parent.is_dir(): parser.error('Output parent must exist.')
    if args.input.resolve().is_relative_to(ROOT/'data'):
        parser.error('Do not pass dataset files to this unlabelled-text interface.')
    rows=load_texts(args.input)
    check_model(args.t2_directory,'T2')
    check_model(args.t3_directory,'T3')
    import numpy as np
    from brighter.final_test import _predict,fuse_en2,LABELS,EN2_THRESHOLDS
    texts=[r['text'] for r in rows]
    t2,_=_predict(args.t2_directory,texts,128,16)
    t3,_=_predict(args.t3_directory,texts,128,2)
    probabilities=fuse_en2(t2,t3)
    decisions=(probabilities>=np.array([EN2_THRESHOLDS[x] for x in LABELS])).astype(int)
    with output.open('x',encoding='utf-8') as f:
        for row,prob,decision in zip(rows,probabilities,decisions,strict=True):
            f.write(json.dumps({'id':row['id'],'probabilities':dict(zip(LABELS,map(float,prob))),
                                'decisions':dict(zip(LABELS,map(int,decision)))},ensure_ascii=False)+'\n')
    print(f'Predictions saved for {len(rows)} unlabelled texts. No evaluation or tuning performed.')

if __name__=='__main__': main()
