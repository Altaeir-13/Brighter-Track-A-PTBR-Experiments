from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from brighter.finalization import train_final_model
if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--system",required=True,choices=("T2","T3")); args=parser.parse_args()
    result=train_final_model(ROOT,args.system); print(f"{args.system} final complete: {result['training_seconds']:.1f}s, {result['parameters']} parameters, peak={result['peak_vram_bytes']} bytes")
    print("Test: SEALED - NOT LOADED OR EVALUATED")
