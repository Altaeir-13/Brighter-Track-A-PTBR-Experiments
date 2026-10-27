from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from brighter.finalization import prepare_deployment_freeze
if __name__=="__main__":
    result=prepare_deployment_freeze(ROOT); d=result["decision"]
    print(f"alpha_t2={d['alpha_t2']:.2f}; weight_t3={d['weight_t3']:.2f}; calibration_only_macro_f1={d['deployment_calibration']['macro_f1']:.6f}")
    print("Test: SEALED - NOT LOADED OR EVALUATED")
