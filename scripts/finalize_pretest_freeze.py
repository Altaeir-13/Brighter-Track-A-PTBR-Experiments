from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from brighter.finalization import dry_run_final_models,finalize_freeze
if __name__=="__main__":
    dry_run_final_models(ROOT); result=finalize_freeze(ROOT); print(result["experiment_status"]); print("Test: SEALED - NOT LOADED OR EVALUATED")
