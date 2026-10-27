"""Run read-only post-hoc diagnostics from frozen OOF/final artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from brighter.generalization_analysis import run_analysis

if __name__=="__main__":
    result=run_analysis(ROOT)
    print(result["summary"]["analysis"])
