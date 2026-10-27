"""Single immutable final evaluation entry point; one-time authorization required."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from brighter.final_test import run_final_test


if __name__=="__main__":
    # No CLI options are accepted: models, alpha, thresholds, and losses come
    # exclusively from the already frozen configs/final_en2.yaml.
    result=run_final_test(ROOT)
    print(f"EN2 final Macro-F1: {result['primary']['metrics']['macro_f1']:.6f}")
    print(f"T2 final Macro-F1: {result['secondary']['metrics']['macro_f1']:.6f}")
