from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brighter.ensemble import run_ensemble


if __name__ == "__main__":
    result = run_ensemble(ROOT)
    for system in ("EN1", "EN2"):
        print(f"{system}: Macro-F1={result['systems'][system]['metrics']['macro_f1']:.6f}")
    print("Training performed: false")
    print("Test: SEALED - NOT LOADED OR EVALUATED")
