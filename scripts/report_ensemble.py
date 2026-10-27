from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brighter.ensemble_reporting import generate_ensemble_reports


if __name__ == "__main__":
    summary = generate_ensemble_reports(ROOT)
    for row in summary["main_results"]:
        print(f"{row['system']}: Macro-F1={row['macro_f1']:.6f}")
    print(f"Decision: {summary['decision']} ({summary['selected_system']})")
    print("Test: SEALED - NOT LOADED OR EVALUATED")
