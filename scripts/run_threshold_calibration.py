from __future__ import annotations

import argparse
from pathlib import Path

from brighter.threshold_calibration import estimate_t3_seconds, run_nested_system

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("T1", "T2", "T3"))
    parser.add_argument("--estimate-t3", action="store_true")
    args = parser.parse_args()
    if args.estimate_t3:
        print(estimate_t3_seconds(ROOT))
    elif args.system:
        result = run_nested_system(ROOT, args.system)
        print(f"{args.system} COMPLETE: {result['fixed_metrics']['macro_f1']:.4f} -> {result['thresholded_metrics']['macro_f1']:.4f}")
        print("Test set: SEALED - NOT EVALUATED")
    else:
        parser.error("select --system or --estimate-t3")
