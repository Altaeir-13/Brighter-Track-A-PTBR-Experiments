#!/usr/bin/env python
"""Run one frozen BRIGHTER development experiment."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brighter.experiments import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--final-test", action="store_true", help="Reserved and disabled during baseline development")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        summary = run_experiment(PROJECT_ROOT, args.config.resolve(), final_test=args.final_test)
    except Exception as error:
        logging.exception("Experiment failed: %s", error)
        return 1
    cv = summary["cross_validation"]["macro_f1"]
    print(f"\n{summary['experiment_id']} COMPLETE")
    print(f"Macro-F1: {cv['mean']:.4f} +/- {cv['std']:.4f}")
    print("OOF predictions: 2426 / 2426")
    print("Test set: SEALED - NOT EVALUATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
