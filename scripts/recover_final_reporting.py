"""Reporting-only recovery; never loads models or the official test dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from brighter.final_reporting import finalize_recovery_state, recover_reporting


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--reporting-only",action="store_true",required=True)
    parser.add_argument("--finalize-state",action="store_true")
    args=parser.parse_args()
    recover_reporting(ROOT)
    if args.finalize_state: finalize_recovery_state(ROOT)


if __name__=="__main__": main()
