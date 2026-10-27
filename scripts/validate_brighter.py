#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brighter.download import validate_existing  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local BRIGHTER PT-BR Parquet splits")
    parser.add_argument("--revision", help="Expected pinned commit SHA")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        summary, analysis = validate_existing(PROJECT_ROOT, args.revision)
    except Exception as exc:
        logging.error("%s", exc)
        return 1
    for warning in analysis["warnings"]:
        logging.warning("%s", warning)
    for error in analysis["errors"]:
        logging.error("%s", error)
    print("\n" + summary)
    return 0 if analysis["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
