#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brighter.download import run_download  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Download revision-pinned BRIGHTER PT-BR data")
    parser.add_argument("--revision", help="Commit SHA, tag, or branch to resolve and pin")
    parser.add_argument("--force", action="store_true", help="Redownload to a temporary area and verify immutability")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        summary, analysis = run_download(PROJECT_ROOT, args.revision, args.force)
    except Exception as exc:
        logging.error("%s", exc)
        return 1
    print("\n" + summary)
    return 0 if analysis["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

