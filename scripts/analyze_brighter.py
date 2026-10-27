#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brighter.curation import FROZEN_REVISION  # noqa: E402
from brighter.eda import run_eda  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic multilabel EDA on BRIGHTER PT-BR Track A")
    parser.add_argument("--revision", default=FROZEN_REVISION, help="Frozen dataset revision")
    parser.add_argument("--tokenizer-revision", help="Optional pinned BERTimbau tokenizer revision")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        summary, _ = run_eda(
            PROJECT_ROOT,
            revision=args.revision,
            tokenizer_revision=args.tokenizer_revision,
        )
    except Exception as exc:
        logging.exception("EDA failed: %s", exc)
        return 1
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
