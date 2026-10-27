#!/usr/bin/env python
"""Generate descriptive threshold diagnostics from frozen B1/B2 OOF files."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brighter.threshold_diagnostics import generate_threshold_diagnostics


if __name__ == "__main__":
    result = generate_threshold_diagnostics(PROJECT_ROOT)
    print(result["warning"])
    print(f"Threshold sweep rows: {result['rows']}")
