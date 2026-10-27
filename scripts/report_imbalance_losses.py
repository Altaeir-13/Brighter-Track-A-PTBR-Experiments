#!/usr/bin/env python
"""Validate and consolidate the completed L1--L4 battery."""

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

from brighter.imbalance_reporting import generate_imbalance_loss_reports


if __name__ == "__main__":
    result = generate_imbalance_loss_reports(PROJECT_ROOT)
    print("L1-L4 reports complete")
    print("Selected losses:", ", ".join(result["selected_losses"]))
