"""Persist the explicit one-time final-test authorization after sealed pre-flight."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from brighter.final_test import create_final_test_authorization

if __name__=="__main__":
    create_final_test_authorization(ROOT)
