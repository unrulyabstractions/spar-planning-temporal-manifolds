"""Relevance-aware horizon decoder experiment. Builds on Alan's `ptm` package, which is imported
from ../alan without modifying or installing it (the alan/ folder is his snapshot; never write there)."""
import sys
from pathlib import Path

_ALAN = Path(__file__).resolve().parents[2] / "alan"
if str(_ALAN) not in sys.path:
    sys.path.insert(0, str(_ALAN))
