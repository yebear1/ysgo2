"""Compatibility imports for the shared navigation implementation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navigation.high_level_nav_policy import *  # noqa: F401,F403
