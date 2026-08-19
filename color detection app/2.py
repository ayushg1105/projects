"""
Color Masking & Multi-Channel Filter (Standard Interface).
Uses real-time HSV isolation and 2x2 split grid monitoring.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from main import run_cli_color_masker


def main() -> None:
    run_cli_color_masker("grid")


if __name__ == "__main__":
    main()
