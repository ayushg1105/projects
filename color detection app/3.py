"""
Advanced Multi-Target Object Tracker (Standard Interface).
Uses real-time spatial segmentation and CIELAB KD-Tree color classification.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from main import run_cli_video_tracker


def main() -> None:
    run_cli_video_tracker("all")


if __name__ == "__main__":
    main()
