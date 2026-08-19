"""
Interactive Image Color Picker (Standard Interface).
Uses CIELAB spatial color classification over 865+ reference colors.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import get_sample_image_path
from main import run_image_inspection


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Color Picker")
    parser.add_argument(
        "-i", "--image",
        default=str(get_sample_image_path()),
        help="Path to image file for color inspection"
    )
    args = parser.parse_args()
    run_image_inspection(Path(args.image))


if __name__ == "__main__":
    main()
