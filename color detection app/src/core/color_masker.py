"""
Real-time Color Masking and Multi-Channel HSV Filter Engine.
Implements color isolation (bitwise AND filtering), multi-channel split views (Red, Green, Blue, Custom),
and interactive HSV range calibration.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from config.settings import VISION_CONFIG
from src.utils.image_processing import apply_morphological_cleanup


class MaskFilterType(Enum):
    """Supported color masking filter presets."""
    RED = "Red"
    GREEN = "Green"
    BLUE = "Blue"
    YELLOW = "Yellow"
    ORANGE = "Orange"
    PURPLE = "Purple"
    MULTI_GRID = "Multi-Channel Grid (RGB)"
    CUSTOM_HSV = "Custom HSV Calibration"


@dataclass
class MaskResult:
    """Encapsulates the outputs of a color masking operation."""
    isolated_frame: np.ndarray       # Bitwise-AND isolated color image (background blacked out)
    binary_mask: np.ndarray          # 2D uint8 binary mask (0 or 255)
    pixel_count: int                 # Number of non-zero pixels detected in mask
    coverage_percentage: float       # Percentage of total frame area matching the color


class ColorMaskEngine:
    """
    High-performance color isolation, masking, and HSV tuning engine.
    """

    def __init__(self):
        self.hsv_ranges = VISION_CONFIG.HSV_COLOR_RANGES
        self.kernel_size = VISION_CONFIG.MORPH_KERNEL_SIZE

    def apply_filter(
        self,
        frame: np.ndarray,
        filter_type: MaskFilterType = MaskFilterType.BLUE,
        custom_lower: Optional[np.ndarray] = None,
        custom_upper: Optional[np.ndarray] = None
    ) -> MaskResult:
        """
        Applies color isolation filter to a single video frame.
        
        Args:
            frame: Input BGR frame.
            filter_type: Desired MaskFilterType preset.
            custom_lower: Optional custom lower HSV bound (shape (3,)).
            custom_upper: Optional custom upper HSV bound (shape (3,)).
            
        Returns:
            MaskResult containing the bitwise-isolated image and binary mask.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        total_pixels = h * w

        if filter_type == MaskFilterType.CUSTOM_HSV:
            lower = custom_lower if custom_lower is not None else np.array([0, 50, 50], dtype=np.uint8)
            upper = custom_upper if custom_upper is not None else np.array([180, 255, 255], dtype=np.uint8)
            raw_mask = cv2.inRange(hsv, lower, upper)
            mask = apply_morphological_cleanup(raw_mask, self.kernel_size)
            
        elif filter_type == MaskFilterType.RED:
            config = self.hsv_ranges["Red"]
            m1 = cv2.inRange(hsv, config["lower1"], config["upper1"])
            m2 = cv2.inRange(hsv, config["lower2"], config["upper2"])
            raw_mask = cv2.bitwise_or(m1, m2)
            mask = apply_morphological_cleanup(raw_mask, self.kernel_size)
            
        else:
            color_name = filter_type.value
            config = self.hsv_ranges.get(color_name, self.hsv_ranges["Blue"])
            raw_mask = cv2.inRange(hsv, config["lower"], config["upper"])
            mask = apply_morphological_cleanup(raw_mask, self.kernel_size)

        # Bitwise-AND isolation (retains colored pixels, blacks out background)
        isolated = cv2.bitwise_and(frame, frame, mask=mask)
        
        pixel_count = int(cv2.countNonZero(mask))
        coverage = (pixel_count / total_pixels) * 100.0

        return MaskResult(
            isolated_frame=isolated,
            binary_mask=mask,
            pixel_count=pixel_count,
            coverage_percentage=round(coverage, 2)
        )

    def generate_multi_grid_view(self, frame: np.ndarray) -> np.ndarray:
        """
        Generates a 2x2 multi-channel monitoring grid:
        - Top-Left: Original Feed
        - Top-Right: Isolated Blue Mask
        - Bottom-Left: Isolated Red Mask
        - Bottom-Right: Isolated Green Mask
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        half_w, half_h = w // 2, h // 2

        # 1. Original (Top-Left)
        tl = cv2.resize(frame, (half_w, half_h))
        cv2.putText(tl, "Original Feed", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # 2. Blue Isolation (Top-Right)
        b_res = self.apply_filter(frame, MaskFilterType.BLUE)
        tr = cv2.resize(b_res.isolated_frame, (half_w, half_h))
        cv2.putText(tr, f"Blue Mask ({b_res.coverage_percentage}%)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        # 3. Red Isolation (Bottom-Left)
        r_res = self.apply_filter(frame, MaskFilterType.RED)
        bl = cv2.resize(r_res.isolated_frame, (half_w, half_h))
        cv2.putText(bl, f"Red Mask ({r_res.coverage_percentage}%)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # 4. Green Isolation (Bottom-Right)
        g_res = self.apply_filter(frame, MaskFilterType.GREEN)
        br = cv2.resize(g_res.isolated_frame, (half_w, half_h))
        cv2.putText(br, f"Green Mask ({g_res.coverage_percentage}%)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Assemble 2x2 grid
        top_row = np.hstack([tl, tr])
        bottom_row = np.hstack([bl, br])
        grid = np.vstack([top_row, bottom_row])
        
        return grid
