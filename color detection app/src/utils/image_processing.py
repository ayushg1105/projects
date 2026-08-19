"""
Image Processing and Color Space Utility Functions.
Includes perceptual luminance calculation, morphological cleanup, and color space transformations.
"""

from typing import Tuple
import cv2
import numpy as np


def resize_with_aspect_ratio(
    image: np.ndarray,
    max_width: int = 900,
    max_height: int = 650,
    allow_upscale: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Resizes an image maintaining the aspect ratio to fit the target dimensions.
    If allow_upscale is True, images will expand to fill the target container.
    
    Args:
        image: Source image numpy array (BGR or RGB).
        max_width: Target max allowed width.
        max_height: Target max allowed height.
        allow_upscale: Whether to upscale smaller images to fill container.
        
    Returns:
        Tuple of (resized_image, scale_factor).
    """
    height, width = image.shape[:2]
    if not allow_upscale and width <= max_width and height <= max_height:
        return image.copy(), 1.0
        
    scale_factor = min(max_width / width, max_height / height)
    new_width = max(1, int(width * scale_factor))
    new_height = max(1, int(height * scale_factor))
    
    interp = cv2.INTER_LANCZOS4 if scale_factor > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image, (new_width, new_height), interpolation=interp)
    return resized, scale_factor


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Converts RGB integers (0-255) to a standard hex color string."""
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}".upper()


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Converts a standard hex string (e.g. '#FFFFFF' or 'FFFFFF') to RGB tuple."""
    clean_hex = hex_str.lstrip('#')
    if len(clean_hex) == 3:
        clean_hex = ''.join([c * 2 for c in clean_hex])
    return tuple(int(clean_hex[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_lab(r: int, g: int, b: int) -> np.ndarray:
    """
    Converts a single sRGB color tuple to CIELAB space using OpenCV colorimetry.
    """
    pixel_bgr = np.uint8([[[b, g, r]]])
    pixel_lab = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2LAB)
    return pixel_lab[0, 0]


def get_contrasting_text_color(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """
    Calculates whether white (255, 255, 255) or black (0, 0, 0) text gives better contrast
    over a given background color using standard relative luminance formula (WCAG 2.0).
    
    Returns:
        BGR tuple for OpenCV drawing: (255, 255, 255) for dark bg, (0, 0, 0) for light bg.
    """
    # Relative luminance
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def apply_morphological_cleanup(mask: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """
    Applies morphological Opening (removes noise) and Closing (closes holes) on a binary mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # Opening: erode then dilate (removes small false positives)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Closing: dilate then erode (fills small holes inside detected contours)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed
