"""
Configuration settings for the Industrial Color Vision & AI Analytics Platform.
Provides centralized, dataclass-based configurations for models, computer vision pipelines,
paths, and GUI components.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple
import numpy as np


# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"

# Fallback path checking (checks data/raw first, then project root)
def get_colors_csv_path() -> Path:
    raw_path = RAW_DATA_DIR / "colors.csv"
    root_path = BASE_DIR / "colors.csv"
    return raw_path if raw_path.exists() else root_path

def get_sample_image_path() -> Path:
    sample_path = SAMPLE_IMAGES_DIR / "colorpic.jpg"
    root_path = BASE_DIR / "colorpic.jpg"
    return sample_path if sample_path.exists() else root_path


@dataclass(frozen=True)
class VisionConfig:
    """Computer vision, color space, and tracking configurations."""
    # Video capture settings
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    FPS_LIMIT: int = 60
    
    # Image resize constraints for interactive analyzer
    MAX_IMAGE_WIDTH: int = 900
    MAX_IMAGE_HEIGHT: int = 650
    
    # Minimum contour area for object detection thresholding (reduces noise)
    MIN_CONTOUR_AREA: int = 500
    ALL_COLORS_MIN_AREA: int = 800
    
    # Morphological kernel size
    MORPH_KERNEL_SIZE: Tuple[int, int] = (5, 5)
    
    # Pre-calibrated HSV bounds for key spectral colors
    # Format: {"ColorName": (Lower_HSV, Upper_HSV, Display_BGR_Color)}
    HSV_COLOR_RANGES: Dict[str, Dict] = field(default_factory=lambda: {
        "Red": {
            "lower1": np.array([0, 70, 50], dtype=np.uint8),
            "upper1": np.array([10, 255, 255], dtype=np.uint8),
            "lower2": np.array([170, 70, 50], dtype=np.uint8),
            "upper2": np.array([180, 255, 255], dtype=np.uint8),
            "bgr": (0, 0, 255)
        },
        "Green": {
            "lower": np.array([35, 50, 50], dtype=np.uint8),
            "upper": np.array([85, 255, 255], dtype=np.uint8),
            "bgr": (0, 255, 0)
        },
        "Blue": {
            "lower": np.array([90, 50, 50], dtype=np.uint8),
            "upper": np.array([130, 255, 255], dtype=np.uint8),
            "bgr": (255, 0, 0)
        },
        "Yellow": {
            "lower": np.array([20, 100, 100], dtype=np.uint8),
            "upper": np.array([34, 255, 255], dtype=np.uint8),
            "bgr": (0, 255, 255)
        },
        "Orange": {
            "lower": np.array([11, 100, 100], dtype=np.uint8),
            "upper": np.array([25, 255, 255], dtype=np.uint8),
            "bgr": (0, 165, 255)
        },
        "Purple": {
            "lower": np.array([130, 50, 50], dtype=np.uint8),
            "upper": np.array([160, 255, 255], dtype=np.uint8),
            "bgr": (128, 0, 128)
        }
    })


@dataclass(frozen=True)
class MLConfig:
    """Machine Learning & Color Science hyperparameter configuration."""
    DEFAULT_K_CLUSTERS: int = 5
    MAX_K_CLUSTERS: int = 10
    MIN_K_CLUSTERS: int = 2
    KMEANS_SAMPLE_SIZE: int = 2000
    KMEANS_RANDOM_STATE: int = 42
    
    # Color classification distance metric: 'cielab_e76' or 'euclidean_rgb'
    DISTANCE_METRIC: str = "cielab_e76"


@dataclass(frozen=True)
class GUIConfig:
    """User Interface styling and geometry parameters."""
    APP_TITLE: str = "ColorPulse Vision - Real-Time Spatial Color Intelligence"
    WINDOW_SIZE: str = "1400x880"
    THEME_APPEARANCE: str = "Dark"
    COLOR_THEME: str = "blue"
    CANVAS_WIDTH: int = 900
    CANVAS_HEIGHT: int = 650


# Global instances
VISION_CONFIG = VisionConfig()
ML_CONFIG = MLConfig()
GUI_CONFIG = GUIConfig()
