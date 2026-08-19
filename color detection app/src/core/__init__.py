"""Core package exports."""
from src.core.color_classifier import ColorClassifier, ColorMatchResult
from src.core.color_extractor import DominantColorExtractor, DominantColor, PaletteExtractionResult
from src.core.object_tracker import ColorObjectTracker, TrackingMode, TrackedObject
from src.core.color_masker import ColorMaskEngine, MaskFilterType, MaskResult
from src.core.video_stream import VideoStream

__all__ = [
    "ColorClassifier",
    "ColorMatchResult",
    "DominantColorExtractor",
    "DominantColor",
    "PaletteExtractionResult",
    "ColorObjectTracker",
    "TrackingMode",
    "TrackedObject",
    "ColorMaskEngine",
    "MaskFilterType",
    "MaskResult",
    "VideoStream"
]
