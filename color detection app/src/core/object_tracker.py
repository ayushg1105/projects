"""
Spatial Color Segmentation and Multi-Object Tracking Engine.
Provides real-time contour detection, centroid calculation, bounding box prediction,
and multi-spectral color tracking.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from config.settings import VISION_CONFIG
from src.core.color_classifier import ColorClassifier
from src.utils.image_processing import apply_morphological_cleanup, get_contrasting_text_color


class TrackingMode(Enum):
    """Supported color tracking operation modes."""
    ALL_COLORS = "all_colors"
    RED = "Red"
    GREEN = "Green"
    BLUE = "Blue"
    YELLOW = "Yellow"
    ORANGE = "Orange"
    PURPLE = "Purple"
    MULTI_SIMULTANEOUS = "multi_simultaneous"
    OFF = "off"


@dataclass
class TrackedObject:
    """Represents a single detected and localized color object."""
    label: str
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    centroid: Tuple[int, int]                # (cx, cy)
    area: float
    bgr_color: Tuple[int, int, int]
    rgb_color: Tuple[int, int, int]


class ColorObjectTracker:
    """
    Real-time multi-target spatial color segmentation and tracker.
    """

    def __init__(self, classifier: Optional[ColorClassifier] = None):
        self.classifier = classifier or ColorClassifier()
        self.hsv_ranges = VISION_CONFIG.HSV_COLOR_RANGES
        self.kernel_size = VISION_CONFIG.MORPH_KERNEL_SIZE
        self.min_area = VISION_CONFIG.MIN_CONTOUR_AREA
        self.all_colors_min_area = VISION_CONFIG.ALL_COLORS_MIN_AREA

    def process_frame(
        self,
        frame: np.ndarray,
        mode: TrackingMode = TrackingMode.ALL_COLORS,
        draw_annotations: bool = True
    ) -> Tuple[np.ndarray, List[TrackedObject], Optional[np.ndarray]]:
        """
        Processes a video frame according to the selected tracking mode.
        
        Args:
            frame: Input BGR frame.
            mode: TrackingMode enum.
            draw_annotations: If True, draws overlays and bounding boxes on frame.
            
        Returns:
            Tuple of (annotated_frame, list_of_tracked_objects, binary_mask).
        """
        output_frame = frame.copy()
        tracked_objects: List[TrackedObject] = []
        mask: Optional[np.ndarray] = None

        if mode == TrackingMode.OFF:
            if draw_annotations:
                cv2.putText(
                    output_frame, "Tracking: OFF (Raw Feed)",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA
                )
            return output_frame, tracked_objects, None

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if mode == TrackingMode.ALL_COLORS:
            # Segment any chromatic objects (filter out low-saturation / extreme-brightness noise)
            lower_chroma = np.array([0, 50, 50], dtype=np.uint8)
            upper_chroma = np.array([180, 255, 255], dtype=np.uint8)
            raw_mask = cv2.inRange(hsv_frame, lower_chroma, upper_chroma)
            mask = apply_morphological_cleanup(raw_mask, self.kernel_size)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.all_colors_min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    cx = x + w // 2
                    cy = y + h // 2
                    
                    # Sample centroid pixel color (clamp bounds)
                    sample_y = max(0, min(frame.shape[0] - 1, cy))
                    sample_x = max(0, min(frame.shape[1] - 1, cx))
                    b, g, r = frame[sample_y, sample_x]
                    
                    # Perceptual color match via KD-Tree
                    match_result = self.classifier.classify_rgb(int(r), int(g), int(b))
                    
                    obj = TrackedObject(
                        label=match_result.name,
                        bounding_box=(x, y, w, h),
                        centroid=(cx, cy),
                        area=float(area),
                        bgr_color=(int(b), int(g), int(r)),
                        rgb_color=(int(r), int(g), int(b))
                    )
                    tracked_objects.append(obj)
                    
                    if draw_annotations:
                        self._draw_object_overlay(output_frame, obj)

            if draw_annotations:
                cv2.putText(
                    output_frame, f"Mode: ALL COLORS ({len(tracked_objects)} detected)",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
                )

        elif mode in [TrackingMode.RED, TrackingMode.GREEN, TrackingMode.BLUE, 
                      TrackingMode.YELLOW, TrackingMode.ORANGE, TrackingMode.PURPLE]:
            color_name = mode.value
            objects, mask = self._track_single_color(frame, hsv_frame, color_name)
            tracked_objects.extend(objects)
            
            if draw_annotations:
                for obj in objects:
                    self._draw_object_overlay(output_frame, obj)
                cv2.putText(
                    output_frame, f"Mode: {color_name.upper()} ({len(objects)} detected)",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.hsv_ranges[color_name]["bgr"], 2, cv2.LINE_AA
                )

        elif mode == TrackingMode.MULTI_SIMULTANEOUS:
            # Track Red, Green, Blue simultaneously
            combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            for color_name in ["Red", "Green", "Blue"]:
                objects, single_mask = self._track_single_color(frame, hsv_frame, color_name)
                tracked_objects.extend(objects)
                if single_mask is not None:
                    combined_mask = cv2.bitwise_or(combined_mask, single_mask)
                    
                if draw_annotations:
                    for obj in objects:
                        self._draw_object_overlay(output_frame, obj)
                        
            mask = combined_mask
            if draw_annotations:
                cv2.putText(
                    output_frame, f"Mode: MULTI-SIMULTANEOUS ({len(tracked_objects)} objects)",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                )

        return output_frame, tracked_objects, mask

    def _track_single_color(
        self,
        frame: np.ndarray,
        hsv_frame: np.ndarray,
        color_name: str
    ) -> Tuple[List[TrackedObject], np.ndarray]:
        """Segments and tracks contours for a specific pre-calibrated color."""
        config = self.hsv_ranges.get(color_name)
        if not config:
            return [], np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        if "lower1" in config and "lower2" in config:
            # Wrapped Hue range (e.g. Red)
            mask1 = cv2.inRange(hsv_frame, config["lower1"], config["upper1"])
            mask2 = cv2.inRange(hsv_frame, config["lower2"], config["upper2"])
            raw_mask = cv2.bitwise_or(mask1, mask2)
        else:
            raw_mask = cv2.inRange(hsv_frame, config["lower"], config["upper"])

        mask = apply_morphological_cleanup(raw_mask, self.kernel_size)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects: List[TrackedObject] = []
        bgr_color = config["bgr"]
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2
                
                objects.append(TrackedObject(
                    label=color_name,
                    bounding_box=(x, y, w, h),
                    centroid=(cx, cy),
                    area=float(area),
                    bgr_color=bgr_color,
                    rgb_color=rgb_color
                ))

        return objects, mask

    def _draw_object_overlay(self, frame: np.ndarray, obj: TrackedObject) -> None:
        """Draws bounding box, center crosshair, and high-visibility text badge."""
        x, y, w, h = obj.bounding_box
        cx, cy = obj.centroid
        
        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), obj.bgr_color, 2)
        
        # Centroid crosshair
        cv2.drawMarker(frame, (cx, cy), obj.bgr_color, cv2.MARKER_CROSS, 14, 2)
        
        # Text label background badge
        label_text = f"{obj.label} ({int(obj.area)}px)"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        badge_y1 = max(0, y - text_h - 10)
        badge_y2 = y
        badge_x1 = x
        badge_x2 = min(frame.shape[1], x + text_w + 10)
        
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), obj.bgr_color, -1)
        
        # Compute WCAG text color for high readability
        text_color = get_contrasting_text_color(obj.rgb_color[0], obj.rgb_color[1], obj.rgb_color[2])
        cv2.putText(
            frame, label_text, (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA
        )
