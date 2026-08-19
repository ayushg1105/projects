"""
Perceptually Uniform Color Classifier Engine.
Uses CIELAB color space and spatial indexing (KD-Tree) for sub-millisecond,
academically and industrially validated nearest color matching.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from config.settings import get_colors_csv_path
from src.utils.logger import app_logger
from src.utils.image_processing import rgb_to_hex


@dataclass
class ColorMatchResult:
    """Represents the structured result of a color classification query."""
    name: str
    hex_code: str
    rgb: Tuple[int, int, int]
    matched_rgb: Tuple[int, int, int]
    delta_e: float
    confidence_score: float  # Scale 0.0 to 1.0 (1.0 = exact match)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "hex_code": self.hex_code,
            "query_rgb": self.rgb,
            "matched_rgb": self.matched_rgb,
            "delta_e": round(self.delta_e, 3),
            "confidence": round(self.confidence_score, 4)
        }


class ColorClassifier:
    """
    Sub-millisecond Color Classifier utilizing KD-Tree spatial indexing over CIELAB color space.
    
    Why CIELAB instead of RGB?
    sRGB Euclidean distances suffer from non-uniform human perceptual sensitivity (e.g. green 
    has higher perceived luminance than blue). CIELAB space models human color vision, where 
    Euclidean distance corresponds directly to perceptual color difference (Delta E 76).
    """

    def __init__(self, dataset_path: Optional[Union[str, Path]] = None, use_lab: bool = True):
        self.use_lab = use_lab
        self.dataset_path = Path(dataset_path) if dataset_path else get_colors_csv_path()
        self.df: pd.DataFrame = pd.DataFrame()
        self.kdtree: Optional[KDTree] = None
        self.points_matrix: Optional[np.ndarray] = None
        
        self._load_dataset()
        self._build_spatial_index()

    def _load_dataset(self) -> None:
        """Loads and sanitizes the 865+ color reference dataset."""
        if not self.dataset_path.exists():
            app_logger.error(f"Color dataset not found at {self.dataset_path}")
            raise FileNotFoundError(f"Missing color reference dataset at {self.dataset_path}")
            
        column_names = ["color_id", "color_name", "hex", "R", "G", "B"]
        self.df = pd.read_csv(self.dataset_path, names=column_names, header=None)
        
        # Clean and validate types
        self.df["R"] = pd.to_numeric(self.df["R"], errors="coerce").fillna(0).astype(int)
        self.df["G"] = pd.to_numeric(self.df["G"], errors="coerce").fillna(0).astype(int)
        self.df["B"] = pd.to_numeric(self.df["B"], errors="coerce").fillna(0).astype(int)
        self.df["color_name"] = self.df["color_name"].astype(str).str.strip()
        self.df["hex"] = self.df["hex"].astype(str).str.strip()
        
        app_logger.info(f"Loaded {len(self.df)} reference colors from {self.dataset_path.name}")

    def _build_spatial_index(self) -> None:
        """
        Builds a KD-Tree in CIELAB or RGB space for O(log N) nearest neighbor search.
        """
        rgb_array = self.df[["R", "G", "B"]].values.astype(np.uint8)
        
        if self.use_lab:
            # Reshape to (N, 1, 3) for OpenCV color conversion, then back to (N, 3)
            rgb_expanded = rgb_array.reshape(-1, 1, 3)
            # Convert RGB to BGR first because cvtColor COLOR_BGR2LAB expects BGR
            bgr_expanded = rgb_expanded[:, :, [2, 1, 0]]
            lab_array = cv2.cvtColor(bgr_expanded, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
            self.points_matrix = lab_array
        else:
            self.points_matrix = rgb_array.astype(np.float32)
            
        self.kdtree = KDTree(self.points_matrix)
        app_logger.debug(f"KD-Tree constructed with {len(self.points_matrix)} nodes (LAB mode={self.use_lab})")

    def classify_rgb(self, r: int, g: int, b: int) -> ColorMatchResult:
        """
        Classifies an input RGB triplet to the closest named color in the dataset.
        
        Args:
            r: Red intensity (0-255)
            g: Green intensity (0-255)
            b: Blue intensity (0-255)
            
        Returns:
            ColorMatchResult object.
        """
        # Clamp inputs
        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))

        if self.use_lab:
            # Convert query point to CIELAB
            query_bgr = np.uint8([[[b, g, r]]])
            query_lab = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2LAB).reshape(1, 3).astype(np.float32)
            dist, idx = self.kdtree.query(query_lab[0], k=1)
        else:
            query_rgb = np.array([r, g, b], dtype=np.float32)
            dist, idx = self.kdtree.query(query_rgb, k=1)

        row = self.df.iloc[idx]
        matched_r, matched_g, matched_b = int(row["R"]), int(row["G"]), int(row["B"])
        
        # Calculate Delta E (perceptual distance)
        delta_e = float(dist)
        # Normalize confidence (Delta E <= 2.3 is JND - Just Noticeable Difference)
        confidence = max(0.0, 1.0 - (delta_e / 100.0))

        return ColorMatchResult(
            name=row["color_name"],
            hex_code=row["hex"] if str(row["hex"]).startswith("#") else f"#{row['hex']}",
            rgb=(r, g, b),
            matched_rgb=(matched_r, matched_g, matched_b),
            delta_e=delta_e,
            confidence_score=confidence
        )

    def classify_bgr(self, b: int, g: int, r: int) -> ColorMatchResult:
        """Convenience method for OpenCV BGR order inputs."""
        return self.classify_rgb(r, g, b)

    def batch_classify_rgb(self, rgb_array: np.ndarray) -> List[ColorMatchResult]:
        """
        Classifies multiple RGB points in parallel using KDTree vectorized batch queries.
        
        Args:
            rgb_array: Shape (N, 3) with R, G, B order.
        """
        if len(rgb_array) == 0:
            return []
            
        rgb_clipped = np.clip(rgb_array, 0, 255).astype(np.uint8)
        
        if self.use_lab:
            bgr_expanded = rgb_clipped[:, [2, 1, 0]].reshape(-1, 1, 3)
            query_points = cv2.cvtColor(bgr_expanded, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        else:
            query_points = rgb_clipped.astype(np.float32)
            
        distances, indices = self.kdtree.query(query_points, k=1)
        
        results = []
        for i in range(len(rgb_clipped)):
            idx = indices[i]
            dist = float(distances[i])
            row = self.df.iloc[idx]
            r, g, b = rgb_clipped[i]
            matched_r, matched_g, matched_b = int(row["R"]), int(row["G"]), int(row["B"])
            confidence = max(0.0, 1.0 - (dist / 100.0))
            
            results.append(ColorMatchResult(
                name=row["color_name"],
                hex_code=row["hex"] if str(row["hex"]).startswith("#") else f"#{row['hex']}",
                rgb=(int(r), int(g), int(b)),
                matched_rgb=(matched_r, matched_g, matched_b),
                delta_e=dist,
                confidence_score=confidence
            ))
            
        return results
