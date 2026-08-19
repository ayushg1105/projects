"""
Unsupervised Machine Learning Color Extractor.
Uses K-Means Clustering to extract dominant color palettes, percentages, and color harmonies from images.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from config.settings import ML_CONFIG
from src.core.color_classifier import ColorClassifier, ColorMatchResult
from src.utils.logger import app_logger
from src.utils.image_processing import rgb_to_hex


@dataclass
class DominantColor:
    """Represents a single dominant cluster color in an image."""
    rgb: Tuple[int, int, int]
    hex_code: str
    percentage: float
    color_name: str
    match_result: ColorMatchResult


@dataclass
class PaletteExtractionResult:
    """Represents the complete extracted palette for an image."""
    dominant_colors: List[DominantColor]
    total_pixels_analyzed: int
    palette_image: np.ndarray  # Visual horizontal swatch image (BGR)


class DominantColorExtractor:
    """
    Extracts dominant color palettes from static images or video regions using K-Means clustering.
    """

    def __init__(self, classifier: Optional[ColorClassifier] = None):
        self.classifier = classifier or ColorClassifier()

    def extract_palette(
        self,
        image_bgr: np.ndarray,
        k: int = ML_CONFIG.DEFAULT_K_CLUSTERS,
        sample_size: int = ML_CONFIG.KMEANS_SAMPLE_SIZE
    ) -> PaletteExtractionResult:
        """
        Extracts top K dominant colors using MiniBatchKMeans for high-speed clustering.
        
        Args:
            image_bgr: Input image in OpenCV BGR format.
            k: Number of dominant color clusters (2 to 10).
            sample_size: Maximum random pixel sample size for fast clustering.
            
        Returns:
            PaletteExtractionResult containing dominant colors sorted by frequency and swatch image.
        """
        k = max(ML_CONFIG.MIN_K_CLUSTERS, min(ML_CONFIG.MAX_K_CLUSTERS, k))
        
        # Convert BGR to RGB for analysis
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pixels = image_rgb.reshape(-1, 3)
        total_pixels = len(pixels)

        # Subsample pixels for faster clustering if image is large
        if total_pixels > sample_size:
            indices = np.random.RandomState(ML_CONFIG.KMEANS_RANDOM_STATE).choice(
                total_pixels, sample_size, replace=False
            )
            sampled_pixels = pixels[indices]
        else:
            sampled_pixels = pixels

        # Fit K-Means
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=ML_CONFIG.KMEANS_RANDOM_STATE,
            batch_size=512,
            n_init="auto"
        )
        labels = kmeans.fit_predict(sampled_pixels)
        cluster_centers = kmeans.cluster_centers_.astype(int)

        # Count frequencies of each cluster
        counts = np.bincount(labels, minlength=k)
        total_counts = len(labels)
        proportions = counts / total_counts

        # Sort clusters from highest proportion to lowest
        sorted_indices = np.argsort(proportions)[::-1]

        dominant_colors: List[DominantColor] = []
        for idx in sorted_indices:
            r, g, b = cluster_centers[idx]
            match_res = self.classifier.classify_rgb(r, g, b)
            percentage = float(proportions[idx] * 100.0)
            
            dominant_colors.append(DominantColor(
                rgb=(int(r), int(g), int(b)),
                hex_code=rgb_to_hex(r, g, b),
                percentage=round(percentage, 2),
                color_name=match_res.name,
                match_result=match_res
            ))

        # Generate visual swatch bar (BGR)
        palette_swatch = self._generate_swatch_bar(dominant_colors, width=600, height=80)

        app_logger.debug(f"Extracted {k} dominant colors successfully.")
        return PaletteExtractionResult(
            dominant_colors=dominant_colors,
            total_pixels_analyzed=len(sampled_pixels),
            palette_image=palette_swatch
        )

    def _generate_swatch_bar(
        self,
        dominant_colors: List[DominantColor],
        width: int = 600,
        height: int = 80
    ) -> np.ndarray:
        """Generates a visual horizontal swatch strip scaled to cluster proportions."""
        swatch = np.zeros((height, width, 3), dtype=np.uint8)
        current_x = 0

        for color_info in dominant_colors:
            # Width proportional to percentage
            seg_width = int(round((color_info.percentage / 100.0) * width))
            end_x = min(width, current_x + seg_width)
            
            # BGR format for OpenCV
            r, g, b = color_info.rgb
            bgr_color = (b, g, r)
            
            swatch[:, current_x:end_x] = bgr_color
            current_x = end_x

        # Fill remaining pixels if any rounding gaps
        if current_x < width and len(dominant_colors) > 0:
            last_r, last_g, last_b = dominant_colors[-1].rgb
            swatch[:, current_x:width] = (last_b, last_g, last_r)

        return swatch
