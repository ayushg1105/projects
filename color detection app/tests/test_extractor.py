"""
Unit Tests for DominantColorExtractor and K-Means Clustering.
"""

import numpy as np
import pytest
from src.core.color_classifier import ColorClassifier
from src.core.color_extractor import DominantColorExtractor, PaletteExtractionResult


@pytest.fixture(scope="module")
def extractor():
    classifier = ColorClassifier()
    return DominantColorExtractor(classifier)


def test_palette_extraction_synthetic(extractor):
    # Create synthetic test image: 100x100 pixels, left half pure red (BGR: 0, 0, 255), right half pure blue (BGR: 255, 0, 0)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :50] = (0, 0, 255)  # Red in BGR
    image[:, 50:] = (255, 0, 0)  # Blue in BGR

    result = extractor.extract_palette(image, k=2)

    assert isinstance(result, PaletteExtractionResult)
    assert len(result.dominant_colors) == 2
    
    # Check that proportions sum up to roughly 100%
    total_pct = sum(c.percentage for c in result.dominant_colors)
    assert 98.0 <= total_pct <= 102.0

    # Verify palette swatch image shape
    assert result.palette_image.ndim == 3
    assert result.palette_image.shape[2] == 3


def test_palette_k_bounds(extractor):
    image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    
    # Test k clamp
    res_k_high = extractor.extract_palette(image, k=20)
    assert len(res_k_high.dominant_colors) <= 10
    
    res_k_low = extractor.extract_palette(image, k=1)
    assert len(res_k_low.dominant_colors) >= 2
