"""
Unit Tests for ColorClassifier and Perceptual CIELAB KD-Tree Indexing.
"""

import numpy as np
import pytest
from src.core.color_classifier import ColorClassifier, ColorMatchResult


@pytest.fixture(scope="module")
def classifier():
    return ColorClassifier()


def test_dataset_loaded(classifier):
    assert classifier.df is not None
    assert len(classifier.df) >= 800
    assert "color_name" in classifier.df.columns
    assert "hex" in classifier.df.columns


def test_classify_pure_red(classifier):
    result = classifier.classify_rgb(255, 0, 0)
    assert isinstance(result, ColorMatchResult)
    assert "Red" in result.name or "red" in result.name.lower()
    assert result.delta_e >= 0.0
    assert 0.0 <= result.confidence_score <= 1.0


def test_classify_pure_green(classifier):
    result = classifier.classify_rgb(0, 255, 0)
    assert isinstance(result, ColorMatchResult)
    assert "green" in result.name.lower() or result.hex_code.lower() in ["#00ff00", "#00ff00".lower()]


def test_classify_pure_blue(classifier):
    result = classifier.classify_rgb(0, 0, 255)
    assert isinstance(result, ColorMatchResult)
    assert "blue" in result.name.lower() or result.hex_code.lower() in ["#0000ff"]


def test_batch_classification(classifier):
    sample_rgb = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 255],
        [0, 0, 0]
    ], dtype=np.uint8)

    results = classifier.batch_classify_rgb(sample_rgb)
    assert len(results) == 5
    for res in results:
        assert isinstance(res, ColorMatchResult)
        assert res.hex_code.startswith("#")
        assert len(res.rgb) == 3


def test_rgb_bounds_clamping(classifier):
    # Test values outside standard 0-255 range to ensure robust handling
    result_high = classifier.classify_rgb(300, 300, 300)
    assert result_high is not None
    result_low = classifier.classify_rgb(-50, -50, -50)
    assert result_low is not None
