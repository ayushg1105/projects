"""
Unit Tests for ColorMaskEngine and Multi-Channel Isolation.
"""

import numpy as np
import pytest
from src.core.color_masker import ColorMaskEngine, MaskFilterType, MaskResult


@pytest.fixture(scope="module")
def mask_engine():
    return ColorMaskEngine()


def test_color_mask_blue_synthetic(mask_engine):
    # 100x100 image, left half blue (BGR: 255, 0, 0), right half black
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :50] = (255, 0, 0)

    result = mask_engine.apply_filter(image, filter_type=MaskFilterType.BLUE)
    assert isinstance(result, MaskResult)
    assert result.pixel_count > 0
    assert 45.0 <= result.coverage_percentage <= 55.0
    assert result.binary_mask.shape == (100, 100)


def test_color_mask_red_synthetic(mask_engine):
    # 100x100 image, right half red (BGR: 0, 0, 255)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, 50:] = (0, 0, 255)

    result = mask_engine.apply_filter(image, filter_type=MaskFilterType.RED)
    assert isinstance(result, MaskResult)
    assert result.pixel_count > 0
    assert 45.0 <= result.coverage_percentage <= 55.0


def test_custom_hsv_tuning(mask_engine):
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    image[:, :] = (0, 255, 0)  # Pure Green

    # Custom HSV bounds around Green (H: 35-85)
    custom_lower = np.array([35, 50, 50], dtype=np.uint8)
    custom_upper = np.array([85, 255, 255], dtype=np.uint8)

    result = mask_engine.apply_filter(
        image,
        filter_type=MaskFilterType.CUSTOM_HSV,
        custom_lower=custom_lower,
        custom_upper=custom_upper
    )
    assert result.coverage_percentage > 95.0


def test_multi_grid_view_generation(mask_engine):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    grid = mask_engine.generate_multi_grid_view(image)
    assert grid.ndim == 3
    assert grid.shape[0] == 120
    assert grid.shape[1] == 160
