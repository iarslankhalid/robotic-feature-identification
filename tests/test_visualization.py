"""
Tests for the visualization module.
"""

import os
import tempfile

import numpy as np
import pytest

from src.feature_detector import DetectedFeature
from src.visualization import (
    create_summary_figure,
    draw_approach_vectors,
    draw_features_on_image,
)


def _make_rgb(h=480, w=640):
    """Create a simple synthetic RGB image for testing."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[100:200, 100:200] = [128, 200, 64]
    return img


def _make_depth(h=480, w=640):
    """Create a simple depth image for testing."""
    depth = np.ones((h, w), dtype=np.float32) * 0.5
    return depth


def _make_features():
    """Create a minimal features_dict for testing."""
    return {
        "holes": [
            DetectedFeature(
                feature_type="hole",
                pixel_coords=(160, 240),
                pixel_region=[(160, 240, 20)],
                confidence=0.75,
                radius=20.0,
            )
        ],
        "surfaces": [
            DetectedFeature(
                feature_type="surface",
                pixel_coords=(320, 240),
                pixel_region=[(270, 190, 100, 100)],
                confidence=0.6,
                area=10000.0,
            )
        ],
        "handles": [
            DetectedFeature(
                feature_type="handle",
                pixel_coords=(500, 300),
                pixel_region=[(480, 280, 60, 20)],
                confidence=0.5,
                area=1200.0,
            )
        ],
    }


class TestDrawFeaturesOnImage:
    def test_returns_rgb_array(self):
        rgb = _make_rgb()
        features = _make_features()
        result = draw_features_on_image(rgb, features)
        assert isinstance(result, np.ndarray)

    def test_correct_shape(self):
        rgb = _make_rgb()
        features = _make_features()
        result = draw_features_on_image(rgb, features)
        assert result.shape == rgb.shape

    def test_correct_dtype(self):
        rgb = _make_rgb()
        features = _make_features()
        result = draw_features_on_image(rgb, features)
        assert result.dtype == np.uint8

    def test_empty_features(self):
        rgb = _make_rgb()
        result = draw_features_on_image(rgb, {"holes": [], "surfaces": [], "handles": []})
        assert result.shape == rgb.shape

    def test_does_not_modify_original(self):
        rgb = _make_rgb()
        original = rgb.copy()
        draw_features_on_image(rgb, _make_features())
        np.testing.assert_array_equal(rgb, original)

    def test_with_world_coords(self):
        rgb = _make_rgb()
        features = _make_features()
        world_coords = {
            "holes": [np.array([0.1, -0.2, 0.45])],
            "surfaces": [np.array([0.0, 0.0, 0.45])],
            "handles": [np.array([0.15, 0.1, 0.48])],
        }
        result = draw_features_on_image(rgb, features, world_coords)
        assert result.shape == rgb.shape


class TestDrawApproachVectors:
    def test_no_crash_empty_features(self):
        """draw_approach_vectors should not crash on empty feature lists."""
        import cv2
        img_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_approach_vectors(img_bgr, {"holes": [], "surfaces": [], "handles": []})
        assert result.shape == (480, 640, 3)

    def test_no_crash_with_features(self):
        import cv2
        img_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        features = _make_features()
        result = draw_approach_vectors(img_bgr, features)
        assert isinstance(result, np.ndarray)

    def test_returns_same_array(self):
        img_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_approach_vectors(img_bgr, {"holes": [], "surfaces": [], "handles": []})
        assert result is img_bgr  # in-place modification


class TestCreateSummaryFigure:
    def test_produces_file_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "summary.png")
            rgb = _make_rgb()
            depth = _make_depth()
            features = _make_features()
            create_summary_figure(rgb, depth, features, output_path=out_path)
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0

    def test_no_crash_empty_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "summary_empty.png")
            rgb = _make_rgb()
            depth = _make_depth()
            create_summary_figure(
                rgb, depth,
                {"holes": [], "surfaces": [], "handles": []},
                output_path=out_path,
            )
            assert os.path.exists(out_path)
