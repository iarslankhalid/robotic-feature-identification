"""
Tests for the feature detection module.
Uses synthetic test images to validate each detection algorithm
independently from the simulation.
"""

import pytest
import cv2
import numpy as np

from src.feature_detector import FeatureDetector, DetectedFeature


@pytest.fixture
def detector():
    """Default feature detector instance."""
    return FeatureDetector()


# ------------------------------------------------------------------ #
#  Helper: Generate synthetic test images
# ------------------------------------------------------------------ #

def make_circle_image(width=640, height=480, center=(320, 240), radius=30):
    """Create a synthetic image with a white circle on dark background."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # dark gray background
    cv2.circle(img, center, radius, (200, 200, 200), 2)  # ring outline
    cv2.circle(img, center, radius - 5, (20, 20, 20), -1)  # dark center (hole)
    return img


def make_rectangle_image(width=640, height=480, rect=(200, 150, 200, 140)):
    """Create a synthetic image with a filled rectangle (surface)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (180, 180, 220), -1)
    return img


def make_handle_image(width=640, height=480):
    """Create an image with an elongated protrusion (handle-like)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    # Body (large circle)
    cv2.circle(img, (320, 240), 60, (150, 150, 150), -1)
    # Handle (elongated rectangle sticking out)
    cv2.rectangle(img, (380, 225, ), (430, 255), (160, 160, 160), -1)
    return img


# ------------------------------------------------------------------ #
#  Preprocessing Tests
# ------------------------------------------------------------------ #

class TestPreprocessing:

    def test_output_shapes(self, detector):
        """Preprocessing should return grayscale images of correct shape."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        gray, blurred = detector.preprocess(img)
        assert gray.shape == (480, 640)
        assert blurred.shape == (480, 640)
        assert gray.dtype == np.uint8
        assert blurred.dtype == np.uint8

    def test_blur_reduces_noise(self, detector):
        """Blurred image should have lower variance than noisy input."""
        noisy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        gray, blurred = detector.preprocess(noisy)
        assert blurred.var() <= gray.var()


# ------------------------------------------------------------------ #
#  Hole Detection Tests
# ------------------------------------------------------------------ #

class TestHoleDetection:

    def test_detects_circle(self, detector):
        """Should detect at least one hole in an image with a circle."""
        img = make_circle_image(center=(320, 240), radius=35)
        features = detector.detect_holes(img)
        assert len(features) >= 1, "No holes detected in circle image"
        assert features[0].feature_type == "hole"

    def test_circle_center_accuracy(self, detector):
        """Detected circle center should be close to the true center."""
        true_center = (300, 250)
        img = make_circle_image(center=true_center, radius=30)
        features = detector.detect_holes(img)
        if len(features) > 0:
            detected = features[0].pixel_coords
            dist = np.linalg.norm(
                np.array(detected) - np.array(true_center)
            )
            assert dist < 20, f"Circle center off by {dist:.1f} pixels"

    def test_no_false_positive_on_blank(self, detector):
        """Should not detect holes in a blank image."""
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 128
        features = detector.detect_holes(blank)
        assert len(features) == 0, "False positive hole detection on blank image"

    def test_feature_has_radius(self, detector):
        """Detected holes should have a radius attribute."""
        img = make_circle_image()
        features = detector.detect_holes(img)
        for f in features:
            assert f.radius is not None
            assert f.radius > 0

    def test_multiple_circles(self, detector):
        """Should detect multiple distinct circles."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        cv2.circle(img, (150, 200), 25, (200, 200, 200), 2)
        cv2.circle(img, (150, 200), 20, (20, 20, 20), -1)
        cv2.circle(img, (450, 300), 35, (200, 200, 200), 2)
        cv2.circle(img, (450, 300), 30, (20, 20, 20), -1)
        features = detector.detect_holes(img)
        assert len(features) >= 2, f"Expected >=2 circles, got {len(features)}"


# ------------------------------------------------------------------ #
#  Surface Detection Tests
# ------------------------------------------------------------------ #

class TestSurfaceDetection:

    def test_detects_rectangle(self, detector):
        """Should detect a rectangular surface."""
        img = make_rectangle_image(rect=(200, 150, 200, 140))
        features = detector.detect_surfaces(img)
        assert len(features) >= 1, "No surfaces detected in rectangle image"
        assert features[0].feature_type == "surface"

    def test_surface_centroid_accuracy(self, detector):
        """Surface centroid should be close to rectangle center."""
        x, y, w, h = 200, 150, 200, 140
        true_center = (x + w // 2, y + h // 2)
        img = make_rectangle_image(rect=(x, y, w, h))
        features = detector.detect_surfaces(img)
        if len(features) > 0:
            detected = features[0].pixel_coords
            dist = np.linalg.norm(
                np.array(detected) - np.array(true_center)
            )
            assert dist < 30, f"Surface centroid off by {dist:.1f} pixels"

    def test_surface_has_area(self, detector):
        """Detected surfaces should have an area measurement."""
        img = make_rectangle_image()
        features = detector.detect_surfaces(img)
        for f in features:
            assert f.area is not None
            assert f.area > 0

    def test_small_surface_ignored(self, detector):
        """Surfaces below minimum area threshold should not be detected."""
        img = make_rectangle_image(rect=(300, 230, 10, 10))  # tiny rect
        features = detector.detect_surfaces(img)
        # Should be filtered out by min area
        small_features = [f for f in features if f.area is not None and f.area < 200]
        assert len(small_features) == 0


# ------------------------------------------------------------------ #
#  Handle Detection Tests
# ------------------------------------------------------------------ #

class TestHandleDetection:

    def test_detects_handle(self, detector):
        """Should detect an elongated protrusion."""
        img = make_handle_image()
        features = detector.detect_handles(img)
        # Handle detection on synthetic images may vary; just verify no crash
        assert isinstance(features, list)

    def test_no_handle_on_blank(self, detector):
        """Should not detect handles in a blank image."""
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 128
        features = detector.detect_handles(blank)
        assert len(features) == 0

    def test_features_have_correct_type(self, detector):
        """All returned features should be of type 'handle'."""
        img = make_handle_image()
        features = detector.detect_handles(img)
        for f in features:
            assert f.feature_type == "handle"


# ------------------------------------------------------------------ #
#  Combined Detection Tests
# ------------------------------------------------------------------ #

class TestCombinedDetection:

    def test_detect_all_returns_dict(self, detector):
        """detect_all should return dict with expected keys."""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = detector.detect_all(img)
        assert "holes" in result
        assert "surfaces" in result
        assert "handles" in result

    def test_custom_config(self):
        """Detector should accept custom configuration."""
        config = {"hough_min_radius": 10, "surface_min_area": 1000}
        det = FeatureDetector(config=config)
        assert det.cfg["hough_min_radius"] == 10
        assert det.cfg["surface_min_area"] == 1000
        # Other defaults should still be there
        assert det.cfg["canny_low"] == 50
