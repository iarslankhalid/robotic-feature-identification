"""
Feature Detection Module
=========================
Classical computer vision algorithms for detecting geometric affordances:
  - Apertures (holes) via Hough Circle Transform
  - Planar surfaces via contour analysis
  - Handles/protrusions via Canny edge detection + morphological ops
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectedFeature:
    """A single detected geometric feature."""
    feature_type: str          # "hole", "surface", "handle"
    pixel_coords: tuple        # (u, v) center in image
    pixel_region: list         # list of pixel coords or bounding rect
    confidence: float = 0.0    # detection confidence [0, 1]
    radius: Optional[float] = None    # for circular features
    area: Optional[float] = None      # for surface features
    contour: Optional[np.ndarray] = None  # raw contour if applicable


class FeatureDetector:
    """Detect geometric affordances in RGB images."""

    def __init__(self, config=None):
        """
        Initialize with optional configuration overrides.

        Args:
            config: Dict of parameter overrides for detection thresholds.
        """
        defaults = {
            # Preprocessing
            "blur_kernel": 5,
            "clahe_clip": 2.0,
            "clahe_grid": (8, 8),

            # Hough Circles (aperture detection)
            "hough_dp": 1.2,
            "hough_min_dist": 30,
            "hough_param1": 50,
            "hough_param2": 30,
            "hough_min_radius": 5,
            "hough_max_radius": 80,

            # Surface detection
            "surface_min_area": 500,
            "surface_max_area": 50000,
            "surface_approx_eps": 0.02,  # contour approximation epsilon factor

            # Edge / handle detection
            "canny_low": 50,
            "canny_high": 150,
            "morph_kernel": (5, 5),
            "handle_min_area": 200,
            "handle_max_area": 10000,
            "handle_aspect_ratio_min": 1.5,  # elongated shapes
        }
        if config:
            defaults.update(config)
        self.cfg = defaults

    # ------------------------------------------------------------------ #
    #  Preprocessing
    # ------------------------------------------------------------------ #

    def preprocess(self, rgb_image):
        """
        Convert to grayscale, apply CLAHE and Gaussian blur.

        Returns:
            gray: Grayscale image (uint8)
            blurred: Blurred grayscale image (uint8)
        """
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        # Adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(
            clipLimit=self.cfg["clahe_clip"],
            tileGridSize=self.cfg["clahe_grid"],
        )
        gray = clahe.apply(gray)

        blurred = cv2.GaussianBlur(
            gray,
            (self.cfg["blur_kernel"], self.cfg["blur_kernel"]),
            0,
        )
        return gray, blurred

    # ------------------------------------------------------------------ #
    #  Aperture Detection (Holes)
    # ------------------------------------------------------------------ #

    def detect_holes(self, rgb_image):
        """
        Detect circular apertures using the Hough Circle Transform.

        Args:
            rgb_image: RGB image as np.ndarray (H, W, 3).

        Returns:
            List of DetectedFeature with feature_type="hole".
        """
        _, blurred = self.preprocess(rgb_image)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.cfg["hough_dp"],
            minDist=self.cfg["hough_min_dist"],
            param1=self.cfg["hough_param1"],
            param2=self.cfg["hough_param2"],
            minRadius=self.cfg["hough_min_radius"],
            maxRadius=self.cfg["hough_max_radius"],
        )

        features = []
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            for cx, cy, r in circles:
                features.append(DetectedFeature(
                    feature_type="hole",
                    pixel_coords=(int(cx), int(cy)),
                    pixel_region=[(int(cx), int(cy), int(r))],
                    confidence=min(1.0, r / self.cfg["hough_max_radius"]),
                    radius=float(r),
                ))
        return features

    # ------------------------------------------------------------------ #
    #  Surface Detection (Flat planar regions)
    # ------------------------------------------------------------------ #

    def detect_surfaces(self, rgb_image):
        """
        Detect planar surfaces via contour analysis and area thresholding.

        Args:
            rgb_image: RGB image as np.ndarray (H, W, 3).

        Returns:
            List of DetectedFeature with feature_type="surface".
        """
        gray, blurred = self.preprocess(rgb_image)

        # Adaptive threshold for surface segmentation
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2,
        )

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        features = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.cfg["surface_min_area"] or area > self.cfg["surface_max_area"]:
                continue

            # Approximate contour
            eps = self.cfg["surface_approx_eps"] * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)

            # Accept roughly rectangular shapes (4-6 vertices)
            if 4 <= len(approx) <= 6:
                moments = cv2.moments(cnt)
                if moments["m00"] == 0:
                    continue
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                x, y, w, h = cv2.boundingRect(cnt)

                features.append(DetectedFeature(
                    feature_type="surface",
                    pixel_coords=(cx, cy),
                    pixel_region=[(x, y, w, h)],
                    confidence=min(1.0, area / self.cfg["surface_max_area"]),
                    area=float(area),
                    contour=cnt,
                ))
        return features

    # ------------------------------------------------------------------ #
    #  Handle / Protrusion Detection
    # ------------------------------------------------------------------ #

    def detect_handles(self, rgb_image):
        """
        Detect elongated protrusions (handles) using Canny edges + morphology.

        Args:
            rgb_image: RGB image as np.ndarray (H, W, 3).

        Returns:
            List of DetectedFeature with feature_type="handle".
        """
        gray, _ = self.preprocess(rgb_image)

        # Canny edge detection
        edges = cv2.Canny(gray, self.cfg["canny_low"], self.cfg["canny_high"])

        # Morphological closing to connect fragmented edges
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, self.cfg["morph_kernel"]
        )
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        features = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.cfg["handle_min_area"] or area > self.cfg["handle_max_area"]:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

            # Handles are elongated
            if aspect_ratio >= self.cfg["handle_aspect_ratio_min"]:
                moments = cv2.moments(cnt)
                if moments["m00"] == 0:
                    continue
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

                features.append(DetectedFeature(
                    feature_type="handle",
                    pixel_coords=(cx, cy),
                    pixel_region=[(x, y, w, h)],
                    confidence=min(1.0, aspect_ratio / 5.0),
                    area=float(area),
                    contour=cnt,
                ))
        return features

    # ------------------------------------------------------------------ #
    #  Combined detection
    # ------------------------------------------------------------------ #

    def detect_all(self, rgb_image):
        """
        Run all detection algorithms and return combined results.

        Returns:
            Dict mapping feature type to list of DetectedFeature.
        """
        return {
            "holes": self.detect_holes(rgb_image),
            "surfaces": self.detect_surfaces(rgb_image),
            "handles": self.detect_handles(rgb_image),
        }
