"""
Evaluation Metrics Module
==========================
Quantitative metrics for comparing detected features against ground truth.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import cv2


# Pixel distance threshold for matching a detection to a ground truth
MATCH_THRESHOLD_PX = 40.0


def detection_rate(detections: List, ground_truths: List) -> float:
    """Compute the fraction of ground truth features that were detected.

    A detection matches a ground truth if its pixel distance is within
    MATCH_THRESHOLD_PX pixels of the projected ground truth position.

    Args:
        detections: List of (u, v) pixel tuples for detected features.
        ground_truths: List of (u, v) pixel tuples for ground-truth projections.

    Returns:
        Float in [0, 1] — matched_gt / total_gt.  Returns 1.0 if ground_truths is empty.
    """
    if not ground_truths:
        return 1.0

    matched = 0
    for gt in ground_truths:
        gt_arr = np.array(gt[:2], dtype=np.float64)
        for det in detections:
            det_arr = np.array(det[:2], dtype=np.float64)
            if np.linalg.norm(det_arr - gt_arr) < MATCH_THRESHOLD_PX:
                matched += 1
                break  # each GT is matched at most once

    return matched / len(ground_truths)


def pixel_localization_error(
    detected_pixel: Tuple[float, float],
    ground_truth_world: List[float],
    transformer: Any,
) -> float:
    """Euclidean pixel distance between a detection and its ground truth projection.

    Args:
        detected_pixel: (u, v) pixel coords of the detection.
        ground_truth_world: [x, y, z] world coordinates of ground truth.
        transformer: CoordinateTransformer instance with world_to_pixel().

    Returns:
        Float — L2 pixel distance.  Returns float('inf') if projection fails.
    """
    result = transformer.world_to_pixel(ground_truth_world)
    if result is None:
        return float("inf")
    gt_u, gt_v, _ = result
    du = detected_pixel[0] - gt_u
    dv = detected_pixel[1] - gt_v
    return float(np.sqrt(du * du + dv * dv))


def position_error_3d(
    detected_world: List[float],
    ground_truth_world: List[float],
) -> float:
    """Euclidean distance in 3D between detected and ground truth world coordinates.

    Args:
        detected_world: [x, y, z] of the detected feature.
        ground_truth_world: [x, y, z] of the ground truth.

    Returns:
        Float — Euclidean distance in meters.
    """
    d = np.array(detected_world, dtype=np.float64)
    g = np.array(ground_truth_world, dtype=np.float64)
    return float(np.linalg.norm(d - g))


def per_axis_mae(
    detected_world: List[float],
    ground_truth_world: List[float],
) -> Dict[str, float]:
    """Mean absolute error per axis between detected and ground truth positions.

    Args:
        detected_world: [x, y, z] of the detected feature.
        ground_truth_world: [x, y, z] of the ground truth.

    Returns:
        Dict with keys "x", "y", "z" — absolute error per axis in meters.
    """
    d = np.array(detected_world, dtype=np.float64)
    g = np.array(ground_truth_world, dtype=np.float64)
    diff = np.abs(d - g)
    return {"x": float(diff[0]), "y": float(diff[1]), "z": float(diff[2])}


def circle_iou(
    detected_center: Tuple[float, float],
    detected_radius: float,
    gt_center: Tuple[float, float],
    gt_radius: float,
    image_shape: Tuple[int, int],
) -> float:
    """Intersection over Union between two circles on the image plane.

    Renders both circles as binary masks and computes IoU.

    Args:
        detected_center: (u, v) center of the detected circle.
        detected_radius: Radius of the detected circle in pixels.
        gt_center: (u, v) center of the ground-truth circle.
        gt_radius: Radius of the ground-truth circle in pixels.
        image_shape: (height, width) of the image.

    Returns:
        Float in [0, 1] — IoU score.
    """
    h, w = image_shape
    mask_det = np.zeros((h, w), dtype=np.uint8)
    mask_gt = np.zeros((h, w), dtype=np.uint8)

    cv2.circle(mask_det, (int(detected_center[0]), int(detected_center[1])),
               int(detected_radius), 1, -1)
    cv2.circle(mask_gt, (int(gt_center[0]), int(gt_center[1])),
               int(gt_radius), 1, -1)

    intersection = np.logical_and(mask_det, mask_gt).sum()
    union = np.logical_or(mask_det, mask_gt).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


def processing_latency(func: Callable, *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """Time a function call in milliseconds.

    Args:
        func: Callable to time.
        *args: Positional arguments forwarded to func.
        **kwargs: Keyword arguments forwarded to func.

    Returns:
        Tuple of (result, elapsed_ms) where elapsed_ms is a float.
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms
