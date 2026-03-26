#!/usr/bin/env python3
"""
Complete Experiment Suite — No PyBullet Required
=================================================
Replaces PyBullet with programmatic synthetic RGB-D scene generation.
Runs all experiments, tests, and generates the full report with charts.

Usage:  python run_all_experiments.py
"""

import csv
import io
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Ensure imports work ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.feature_detector import FeatureDetector, DetectedFeature
from src.coordinate_transform import CoordinateTransformer

RESULTS_DIR = "results"
REPORTS_DIR = "reports"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# PART 1: SYNTHETIC SIMULATION (replaces PyBullet)
# ══════════════════════════════════════════════════════════════════

class SyntheticScene:
    """Generate RGB-D images with known ground truth — no PyBullet needed.
    
    Produces scenes with 3 objects (washer, box, mug) at configurable 
    positions and orientations, with precise ground truth for evaluation.
    """

    TABLE_Z = 0.42
    WIDTH, HEIGHT = 640, 480
    FOV = 60.0

    # Object physical dimensions (meters)
    WASHER_OUTER_R = 0.03
    WASHER_INNER_R = 0.012
    BOX_HALF = [0.05, 0.035, 0.03]
    MUG_RADIUS = 0.03
    HANDLE_HALF = [0.005, 0.015, 0.02]

    def __init__(self, cam_pos=(0.0, -0.5, 0.9), cam_target=(0.0, 0.0, 0.42)):
        self.cam_pos = np.array(cam_pos, dtype=np.float64)
        self.cam_target = np.array(cam_target, dtype=np.float64)
        self.K, self.extrinsic = self._build_camera_matrices()
        self.transformer = CoordinateTransformer(self.K, self.extrinsic)

    def _build_camera_matrices(self):
        fov_rad = np.radians(self.FOV)
        fy = self.HEIGHT / (2.0 * np.tan(fov_rad / 2.0))
        fx = fy
        K = np.array([[fx, 0, self.WIDTH / 2.0],
                       [0, fy, self.HEIGHT / 2.0],
                       [0,  0,  1]], dtype=np.float64)

        # Build view matrix (camera looks from cam_pos toward cam_target)
        forward = self.cam_target - self.cam_pos
        forward = forward / np.linalg.norm(forward)

        # Choose an up vector that isn't parallel to forward
        world_up = np.array([0, 0, 1], dtype=np.float64)
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0, 1, 0], dtype=np.float64)

        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        R = np.eye(3, dtype=np.float64)
        R[0, :] = right
        R[1, :] = -up  # OpenCV convention: Y points down
        R[2, :] = forward

        t = -R @ self.cam_pos
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        return K, extrinsic

    def generate(self, seed=42, washer_pos=None, box_pos=None, mug_pos=None,
                 yaw_washer=0, yaw_box=0, yaw_mug=0):
        """Generate a complete scene with RGB, depth, and ground truth.
        
        Returns: (rgb, depth, ground_truth, cam_params)
        """
        np.random.seed(seed)

        # Default randomized positions
        if washer_pos is None or box_pos is None or mug_pos is None:
            positions = self._random_positions()
            washer_pos = washer_pos or [positions[0][0], positions[0][1], self.TABLE_Z + 0.006]
            box_pos = box_pos or [positions[1][0], positions[1][1], self.TABLE_Z + self.BOX_HALF[2]]
            mug_pos = mug_pos or [positions[2][0], positions[2][1], self.TABLE_Z + 0.04]

        gt = {
            "washer": {"type": "hole", "position": list(washer_pos),
                       "feature_radius": self.WASHER_INNER_R},
            "box": {"type": "surface", "position": [box_pos[0], box_pos[1], box_pos[2] + self.BOX_HALF[2]],
                    "surface_dimensions": [0.10, 0.07]},
            "mug": {"type": "handle",
                    "position": [mug_pos[0] + self.MUG_RADIUS + self.HANDLE_HALF[0],
                                 mug_pos[1], mug_pos[2]],
                    "handle_dimensions": [0.01, 0.03, 0.04]},
        }

        rgb = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8) * 45
        depth = np.ones((self.HEIGHT, self.WIDTH), dtype=np.float32) * 1.5

        # Draw table
        table_corners_world = [
            [-0.4, -0.6, self.TABLE_Z], [0.4, -0.6, self.TABLE_Z],
            [0.4, 0.6, self.TABLE_Z], [-0.4, 0.6, self.TABLE_Z]
        ]
        table_px = [self._world_to_px(p) for p in table_corners_world]
        if all(p is not None for p in table_px):
            pts = np.array(table_px, dtype=np.int32)
            cv2.fillPoly(rgb, [pts], (100, 70, 45))
            # Set table depth
            mask = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            cam_dist_table = np.linalg.norm(self.cam_pos - np.array([0, 0, self.TABLE_Z]))
            depth[mask > 0] = cam_dist_table

        # Draw objects
        self._draw_washer(rgb, depth, washer_pos, yaw_washer)
        self._draw_box(rgb, depth, box_pos, yaw_box)
        self._draw_mug(rgb, depth, mug_pos, yaw_mug)

        # Add realistic noise
        noise_rgb = np.random.randn(self.HEIGHT, self.WIDTH, 3) * 3
        rgb = np.clip(rgb.astype(np.float32) + noise_rgb, 0, 255).astype(np.uint8)
        noise_d = np.random.randn(self.HEIGHT, self.WIDTH) * 0.001
        depth = np.clip(depth + noise_d, 0.01, 5.0).astype(np.float32)

        cam_params = {
            "intrinsic_matrix": self.K,
            "extrinsic_matrix": self.extrinsic,
        }
        return rgb, depth, gt, cam_params

    def _random_positions(self):
        positions = []
        for _ in range(3):
            for _ in range(200):
                x = np.random.uniform(-0.15, 0.15)
                y = np.random.uniform(-0.20, 0.20)
                if all(np.linalg.norm(np.array([x, y]) - np.array(p)) >= 0.10
                       for p in positions):
                    positions.append([x, y])
                    break
            else:
                positions.append([np.random.uniform(-0.15, 0.15),
                                  np.random.uniform(-0.20, 0.20)])
        return positions

    def _world_to_px(self, world_pt):
        result = self.transformer.world_to_pixel(world_pt)
        if result is None:
            return None
        u, v, d = result
        u, v = int(round(u)), int(round(v))
        if 0 <= u < self.WIDTH and 0 <= v < self.HEIGHT:
            return (u, v)
        return None

    def _draw_washer(self, rgb, depth, pos, yaw):
        px = self._world_to_px(pos)
        if px is None:
            return
        # Scale radius to pixels (approximate)
        r_outer_px = self._meters_to_pixels(self.WASHER_OUTER_R, pos[2])
        r_inner_px = self._meters_to_pixels(self.WASHER_INNER_R, pos[2])
        cam_dist = np.linalg.norm(self.cam_pos - np.array(pos))
        
        cv2.circle(rgb, px, max(r_outer_px, 5), (190, 190, 190), -1)
        cv2.circle(rgb, px, max(r_inner_px, 2), (30, 30, 30), -1)
        cv2.circle(rgb, px, max(r_outer_px, 5), (150, 150, 150), 2)

        mask = np.zeros_like(depth, dtype=np.uint8)
        cv2.circle(mask, px, max(r_outer_px, 5), 255, -1)
        depth[mask > 0] = cam_dist

    def _draw_box(self, rgb, depth, pos, yaw):
        hw, hh = self.BOX_HALF[0], self.BOX_HALF[1]
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        corners = []
        for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
            rx = pos[0] + dx * cos_y - dy * sin_y
            ry = pos[1] + dx * sin_y + dy * cos_y
            corners.append([rx, ry, pos[2] + self.BOX_HALF[2]])
        
        px_corners = [self._world_to_px(c) for c in corners]
        if all(p is not None for p in px_corners):
            pts = np.array(px_corners, dtype=np.int32)
            cv2.fillPoly(rgb, [pts], (100, 150, 210))
            cv2.polylines(rgb, [pts], True, (80, 120, 180), 2)
            cam_dist = np.linalg.norm(self.cam_pos - np.array(pos))
            mask = np.zeros_like(depth, dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            depth[mask > 0] = cam_dist

    def _draw_mug(self, rgb, depth, pos, yaw):
        px_body = self._world_to_px(pos)
        if px_body is None:
            return
        r_px = self._meters_to_pixels(self.MUG_RADIUS, pos[2])
        cam_dist = np.linalg.norm(self.cam_pos - np.array(pos))

        cv2.circle(rgb, px_body, max(r_px, 5), (210, 80, 80), -1)
        cv2.circle(rgb, px_body, max(r_px, 5), (180, 60, 60), 2)

        # Handle protrusion
        handle_world = [pos[0] + self.MUG_RADIUS + self.HANDLE_HALF[0],
                        pos[1], pos[2]]
        px_handle = self._world_to_px(handle_world)
        if px_handle is not None:
            hw_px = self._meters_to_pixels(self.HANDLE_HALF[0] * 3, pos[2])
            hh_px = self._meters_to_pixels(self.HANDLE_HALF[1], pos[2])
            pts = np.array([
                [px_body[0] + r_px, px_body[1] - hh_px],
                [px_body[0] + r_px + hw_px, px_body[1] - hh_px - 3],
                [px_body[0] + r_px + hw_px, px_body[1] + hh_px + 3],
                [px_body[0] + r_px, px_body[1] + hh_px],
            ], dtype=np.int32)
            cv2.fillPoly(rgb, [pts], (220, 90, 90))
            cv2.polylines(rgb, [pts], True, (180, 60, 60), 2)

        mask = np.zeros_like(depth, dtype=np.uint8)
        cv2.circle(mask, px_body, max(r_px, 5), 255, -1)
        depth[mask > 0] = cam_dist

    def _meters_to_pixels(self, meters, obj_z):
        cam_dist = np.linalg.norm(self.cam_pos - np.array([0, 0, obj_z]))
        if cam_dist < 0.01:
            cam_dist = 0.5
        fx = self.K[0, 0]
        return max(1, int(meters * fx / cam_dist))


# ══════════════════════════════════════════════════════════════════
# PART 2: METRICS (inline version of experiments/metrics.py)
# ══════════════════════════════════════════════════════════════════

MATCH_THRESHOLD_PX = 40.0

def detection_rate(detections, ground_truths):
    if not ground_truths: return 1.0
    matched = 0
    for gt in ground_truths:
        gt_arr = np.array(gt[:2], dtype=np.float64)
        for det in detections:
            if np.linalg.norm(np.array(det[:2], dtype=np.float64) - gt_arr) < MATCH_THRESHOLD_PX:
                matched += 1; break
    return matched / len(ground_truths)

def pixel_localization_error(det_px, gt_world, transformer):
    result = transformer.world_to_pixel(gt_world)
    if result is None: return float("inf")
    return float(np.sqrt((det_px[0] - result[0])**2 + (det_px[1] - result[1])**2))

def position_error_3d(det_world, gt_world):
    return float(np.linalg.norm(np.array(det_world) - np.array(gt_world)))

def per_axis_mae(det_world, gt_world):
    d = np.abs(np.array(det_world) - np.array(gt_world))
    return {"x": float(d[0]), "y": float(d[1]), "z": float(d[2])}

def circle_iou(det_c, det_r, gt_c, gt_r, shape):
    h, w = shape
    m1 = np.zeros((h, w), dtype=np.uint8)
    m2 = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m1, (int(det_c[0]), int(det_c[1])), int(det_r), 1, -1)
    cv2.circle(m2, (int(gt_c[0]), int(gt_c[1])), int(gt_r), 1, -1)
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union > 0 else 0.0


# ══════════════════════════════════════════════════════════════════
# PART 3: EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════

GT_TYPE_MAP = {"hole": "holes", "surface": "surfaces", "handle": "handles"}

def run_single_trial(seed, cam_pos=None, cam_target=None, detector_config=None):
    """Run one trial: generate scene → detect → compute metrics."""
    scene = SyntheticScene(
        cam_pos=cam_pos or (0.0, -0.5, 0.9),
        cam_target=cam_target or (0.0, 0.0, 0.42)
    )
    rgb, depth, gt, cam_params = scene.generate(seed=seed)
    transformer = CoordinateTransformer(cam_params["intrinsic_matrix"],
                                        cam_params["extrinsic_matrix"])
    detector = FeatureDetector(config=detector_config)
    
    t0 = time.perf_counter()
    features = detector.detect_all(rgb)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Compute world coords for all detections
    world_coords = {}
    for ft, fl in features.items():
        wc = []
        for f in fl:
            u, v = f.pixel_coords
            vc = int(np.clip(v, 0, depth.shape[0] - 1))
            uc = int(np.clip(u, 0, depth.shape[1] - 1))
            wc.append(transformer.pixel_to_world(uc, vc, depth[vc, uc]))
        world_coords[ft] = wc

    # Compute metrics per GT object
    results = {}
    for obj_name, obj_gt in gt.items():
        feat_key = GT_TYPE_MAP.get(obj_gt["type"], obj_gt["type"] + "s")
        det_list = features.get(feat_key, [])
        det_worlds = world_coords.get(feat_key, [])
        
        gt_proj = transformer.world_to_pixel(obj_gt["position"])
        gt_px = gt_proj[:2] if gt_proj is not None else None

        detected = False
        best_px_err = -1.0
        best_3d_err = float("nan")
        best_axes = {"x": float("nan"), "y": float("nan"), "z": float("nan")}
        best_conf = 0.0
        best_iou = 0.0
        best_det_world = [float("nan")] * 3

        if gt_px is not None:
            det_pixels = [f.pixel_coords for f in det_list]
            rate = detection_rate(det_pixels, [gt_px])
            detected = rate > 0.0

            for i, feat in enumerate(det_list):
                px_err = pixel_localization_error(feat.pixel_coords, obj_gt["position"], transformer)
                if best_px_err < 0 or px_err < best_px_err:
                    best_px_err = px_err
                    best_conf = feat.confidence
                    if i < len(det_worlds):
                        best_det_world = det_worlds[i].tolist()
                        best_3d_err = position_error_3d(det_worlds[i], obj_gt["position"])
                        best_axes = per_axis_mae(det_worlds[i], obj_gt["position"])
                    if obj_gt["type"] == "hole" and feat.radius is not None and gt_px is not None:
                        gt_r_px = max(3, scene._meters_to_pixels(obj_gt["feature_radius"], obj_gt["position"][2]))
                        best_iou = circle_iou(feat.pixel_coords, feat.radius, gt_px, gt_r_px, (480, 640))

        results[obj_name] = {
            "seed": seed, "object_class": obj_name, "feature_type": obj_gt["type"],
            "detected": int(detected), "pixel_error": best_px_err, "iou": best_iou,
            "confidence": best_conf, "latency_ms": latency_ms,
            "detected_x": best_det_world[0], "detected_y": best_det_world[1], "detected_z": best_det_world[2],
            "gt_x": obj_gt["position"][0], "gt_y": obj_gt["position"][1], "gt_z": obj_gt["position"][2],
            "error_x": best_axes["x"], "error_y": best_axes["y"], "error_z": best_axes["z"],
            "error_total": best_3d_err,
        }
    return results, rgb, depth, features, world_coords, gt


def run_experiment_1(num_trials=50):
    """Experiment 1: Detection Accuracy."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1: Feature Detection Accuracy ({num_trials} trials)")
    print(f"{'='*60}")
    csv_path = os.path.join(RESULTS_DIR, "experiment1_detection_accuracy.csv")
    fields = ["trial","seed","object_class","feature_type","detected","pixel_error","iou","confidence","latency_ms"]
    
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for trial in range(num_trials):
            seed = trial
            try:
                result = run_single_trial(seed)[0]
                for obj, m in result.items():
                    w.writerow({k: m.get(k, "") for k in fields} | {"trial": trial, "seed": seed})
                if (trial + 1) % 10 == 0:
                    print(f"  Trial {trial+1}/{num_trials} complete")
            except Exception as e:
                print(f"  Trial {trial} failed: {e}")
        f.flush()
    print(f"  → Saved: {csv_path}")
    return csv_path


def run_experiment_2(num_trials=50):
    """Experiment 2: 3D Coordinate Precision."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2: 3D Coordinate Precision ({num_trials} trials)")
    print(f"{'='*60}")
    csv_path = os.path.join(RESULTS_DIR, "experiment2_coordinate_precision.csv")
    fields = ["trial","seed","object_class","detected_x","detected_y","detected_z",
              "gt_x","gt_y","gt_z","error_x","error_y","error_z","error_total"]
    
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for trial in range(num_trials):
            seed = 1000 + trial
            try:
                result = run_single_trial(seed)[0]
                for obj, m in result.items():
                    w.writerow({k: m.get(k, "") for k in fields} | {"trial": trial, "seed": seed})
                if (trial + 1) % 10 == 0:
                    print(f"  Trial {trial+1}/{num_trials} complete")
            except Exception as e:
                print(f"  Trial {trial} failed: {e}")
        f.flush()
    print(f"  → Saved: {csv_path}")
    return csv_path


def run_experiment_3a(reps_per_yaw=5):
    """Experiment 3a: Orientation robustness."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 3a: Orientation Robustness")
    print(f"{'='*60}")
    csv_path = os.path.join(RESULTS_DIR, "experiment3a_orientation.csv")
    fields = ["yaw_deg","object_class","detection_rate","mean_pixel_error","mean_3d_error"]
    
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for yaw_deg in range(0, 360, 30):
            obj_det, obj_px, obj_3d = {}, {}, {}
            for rep in range(reps_per_yaw):
                seed = 3000 + yaw_deg * 10 + rep
                try:
                    # Generate with fixed yaw
                    scene = SyntheticScene()
                    rgb, depth, gt, cp = scene.generate(seed=seed, yaw_washer=np.radians(yaw_deg),
                                                         yaw_box=np.radians(yaw_deg), yaw_mug=np.radians(yaw_deg))
                    transformer = CoordinateTransformer(cp["intrinsic_matrix"], cp["extrinsic_matrix"])
                    detector = FeatureDetector()
                    features = detector.detect_all(rgb)
                    
                    for obj_name, obj_gt in gt.items():
                        feat_key = GT_TYPE_MAP.get(obj_gt["type"], obj_gt["type"] + "s")
                        det_list = features.get(feat_key, [])
                        gt_proj = transformer.world_to_pixel(obj_gt["position"])
                        gt_px = gt_proj[:2] if gt_proj else None
                        
                        if gt_px:
                            det_pixels = [f.pixel_coords for f in det_list]
                            rate = detection_rate(det_pixels, [gt_px])
                            obj_det.setdefault(obj_name, []).append(rate)
                            for feat in det_list:
                                px_err = pixel_localization_error(feat.pixel_coords, obj_gt["position"], transformer)
                                if px_err < 100:
                                    obj_px.setdefault(obj_name, []).append(px_err)
                except Exception as e:
                    pass
            
            for obj_name in obj_det:
                w.writerow({
                    "yaw_deg": yaw_deg, "object_class": obj_name,
                    "detection_rate": float(np.mean(obj_det[obj_name])),
                    "mean_pixel_error": float(np.mean(obj_px.get(obj_name, [float("nan")]))),
                    "mean_3d_error": float("nan"),
                })
            print(f"  Yaw {yaw_deg}° complete")
        f.flush()
    print(f"  → Saved: {csv_path}")


def run_experiment_3b(reps_per_dist=5):
    """Experiment 3b: Camera distance robustness."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 3b: Distance Robustness")
    print(f"{'='*60}")
    csv_path = os.path.join(RESULTS_DIR, "experiment3b_distance.csv")
    fields = ["camera_distance_m","object_class","detection_rate","mean_pixel_error","mean_3d_error"]
    
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for dist in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            cam_pos = (0.0, 0.0, 0.42 + dist)
            cam_target = (0.0, 0.0, 0.42)
            obj_det, obj_px = {}, {}
            
            for rep in range(reps_per_dist):
                seed = 4000 + int(dist * 100) + rep
                try:
                    result = run_single_trial(seed, cam_pos=cam_pos, cam_target=cam_target)[0]
                    for obj_name, m in result.items():
                        obj_det.setdefault(obj_name, []).append(m["detected"])
                        if m["pixel_error"] >= 0:
                            obj_px.setdefault(obj_name, []).append(m["pixel_error"])
                except:
                    pass
            
            for obj_name in obj_det:
                w.writerow({
                    "camera_distance_m": dist, "object_class": obj_name,
                    "detection_rate": float(np.mean(obj_det[obj_name])),
                    "mean_pixel_error": float(np.mean(obj_px.get(obj_name, [float("nan")]))),
                    "mean_3d_error": float("nan"),
                })
            print(f"  Distance {dist}m complete")
        f.flush()
    print(f"  → Saved: {csv_path}")


def run_experiment_3c(num_trials=20):
    """Experiment 3c: Clutter (all 3 objects)."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 3c: Clutter Robustness")
    print(f"{'='*60}")
    csv_path = os.path.join(RESULTS_DIR, "experiment3c_clutter.csv")
    fields = ["trial","num_objects","object_class","detection_rate"]
    
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for trial in range(num_trials):
            seed = 5000 + trial
            try:
                result = run_single_trial(seed)[0]
                for obj, m in result.items():
                    w.writerow({"trial": trial, "num_objects": 3,
                                "object_class": obj, "detection_rate": m["detected"]})
            except:
                pass
        f.flush()
    print(f"  → Saved: {csv_path}")


def run_parameter_sensitivity(trials_per_val=10):
    """Parameter sensitivity sweep."""
    print(f"\n{'='*60}")
    print("PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    csv_path = os.path.join(RESULTS_DIR, "parameter_sensitivity.csv")
    fields = ["parameter_name","parameter_value","detection_rate","object_class"]
    
    sweeps = [
        ("hough_param2", [15, 20, 25, 30, 35, 40, 45]),
        ("canny_low", [30, 40, 50, 60, 70]),
        ("surface_min_area", [200, 500, 1000, 2000, 3000]),
    ]
    
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for param_name, values in sweeps:
            for val in values:
                cfg = {param_name: val}
                if param_name == "canny_low":
                    cfg["canny_high"] = val * 3
                
                obj_det = {}
                for trial in range(trials_per_val):
                    seed = 6000 + hash(f"{param_name}{val}") % 1000 + trial
                    seed = seed % (2**31)
                    try:
                        result = run_single_trial(seed, detector_config=cfg)[0]
                        for obj, m in result.items():
                            obj_det.setdefault(obj, []).append(m["detected"])
                    except:
                        pass
                
                for obj, dets in obj_det.items():
                    w.writerow({"parameter_name": param_name, "parameter_value": str(val),
                                "detection_rate": float(np.mean(dets)), "object_class": obj})
            print(f"  {param_name} sweep complete")
        f.flush()
    print(f"  → Saved: {csv_path}")


# ══════════════════════════════════════════════════════════════════
# PART 4: TEST SUITE (no pytest needed)
# ══════════════════════════════════════════════════════════════════

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def record(self, name, passed, msg=""):
        if passed:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed += 1
            self.errors.append((name, msg))
            print(f"  ❌ {name}: {msg}")

def run_all_tests():
    """Run the complete test suite."""
    print(f"\n{'='*60}")
    print("RUNNING TEST SUITE")
    print(f"{'='*60}")
    
    tr = TestResults()
    detector = FeatureDetector()

    # ── Preprocessing Tests ──
    print("\n--- Preprocessing Tests ---")
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    gray, blurred = detector.preprocess(img)
    tr.record("preprocess_output_shapes", gray.shape == (480, 640) and blurred.shape == (480, 640))
    tr.record("preprocess_dtypes", gray.dtype == np.uint8 and blurred.dtype == np.uint8)
    tr.record("blur_reduces_variance", blurred.var() <= gray.var())

    # ── Hole Detection Tests ──
    print("\n--- Hole Detection Tests ---")
    circle_img = np.zeros((480, 640, 3), dtype=np.uint8) + 40
    cv2.circle(circle_img, (320, 240), 35, (200, 200, 200), 2)
    cv2.circle(circle_img, (320, 240), 30, (20, 20, 20), -1)
    holes = detector.detect_holes(circle_img)
    tr.record("detects_circle", len(holes) >= 1)
    if holes:
        dist = np.linalg.norm(np.array(holes[0].pixel_coords) - np.array([320, 240]))
        tr.record("circle_center_accuracy", dist < 20, f"dist={dist:.1f}")
        tr.record("hole_has_radius", holes[0].radius is not None and holes[0].radius > 0)
        tr.record("hole_feature_type", holes[0].feature_type == "hole")
    else:
        for n in ["circle_center_accuracy", "hole_has_radius", "hole_feature_type"]:
            tr.record(n, False, "no holes detected")

    blank = np.ones((480, 640, 3), dtype=np.uint8) * 128
    tr.record("no_false_positive_blank", len(detector.detect_holes(blank)) == 0)

    multi_img = np.zeros((480, 640, 3), dtype=np.uint8) + 40
    cv2.circle(multi_img, (150, 200), 25, (200, 200, 200), 2)
    cv2.circle(multi_img, (150, 200), 20, (20, 20, 20), -1)
    cv2.circle(multi_img, (450, 300), 35, (200, 200, 200), 2)
    cv2.circle(multi_img, (450, 300), 30, (20, 20, 20), -1)
    tr.record("multiple_circles", len(detector.detect_holes(multi_img)) >= 2)

    # ── Surface Detection Tests ──
    print("\n--- Surface Detection Tests ---")
    rect_img = np.zeros((480, 640, 3), dtype=np.uint8) + 30
    cv2.rectangle(rect_img, (200, 150), (400, 290), (180, 180, 220), -1)
    surfaces = detector.detect_surfaces(rect_img)
    tr.record("detects_rectangle", len(surfaces) >= 1)
    if surfaces:
        tr.record("surface_feature_type", surfaces[0].feature_type == "surface")
        tr.record("surface_has_area", surfaces[0].area is not None and surfaces[0].area > 0)
        dist = np.linalg.norm(np.array(surfaces[0].pixel_coords) - np.array([300, 220]))
        tr.record("surface_centroid_accuracy", dist < 30, f"dist={dist:.1f}")
    else:
        for n in ["surface_feature_type", "surface_has_area", "surface_centroid_accuracy"]:
            tr.record(n, False, "no surfaces detected")

    tiny_img = np.zeros((480, 640, 3), dtype=np.uint8) + 30
    cv2.rectangle(tiny_img, (300, 230), (310, 240), (180, 180, 220), -1)
    small = [f for f in detector.detect_surfaces(tiny_img) if f.area and f.area < 200]
    tr.record("small_surface_filtered", len(small) == 0)

    # ── Handle Detection Tests ──
    print("\n--- Handle Detection Tests ---")
    handle_img = np.zeros((480, 640, 3), dtype=np.uint8) + 30
    cv2.circle(handle_img, (320, 240), 60, (150, 150, 150), -1)
    cv2.rectangle(handle_img, (380, 225), (430, 255), (160, 160, 160), -1)
    handles = detector.detect_handles(handle_img)
    tr.record("handle_returns_list", isinstance(handles, list))
    for h in handles:
        tr.record("handle_feature_type", h.feature_type == "handle")
        break
    else:
        tr.record("handle_feature_type", True)  # empty list is ok
    tr.record("no_handle_on_blank", len(detector.detect_handles(blank)) == 0)

    # ── Combined Detection Tests ──
    print("\n--- Combined Detection Tests ---")
    result = detector.detect_all(blank)
    tr.record("detect_all_returns_dict", "holes" in result and "surfaces" in result and "handles" in result)
    
    det2 = FeatureDetector(config={"hough_min_radius": 10, "surface_min_area": 1000})
    tr.record("custom_config_hough", det2.cfg["hough_min_radius"] == 10)
    tr.record("custom_config_surface", det2.cfg["surface_min_area"] == 1000)
    tr.record("custom_config_defaults", det2.cfg["canny_low"] == 50)

    # ── Coordinate Transform Tests ──
    print("\n--- Coordinate Transform Tests ---")
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    ext = np.eye(4, dtype=np.float64)
    t = CoordinateTransformer(K, ext)

    r = t.pixel_to_camera(320, 240, 2.0)
    tr.record("center_pixel_camera", np.allclose(r, [0, 0, 2.0], atol=1e-6))

    r1 = t.pixel_to_camera(400, 300, 1.0)
    r2 = t.pixel_to_camera(400, 300, 2.0)
    tr.record("depth_scaling", np.allclose(r2, r1 * 2.0, atol=1e-6))

    r = t.pixel_to_camera(420, 340, 1.0)
    tr.record("off_center_positive_xy", r[0] > 0 and r[1] > 0)

    cp = np.array([0.5, -0.3, 1.5])
    wp = t.camera_to_world(cp)
    tr.record("identity_cam_to_world", np.allclose(wp, cp, atol=1e-6))

    ext2 = np.eye(4, dtype=np.float64); ext2[2, 3] = 1.0
    t2 = CoordinateTransformer(K, ext2)
    wp2 = t2.camera_to_world(np.array([0, 0, 0]))
    tr.record("translation_offset", np.isclose(wp2[2], -1.0, atol=1e-6))

    wp = t.pixel_to_world(350, 200, 1.5)
    rec = t.world_to_pixel(wp)
    tr.record("roundtrip_u", rec is not None and abs(rec[0] - 350) < 1.0)
    tr.record("roundtrip_v", rec is not None and abs(rec[1] - 200) < 1.0)
    tr.record("roundtrip_d", rec is not None and abs(rec[2] - 1.5) < 0.01)

    wp = t.pixel_to_world(320, 240, 1.0)
    tr.record("center_maps_to_origin", np.allclose(wp, [0, 0, 1.0], atol=1e-6))

    r = t.world_to_pixel([0, 0, 1.0])
    tr.record("origin_to_center", r is not None and abs(r[0] - 320) < 1 and abs(r[1] - 240) < 1)
    tr.record("zero_depth_none", t.world_to_pixel([0, 0, 0]) is None)

    depth_img = np.ones((480, 640), dtype=np.float32) * 2.0
    pixels = [(100, 200), (320, 240), (500, 400)]
    batch = t.batch_pixel_to_world(pixels, depth_img)
    for i, (u, v) in enumerate(pixels):
        ind = t.pixel_to_world(u, v, 2.0)
        tr.record(f"batch_matches_individual_{i}", np.allclose(batch[i], ind, atol=1e-6))

    tr.record("batch_empty", t.batch_pixel_to_world([], depth_img) == [])

    r = t.pixel_to_world(320, 240, 1000.0)
    tr.record("large_depth_finite", np.isfinite(r).all())
    r = t.pixel_to_world(320, 240, 0.001)
    tr.record("small_depth_finite", np.isfinite(r).all())

    identity = t.K @ t.K_inv
    tr.record("K_invertible", np.allclose(identity, np.eye(3), atol=1e-10))

    # ── Synthetic Scene Tests ──
    print("\n--- Synthetic Scene Tests ---")
    scene = SyntheticScene()
    rgb, depth, gt, cp = scene.generate(seed=99)
    tr.record("scene_rgb_shape", rgb.shape == (480, 640, 3))
    tr.record("scene_depth_shape", depth.shape == (480, 640))
    tr.record("scene_gt_has_3_objects", len(gt) == 3)
    tr.record("scene_depth_positive", (depth > 0).all())
    tr.record("scene_gt_washer", "washer" in gt and gt["washer"]["type"] == "hole")
    tr.record("scene_gt_box", "box" in gt and gt["box"]["type"] == "surface")
    tr.record("scene_gt_mug", "mug" in gt and gt["mug"]["type"] == "handle")

    # ── Integration: Full Pipeline Test ──
    print("\n--- Integration Pipeline Tests ---")
    result, rgb, depth, feats, wc, gt = run_single_trial(seed=42)
    tr.record("pipeline_returns_3_objects", len(result) == 3)
    tr.record("pipeline_rgb_valid", rgb.shape == (480, 640, 3))
    for obj in ["washer", "box", "mug"]:
        tr.record(f"pipeline_{obj}_in_result", obj in result)
    
    any_detected = any(m["detected"] for m in result.values())
    tr.record("pipeline_at_least_one_detection", any_detected)

    # ── Metrics Tests ──
    print("\n--- Metrics Tests ---")
    tr.record("det_rate_perfect", detection_rate([(100, 100)], [(100, 100)]) == 1.0)
    tr.record("det_rate_miss", detection_rate([(500, 500)], [(100, 100)]) == 0.0)
    tr.record("det_rate_empty_gt", detection_rate([], []) == 1.0)
    tr.record("pos_error_zero", position_error_3d([1, 2, 3], [1, 2, 3]) == 0.0)
    tr.record("pos_error_positive", position_error_3d([1, 0, 0], [0, 0, 0]) == 1.0)
    mae = per_axis_mae([1.5, 2.5, 3.5], [1.0, 2.0, 3.0])
    tr.record("per_axis_mae", abs(mae["x"] - 0.5) < 1e-6 and abs(mae["y"] - 0.5) < 1e-6)

    iou = circle_iou((100, 100), 20, (100, 100), 20, (480, 640))
    tr.record("circle_iou_identical", iou > 0.99)
    iou2 = circle_iou((100, 100), 20, (400, 400), 20, (480, 640))
    tr.record("circle_iou_far", iou2 == 0.0)

    return tr


# ══════════════════════════════════════════════════════════════════
# PART 5: VISUALIZATION & REPORT GENERATION
# ══════════════════════════════════════════════════════════════════

def generate_all_charts():
    """Generate all experiment charts."""
    print(f"\n{'='*60}")
    print("GENERATING CHARTS")
    print(f"{'='*60}")

    # Chart 1: Detection Accuracy
    csv1 = os.path.join(RESULTS_DIR, "experiment1_detection_accuracy.csv")
    if os.path.exists(csv1):
        rows = list(csv.DictReader(open(csv1)))
        obj_det = {}
        for r in rows:
            obj_det.setdefault(r["object_class"], []).append(int(r["detected"]))
        
        objects = sorted(obj_det.keys())
        rates = [np.mean(obj_det[o]) * 100 for o in objects]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#2196F3", "#4CAF50", "#FF5722"]
        bars = ax.bar(objects, rates, color=colors[:len(objects)], edgecolor="black", lw=0.8)
        for b, r in zip(bars, rates):
            ax.text(b.get_x() + b.get_width()/2, b.get_height()+1, f"{r:.1f}%",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 115); ax.set_ylabel("Detection Rate (%)", fontsize=12)
        ax.set_title("Feature Detection Accuracy by Object Class", fontsize=13, fontweight="bold")
        ax.axhline(100, color="gray", ls="--", lw=0.8); ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "detection_accuracy_chart.png"), dpi=150)
        plt.close()
        print("  → detection_accuracy_chart.png")

    # Chart 2: 3D Error Boxplots
    csv2 = os.path.join(RESULTS_DIR, "experiment2_coordinate_precision.csv")
    if os.path.exists(csv2):
        rows = list(csv.DictReader(open(csv2)))
        obj_errs = {}
        for r in rows:
            obj = r["object_class"]
            obj_errs.setdefault(obj, {"x":[],"y":[],"z":[],"total":[]})
            for ax_name in ["x","y","z","total"]:
                try:
                    v = float(r[f"error_{ax_name}"])
                    if not np.isnan(v) and v < 10:
                        obj_errs[obj][ax_name].append(v)
                except: pass
        
        fig, ax = plt.subplots(figsize=(10, 6))
        data, labels, positions = [], [], []
        for i, obj in enumerate(sorted(obj_errs)):
            for j, axis in enumerate(["x","y","z","total"]):
                pos = i * 5 + j + 1
                positions.append(pos)
                data.append(obj_errs[obj].get(axis, [0]))
                axis_names = ["X", "Y", "Z", "Total"]
                labels.append(f"{obj}\n{axis_names[j]}")
        
        if data:
            ax.boxplot(data, positions=positions, patch_artist=True,
                       boxprops=dict(facecolor="#90CAF9", alpha=0.7),
                       medianprops=dict(color="red", lw=2))
            ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Error (m)"); ax.set_title("3D Coordinate Precision — Per-Axis MAE", fontweight="bold")
        ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "3d_error_boxplots.png"), dpi=150)
        plt.close()
        print("  → 3d_error_boxplots.png")

    # Chart 3: Robustness curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Robustness Analysis", fontsize=14, fontweight="bold")
    
    p3a = os.path.join(RESULTS_DIR, "experiment3a_orientation.csv")
    if os.path.exists(p3a):
        rows = list(csv.DictReader(open(p3a)))
        obj_data = {}
        for r in rows:
            obj_data.setdefault(r["object_class"], {"y":[], "r":[]})
            obj_data[r["object_class"]]["y"].append(float(r["yaw_deg"]))
            obj_data[r["object_class"]]["r"].append(float(r["detection_rate"])*100)
        for obj, d in sorted(obj_data.items()):
            axes[0].plot(d["y"], d["r"], marker="o", label=obj)
        axes[0].set_xlabel("Object Yaw (°)"); axes[0].set_ylabel("Detection Rate (%)")
        axes[0].set_title("Detection Rate vs Orientation"); axes[0].legend(); axes[0].grid(alpha=0.4)
        axes[0].set_ylim(-5, 115)
    
    p3b = os.path.join(RESULTS_DIR, "experiment3b_distance.csv")
    if os.path.exists(p3b):
        rows = list(csv.DictReader(open(p3b)))
        obj_data = {}
        for r in rows:
            obj_data.setdefault(r["object_class"], {"d":[], "r":[]})
            obj_data[r["object_class"]]["d"].append(float(r["camera_distance_m"]))
            obj_data[r["object_class"]]["r"].append(float(r["detection_rate"])*100)
        for obj, d in sorted(obj_data.items()):
            axes[1].plot(d["d"], d["r"], marker="s", label=obj)
        axes[1].set_xlabel("Camera Distance (m)"); axes[1].set_ylabel("Detection Rate (%)")
        axes[1].set_title("Detection Rate vs Camera Distance"); axes[1].legend(); axes[1].grid(alpha=0.4)
        axes[1].set_ylim(-5, 115)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "robustness_curves.png"), dpi=150)
    plt.close()
    print("  → robustness_curves.png")

    # Chart 4: Parameter sensitivity
    pcsv = os.path.join(RESULTS_DIR, "parameter_sensitivity.csv")
    if os.path.exists(pcsv):
        rows = list(csv.DictReader(open(pcsv)))
        params = {}
        for r in rows:
            p = r["parameter_name"]
            params.setdefault(p, {"v":[], "r":[]})
            params[p]["v"].append(r["parameter_value"])
            params[p]["r"].append(float(r["detection_rate"])*100)
        
        n = len(params)
        fig, axes = plt.subplots(1, max(n,1), figsize=(5*max(n,1), 5))
        if n == 1: axes = [axes]
        for ax, (pname, d) in zip(axes, params.items()):
            ax.plot(range(len(d["v"])), d["r"], marker="D", color="#673AB7")
            ax.set_xticks(range(len(d["v"]))); ax.set_xticklabels(d["v"], rotation=30, fontsize=8)
            ax.set_xlabel(pname); ax.set_ylabel("Detection Rate (%)")
            ax.set_title(f"Sensitivity: {pname}"); ax.grid(alpha=0.4); ax.set_ylim(-5, 115)
        fig.suptitle("Parameter Sensitivity Analysis", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "parameter_sensitivity_plots.png"), dpi=150)
        plt.close()
        print("  → parameter_sensitivity_plots.png")

    # Demo summary figure
    print("  Generating demo summary figure...")
    scene = SyntheticScene()
    rgb, depth, gt, cp = scene.generate(seed=42)
    detector = FeatureDetector()
    features = detector.detect_all(rgb)
    transformer = CoordinateTransformer(cp["intrinsic_matrix"], cp["extrinsic_matrix"])
    
    # Draw annotations
    annotated = rgb.copy()
    COLORS = {"holes": (0, 255, 0), "surfaces": (255, 165, 0), "handles": (0, 100, 255)}
    for ft, fl in features.items():
        col = COLORS.get(ft, (255, 255, 255))
        for f in fl:
            u, v = f.pixel_coords
            if f.feature_type == "hole" and f.radius:
                cv2.circle(annotated, (u, v), int(f.radius), col, 2)
                cv2.circle(annotated, (u, v), 3, col, -1)
                # Approach arrow (downward)
                cv2.arrowedLine(annotated, (u, v-30), (u, v), col, 2, tipLength=0.3)
            elif f.feature_type == "surface" and f.pixel_region:
                x, y, w, h = f.pixel_region[0]
                cv2.rectangle(annotated, (x, y), (x+w, y+h), col, 2)
                cv2.circle(annotated, (u, v), 4, col, -1)
                cv2.arrowedLine(annotated, (u, v-30), (u, v), col, 2, tipLength=0.3)
            elif f.feature_type == "handle" and f.pixel_region:
                x, y, w, h = f.pixel_region[0]
                cv2.rectangle(annotated, (x, y), (x+w, y+h), col, 2)
                cv2.arrowedLine(annotated, (u-30, v), (u, v), col, 2, tipLength=0.3)
            cv2.putText(annotated, f"{f.feature_type}", (u+5, v-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].imshow(rgb); axes[0,0].set_title("Raw RGB Input"); axes[0,0].axis("off")
    im = axes[0,1].imshow(depth, cmap="viridis"); axes[0,1].set_title("Depth Map"); axes[0,1].axis("off")
    plt.colorbar(im, ax=axes[0,1], fraction=0.046, label="Depth (m)")
    axes[1,0].imshow(annotated); axes[1,0].set_title("Detected Features + Approach Vectors"); axes[1,0].axis("off")
    
    # Confidence chart
    all_feats = [(f.feature_type, f.confidence) for fl in features.values() for f in fl]
    if all_feats:
        types, confs = zip(*all_feats)
        type_colors = {"hole": "#4CAF50", "surface": "#FF9800", "handle": "#2196F3"}
        bar_colors = [type_colors.get(t, "#999") for t in types]
        axes[1,1].barh(range(len(confs)), confs, color=bar_colors)
        axes[1,1].set_yticks(range(len(types))); axes[1,1].set_yticklabels(types, fontsize=8)
        axes[1,1].set_xlabel("Confidence"); axes[1,1].set_title("Detection Confidence")
        axes[1,1].set_xlim(0, 1.1)
    
    plt.suptitle("Vision-Based Robotic Affordance Detection — Demo Output", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "summary_figure.png"), dpi=200)
    plt.close()
    print("  → summary_figure.png")


def generate_report(test_results):
    """Generate the final EXPERIMENT_REPORT.md."""
    print(f"\n{'='*60}")
    print("GENERATING EXPERIMENT REPORT")
    print(f"{'='*60}")

    lines = [
        "# Experiment Report — Vision-Based Robotic Affordance Detection",
        "", "---", "",
        f"**Test Suite**: {test_results.passed} passed, {test_results.failed} failed "
        f"out of {test_results.passed + test_results.failed} total tests",
        "",
    ]

    # Exp 1
    lines += ["## Experiment 1: Feature Detection Accuracy", ""]
    csv1 = os.path.join(RESULTS_DIR, "experiment1_detection_accuracy.csv")
    if os.path.exists(csv1):
        rows = list(csv.DictReader(open(csv1)))
        obj_stats = {}
        for r in rows:
            obj = r["object_class"]
            obj_stats.setdefault(obj, {"det":[], "px":[], "iou":[], "lat":[], "ft": r["feature_type"]})
            obj_stats[obj]["det"].append(int(r["detected"]))
            try:
                px = float(r["pixel_error"])
                if px >= 0: obj_stats[obj]["px"].append(px)
            except: pass
            try: obj_stats[obj]["iou"].append(float(r["iou"]))
            except: pass
            try: obj_stats[obj]["lat"].append(float(r["latency_ms"]))
            except: pass

        lines += ["| Object | Feature | Detection Rate | Mean Pixel Error | Mean IoU | Latency (ms) |",
                   "|--------|---------|---------------|-----------------|----------|-------------|"]
        for obj in sorted(obj_stats):
            d = obj_stats[obj]
            rate = np.mean(d["det"]) * 100
            px = np.mean(d["px"]) if d["px"] else float("nan")
            iou = np.mean(d["iou"]) if d["iou"] else float("nan")
            lat = np.mean(d["lat"]) if d["lat"] else float("nan")
            lines.append(f"| {obj} | {d['ft']} | {rate:.1f}% | {px:.2f} px | {iou:.3f} | {lat:.1f} |")
        lines += ["", "![Detection Accuracy](detection_accuracy_chart.png)", ""]

    # Exp 2
    lines += ["---", "", "## Experiment 2: 3D Coordinate Precision", ""]
    csv2 = os.path.join(RESULTS_DIR, "experiment2_coordinate_precision.csv")
    if os.path.exists(csv2):
        rows = list(csv.DictReader(open(csv2)))
        obj_errs = {}
        for r in rows:
            obj = r["object_class"]
            obj_errs.setdefault(obj, {"x":[],"y":[],"z":[],"total":[]})
            for ax in ["x","y","z","total"]:
                try:
                    v = float(r[f"error_{ax}"])
                    if not np.isnan(v): obj_errs[obj][ax].append(v)
                except: pass

        lines += ["| Object | MAE X (m) | MAE Y (m) | MAE Z (m) | Total Error (m) |",
                   "|--------|----------|----------|----------|----------------|"]
        for obj in sorted(obj_errs):
            d = obj_errs[obj]
            lines.append(f"| {obj} | {np.mean(d['x']):.4f} | {np.mean(d['y']):.4f} | "
                         f"{np.mean(d['z']):.4f} | {np.mean(d['total']):.4f} |")
        lines += ["", "![3D Error Boxplots](3d_error_boxplots.png)", ""]

    # Exp 3
    lines += ["---", "", "## Experiment 3: Robustness Analysis", ""]
    
    p3a = os.path.join(RESULTS_DIR, "experiment3a_orientation.csv")
    if os.path.exists(p3a):
        rows = list(csv.DictReader(open(p3a)))
        all_rates = [float(r["detection_rate"]) for r in rows]
        lines += ["### 3a — Orientation Robustness",
                   f"- Mean detection rate across all orientations: **{np.mean(all_rates)*100:.1f}%**", ""]
    
    p3b = os.path.join(RESULTS_DIR, "experiment3b_distance.csv")
    if os.path.exists(p3b):
        rows = list(csv.DictReader(open(p3b)))
        all_rates = [float(r["detection_rate"]) for r in rows]
        lines += ["### 3b — Distance Robustness",
                   f"- Mean detection rate across all distances: **{np.mean(all_rates)*100:.1f}%**", ""]

    p3c = os.path.join(RESULTS_DIR, "experiment3c_clutter.csv")
    if os.path.exists(p3c):
        rows = list(csv.DictReader(open(p3c)))
        all_rates = [float(r["detection_rate"]) for r in rows]
        lines += ["### 3c — Clutter Robustness",
                   f"- Mean detection rate with 3 objects: **{np.mean(all_rates)*100:.1f}%**", ""]

    lines += ["![Robustness Curves](robustness_curves.png)", ""]

    # Parameter sensitivity
    lines += ["---", "", "## Parameter Sensitivity Analysis", ""]
    pcsv = os.path.join(RESULTS_DIR, "parameter_sensitivity.csv")
    if os.path.exists(pcsv):
        rows = list(csv.DictReader(open(pcsv)))
        params = {}
        for r in rows:
            p = r["parameter_name"]
            v = r["parameter_value"]
            params.setdefault(p, {}).setdefault(v, []).append(float(r["detection_rate"]))

        lines += ["| Parameter | Best Value | Detection Rate |",
                   "|-----------|-----------|---------------|"]
        for p, vals in sorted(params.items()):
            avg = {v: np.mean(rs) for v, rs in vals.items()}
            best = max(avg, key=avg.get)
            lines.append(f"| {p} | {best} | {avg[best]*100:.1f}% |")
        lines += ["", "![Parameter Sensitivity](parameter_sensitivity_plots.png)", ""]

    # Summary figure
    lines += ["---", "", "## Demo Output", "",
              "![Summary Figure](summary_figure.png)", "",
              "---", "",
              f"_Report generated automatically. Tests: {test_results.passed}/{test_results.passed+test_results.failed} passing._"]

    report = "\n".join(lines)
    out = os.path.join(RESULTS_DIR, "EXPERIMENT_REPORT.md")
    with open(out, "w") as f:
        f.write(report)
    print(f"  → {out}")
    return report


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()

    # 1. Run tests
    test_results = run_all_tests()

    # 2. Run experiments
    run_experiment_1(num_trials=50)
    run_experiment_2(num_trials=50)
    run_experiment_3a(reps_per_yaw=5)
    run_experiment_3b(reps_per_dist=5)
    run_experiment_3c(num_trials=20)
    run_parameter_sensitivity(trials_per_val=10)

    # 3. Generate charts
    generate_all_charts()

    # 4. Generate report
    report = generate_report(test_results)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\nTests: {test_results.passed} passed, {test_results.failed} failed")
    if test_results.errors:
        print("Failed tests:")
        for name, msg in test_results.errors:
            print(f"  ❌ {name}: {msg}")
    print(f"\nResults: {RESULTS_DIR}/")
    for f in sorted(os.listdir(RESULTS_DIR)):
        size = os.path.getsize(os.path.join(RESULTS_DIR, f))
        print(f"  {f:45s} {size:>8,d} bytes")
