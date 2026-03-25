"""
Experiment Runner Module
=========================
Executes the full evaluation suite: detection accuracy, 3D coordinate precision,
robustness testing, and parameter sensitivity analysis.
"""

import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulation.environment import SimulationEnvironment
from src.camera import VirtualCamera
from src.coordinate_transform import CoordinateTransformer
from src.feature_detector import FeatureDetector
from src.logger import setup_logger
from experiments.metrics import (
    MATCH_THRESHOLD_PX,
    circle_iou,
    detection_rate,
    per_axis_mae,
    pixel_localization_error,
    position_error_3d,
    processing_latency,
)

logger = setup_logger(__name__)

# Default camera configuration
DEFAULT_CAM_POS = (0.0, -0.5, 0.9)
DEFAULT_CAM_TARGET = (0.0, 0.0, 0.42)
DEFAULT_CAM_UP = (0, 0, 1)


class ExperimentRunner:
    """Runs all evaluation experiments and writes results to CSV files.

    Attributes:
        num_trials: Number of randomized trials per experiment.
        seed_start: Starting seed; trial i uses seed seed_start + i.
        output_dir: Directory where result CSV files are written.
    """

    def __init__(
        self,
        num_trials: int = 50,
        seed_start: int = 0,
        output_dir: str = "results",
    ) -> None:
        """Initialize the experiment runner.

        Args:
            num_trials: Number of randomized trials per experiment.
            seed_start: Seed for the first trial; subsequent trials increment by 1.
            output_dir: Directory for CSV and PNG outputs.
        """
        self.num_trials = num_trials
        self.seed_start = seed_start
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("ExperimentRunner initialized: trials=%d seed_start=%d output=%s",
                    num_trials, seed_start, output_dir)

    # ------------------------------------------------------------------ #
    #  Core trial executor
    # ------------------------------------------------------------------ #

    def _single_trial(
        self,
        seed: int,
        camera_pos: Optional[Tuple] = None,
        camera_target: Optional[Tuple] = None,
        detector_config: Optional[Dict] = None,
    ) -> Dict:
        """Execute one complete trial and return a results dict.

        Steps:
            1. Create SimulationEnvironment(gui=False)
            2. Call spawn_randomized(seed=seed)
            3. Settle 500 steps (already done inside spawn_randomized)
            4. Capture RGB-D from VirtualCamera
            5. Run FeatureDetector.detect_all()
            6. Run CoordinateTransformer on all detections
            7. Get ground truth from env.get_ground_truth()
            8. Compute all metrics
            9. Close environment
            10. Return results dict

        Args:
            seed: Integer seed for reproducible object placement.
            camera_pos: Camera position tuple; defaults to DEFAULT_CAM_POS.
            camera_target: Camera target tuple; defaults to DEFAULT_CAM_TARGET.
            detector_config: Optional config overrides for FeatureDetector.

        Returns:
            Dict with per-object-class metric values.
        """
        cam_pos = camera_pos or DEFAULT_CAM_POS
        cam_target = camera_target or DEFAULT_CAM_TARGET

        env = SimulationEnvironment(gui=False)
        try:
            env.setup()
            env.spawn_randomized(seed=seed)

            camera = VirtualCamera(
                position=cam_pos,
                target=cam_target,
                up_vector=DEFAULT_CAM_UP,
                width=640,
                height=480,
                fov=60.0,
            )
            rgb_image, depth_image = camera.capture()
            cam_params = camera.get_camera_params()

            transformer = CoordinateTransformer(
                intrinsic_matrix=cam_params["intrinsic_matrix"],
                extrinsic_matrix=cam_params["extrinsic_matrix"],
            )

            detector = FeatureDetector(config=detector_config)
            (features, latency_ms) = processing_latency(detector.detect_all, rgb_image)

            gt = env.get_ground_truth()

            # Build world coordinates for all detections
            world_coords: Dict[str, List] = {}
            for feat_type, feat_list in features.items():
                coords = []
                for feat in feat_list:
                    u, v = feat.pixel_coords
                    v_c = int(np.clip(v, 0, depth_image.shape[0] - 1))
                    u_c = int(np.clip(u, 0, depth_image.shape[1] - 1))
                    wc = transformer.pixel_to_world(u_c, v_c, depth_image[v_c, u_c])
                    coords.append(wc)
                world_coords[feat_type] = coords

            result = self._compute_metrics(
                seed, features, world_coords, gt, transformer,
                depth_image, latency_ms, rgb_image.shape
            )
        finally:
            env.close()

        return result

    def _compute_metrics(
        self,
        seed: int,
        features: Dict,
        world_coords: Dict,
        gt: Dict,
        transformer,
        depth_image: np.ndarray,
        latency_ms: float,
        image_shape: Tuple,
    ) -> Dict:
        """Compute all metrics for one trial.

        Args:
            seed: Trial seed.
            features: Detected features dict.
            world_coords: Detected world coordinates per feature type.
            gt: Ground truth dict from env.get_ground_truth().
            transformer: CoordinateTransformer instance.
            depth_image: Depth image array.
            latency_ms: Total detection latency in milliseconds.
            image_shape: (H, W, C) of the RGB image.

        Returns:
            Dict with all metric results.
        """
        h, w = image_shape[:2]

        # Map GT feature types to detector output keys
        gt_type_map = {"hole": "holes", "surface": "surfaces", "handle": "handles"}

        per_object = {}
        for obj_name, obj_gt in gt.items():
            feat_key = gt_type_map.get(obj_gt["type"], obj_gt["type"] + "s")
            det_list = features.get(feat_key, [])
            det_pixels = [f.pixel_coords for f in det_list]
            det_worlds = world_coords.get(feat_key, [])

            # Project GT world position to pixel
            gt_proj = transformer.world_to_pixel(obj_gt["position"])
            gt_px = gt_proj[:2] if gt_proj is not None else None

            # Detection rate (binary: did any detection match this GT?)
            detected = False
            best_px_err = float("inf")
            best_3d_err = float("inf")
            best_axes = {"x": float("inf"), "y": float("inf"), "z": float("inf")}
            best_conf = 0.0
            best_iou = 0.0

            if gt_px is not None:
                rate = detection_rate(det_pixels, [gt_px])
                detected = rate > 0.0

                for i, feat in enumerate(det_list):
                    det_px = feat.pixel_coords
                    px_err = pixel_localization_error(det_px, obj_gt["position"], transformer)
                    if px_err < best_px_err:
                        best_px_err = px_err
                        best_conf = feat.confidence
                        if i < len(det_worlds):
                            best_3d_err = position_error_3d(det_worlds[i], obj_gt["position"])
                            best_axes = per_axis_mae(det_worlds[i], obj_gt["position"])

                        if obj_gt["type"] == "hole" and feat.radius is not None:
                            gt_radius_px = obj_gt.get("feature_radius", 0.012) / 0.001  # approx
                            best_iou = circle_iou(
                                det_px, feat.radius, gt_px,
                                gt_radius_px, (h, w)
                            )

            per_object[obj_name] = {
                "seed": seed,
                "object_class": obj_name,
                "feature_type": obj_gt["type"],
                "detected": int(detected),
                "pixel_error": best_px_err if best_px_err != float("inf") else -1.0,
                "iou": best_iou,
                "confidence": best_conf,
                "latency_ms": latency_ms,
                "detected_x": det_worlds[0][0] if detected and det_worlds else float("nan"),
                "detected_y": det_worlds[0][1] if detected and det_worlds else float("nan"),
                "detected_z": det_worlds[0][2] if detected and det_worlds else float("nan"),
                "gt_x": obj_gt["position"][0],
                "gt_y": obj_gt["position"][1],
                "gt_z": obj_gt["position"][2],
                "error_x": best_axes["x"] if best_axes["x"] != float("inf") else float("nan"),
                "error_y": best_axes["y"] if best_axes["y"] != float("inf") else float("nan"),
                "error_z": best_axes["z"] if best_axes["z"] != float("inf") else float("nan"),
                "error_total": best_3d_err if best_3d_err != float("inf") else float("nan"),
            }

        return per_object

    # ------------------------------------------------------------------ #
    #  Experiment 1: Detection Accuracy
    # ------------------------------------------------------------------ #

    def run_experiment_1_detection_accuracy(self) -> None:
        """Run Experiment 1: Feature Detection Accuracy.

        50 randomized trials per object class. Outputs
        results/experiment1_detection_accuracy.csv.
        """
        logger.info("=== Experiment 1: Detection Accuracy (%d trials) ===", self.num_trials)
        csv_path = os.path.join(self.output_dir, "experiment1_detection_accuracy.csv")
        fieldnames = [
            "trial", "seed", "object_class", "feature_type",
            "detected", "pixel_error", "iou", "confidence", "latency_ms",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for trial in range(self.num_trials):
                seed = self.seed_start + trial
                logger.info("Exp1 trial %d/%d (seed=%d)", trial + 1, self.num_trials, seed)
                try:
                    result = self._single_trial(seed)
                    for obj_name, metrics in result.items():
                        writer.writerow({
                            "trial": trial,
                            "seed": seed,
                            "object_class": obj_name,
                            "feature_type": metrics["feature_type"],
                            "detected": metrics["detected"],
                            "pixel_error": metrics["pixel_error"],
                            "iou": metrics["iou"],
                            "confidence": metrics["confidence"],
                            "latency_ms": metrics["latency_ms"],
                        })
                    f.flush()
                except Exception as e:
                    logger.warning("Exp1 trial %d failed: %s", trial, e)

        logger.info("Experiment 1 complete → %s", csv_path)

    # ------------------------------------------------------------------ #
    #  Experiment 2: 3D Coordinate Precision
    # ------------------------------------------------------------------ #

    def run_experiment_2_coordinate_precision(self) -> None:
        """Run Experiment 2: 3D Coordinate Precision.

        50 randomized trials. Outputs
        results/experiment2_coordinate_precision.csv.
        """
        logger.info("=== Experiment 2: 3D Coordinate Precision (%d trials) ===", self.num_trials)
        csv_path = os.path.join(self.output_dir, "experiment2_coordinate_precision.csv")
        fieldnames = [
            "trial", "seed", "object_class",
            "detected_x", "detected_y", "detected_z",
            "gt_x", "gt_y", "gt_z",
            "error_x", "error_y", "error_z", "error_total",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for trial in range(self.num_trials):
                seed = self.seed_start + trial
                logger.info("Exp2 trial %d/%d (seed=%d)", trial + 1, self.num_trials, seed)
                try:
                    result = self._single_trial(seed)
                    for obj_name, metrics in result.items():
                        writer.writerow({
                            "trial": trial,
                            "seed": seed,
                            "object_class": obj_name,
                            "detected_x": metrics["detected_x"],
                            "detected_y": metrics["detected_y"],
                            "detected_z": metrics["detected_z"],
                            "gt_x": metrics["gt_x"],
                            "gt_y": metrics["gt_y"],
                            "gt_z": metrics["gt_z"],
                            "error_x": metrics["error_x"],
                            "error_y": metrics["error_y"],
                            "error_z": metrics["error_z"],
                            "error_total": metrics["error_total"],
                        })
                    f.flush()
                except Exception as e:
                    logger.warning("Exp2 trial %d failed: %s", trial, e)

        logger.info("Experiment 2 complete → %s", csv_path)

    # ------------------------------------------------------------------ #
    #  Experiment 3: Robustness
    # ------------------------------------------------------------------ #

    def run_experiment_3_robustness(self) -> None:
        """Run Experiment 3: Robustness Testing.

        3a: Varying orientation (0°–360° in 30° steps).
        3b: Varying camera distance (0.3m–1.0m in 0.1m steps).
        3c: Multi-object clutter (20 random layouts).
        """
        self._run_3a_orientation()
        self._run_3b_distance()
        self._run_3c_clutter()

    def _run_3a_orientation(self) -> None:
        """Experiment 3a: vary object yaw in fixed steps."""
        logger.info("=== Experiment 3a: Orientation Robustness ===")
        csv_path = os.path.join(self.output_dir, "experiment3a_orientation.csv")
        yaw_steps = list(range(0, 360, 30))  # 12 orientations
        fieldnames = ["yaw_deg", "object_class", "detection_rate",
                      "mean_pixel_error", "mean_3d_error"]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for yaw_deg in yaw_steps:
                logger.info("3a: yaw=%d°", yaw_deg)
                # Run 5 trials per orientation, same fixed seed family
                obj_detections: Dict[str, List] = {}
                obj_px_err: Dict[str, List] = {}
                obj_3d_err: Dict[str, List] = {}

                for rep in range(5):
                    seed = 3000 + yaw_deg * 10 + rep
                    try:
                        result = self._single_trial_fixed_yaw(seed, yaw_deg)
                        for obj_name, metrics in result.items():
                            obj_detections.setdefault(obj_name, []).append(metrics["detected"])
                            if metrics["pixel_error"] >= 0:
                                obj_px_err.setdefault(obj_name, []).append(metrics["pixel_error"])
                            if not np.isnan(metrics["error_total"]):
                                obj_3d_err.setdefault(obj_name, []).append(metrics["error_total"])
                    except Exception as e:
                        logger.warning("3a yaw=%d rep=%d failed: %s", yaw_deg, rep, e)

                for obj_name in obj_detections:
                    writer.writerow({
                        "yaw_deg": yaw_deg,
                        "object_class": obj_name,
                        "detection_rate": float(np.mean(obj_detections[obj_name])),
                        "mean_pixel_error": float(np.mean(obj_px_err.get(obj_name, [float("nan")]))),
                        "mean_3d_error": float(np.mean(obj_3d_err.get(obj_name, [float("nan")]))),
                    })
                f.flush()

        logger.info("Experiment 3a complete → %s", csv_path)

    def _single_trial_fixed_yaw(self, seed: int, yaw_deg: float) -> Dict:
        """Single trial where all objects share the same fixed yaw.

        Args:
            seed: Random seed for XY position.
            yaw_deg: Fixed yaw angle in degrees for all objects.

        Returns:
            Metrics dict same format as _single_trial.
        """
        import pybullet as p_mod

        yaw_rad = np.radians(yaw_deg)
        env = SimulationEnvironment(gui=False)
        try:
            env.setup()
            # Randomize XY but fix yaw
            np.random.seed(seed)
            positions = []
            min_sep = 0.1
            table_z = 0.42
            for _ in range(3):
                for _ in range(200):
                    x = np.random.uniform(-0.25, 0.25)
                    y = np.random.uniform(-0.35, 0.35)
                    cand = np.array([x, y])
                    if all(np.linalg.norm(cand - np.array(prev)) >= min_sep for prev in positions):
                        positions.append([x, y])
                        break
                else:
                    positions.append([np.random.uniform(-0.25, 0.25),
                                      np.random.uniform(-0.35, 0.35)])

            orn = p_mod.getQuaternionFromEuler([0, 0, yaw_rad])

            # Remove existing and respawn with fixed yaw
            for name, oid in list(env.objects.items()):
                if not name.startswith("_"):
                    p_mod.removeBody(oid)
            env.objects.clear()

            washer_h = 0.006
            env.objects["washer"] = env._create_washer(
                position=[positions[0][0], positions[0][1], table_z + washer_h],
                outer_radius=0.03, inner_radius=0.012, height=washer_h, orientation=orn,
            )
            bh = [0.05, 0.035, 0.03]
            bc = p_mod.createCollisionShape(p_mod.GEOM_BOX, halfExtents=bh)
            bv = p_mod.createVisualShape(p_mod.GEOM_BOX, halfExtents=bh, rgbaColor=[0.2, 0.5, 0.8, 1.0])
            env.objects["box"] = p_mod.createMultiBody(
                baseMass=0.1, baseCollisionShapeIndex=bc, baseVisualShapeIndex=bv,
                basePosition=[positions[1][0], positions[1][1], table_z + bh[2]], baseOrientation=orn,
            )
            env.objects["mug"] = env._create_mug(
                position=[positions[2][0], positions[2][1], table_z + 0.04], orientation=orn,
            )
            for _ in range(500):
                p_mod.stepSimulation()

            camera = VirtualCamera(position=DEFAULT_CAM_POS, target=DEFAULT_CAM_TARGET,
                                   up_vector=DEFAULT_CAM_UP, width=640, height=480, fov=60.0)
            rgb_image, depth_image = camera.capture()
            cam_params = camera.get_camera_params()
            transformer = CoordinateTransformer(
                intrinsic_matrix=cam_params["intrinsic_matrix"],
                extrinsic_matrix=cam_params["extrinsic_matrix"],
            )
            detector = FeatureDetector()
            features, latency_ms = processing_latency(detector.detect_all, rgb_image)

            world_coords: Dict[str, List] = {}
            for feat_type, feat_list in features.items():
                coords = []
                for feat in feat_list:
                    u, v = feat.pixel_coords
                    vc = int(np.clip(v, 0, depth_image.shape[0] - 1))
                    uc = int(np.clip(u, 0, depth_image.shape[1] - 1))
                    coords.append(transformer.pixel_to_world(uc, vc, depth_image[vc, uc]))
                world_coords[feat_type] = coords

            gt = env.get_ground_truth()
            return self._compute_metrics(seed, features, world_coords, gt, transformer,
                                         depth_image, latency_ms, rgb_image.shape)
        finally:
            env.close()

    def _run_3b_distance(self) -> None:
        """Experiment 3b: vary camera height/distance."""
        logger.info("=== Experiment 3b: Distance Robustness ===")
        csv_path = os.path.join(self.output_dir, "experiment3b_distance.csv")
        distances = [round(d, 1) for d in np.arange(0.3, 1.1, 0.1)]
        fieldnames = ["camera_distance_m", "object_class", "detection_rate",
                      "mean_pixel_error", "mean_3d_error"]
        table_height = 0.42

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for dist in distances:
                logger.info("3b: distance=%.1fm", dist)
                cam_pos = (0.0, 0.0, table_height + dist)
                cam_target = (0.0, 0.0, table_height)
                cam_up = (0, 1, 0)

                obj_detections: Dict[str, List] = {}
                obj_px_err: Dict[str, List] = {}
                obj_3d_err: Dict[str, List] = {}

                for rep in range(5):
                    seed = 4000 + int(dist * 10) + rep
                    try:
                        env = SimulationEnvironment(gui=False)
                        try:
                            env.setup()
                            env.spawn_randomized(seed=seed)
                            camera = VirtualCamera(position=cam_pos, target=cam_target,
                                                   up_vector=cam_up, width=640, height=480, fov=60.0)
                            rgb_image, depth_image = camera.capture()
                            cam_params = camera.get_camera_params()
                            transformer = CoordinateTransformer(
                                intrinsic_matrix=cam_params["intrinsic_matrix"],
                                extrinsic_matrix=cam_params["extrinsic_matrix"],
                            )
                            detector = FeatureDetector()
                            features, lat = processing_latency(detector.detect_all, rgb_image)
                            wc: Dict[str, List] = {}
                            for ft, fl in features.items():
                                wc[ft] = []
                                for feat in fl:
                                    u, v = feat.pixel_coords
                                    vc = int(np.clip(v, 0, depth_image.shape[0] - 1))
                                    uc = int(np.clip(u, 0, depth_image.shape[1] - 1))
                                    wc[ft].append(transformer.pixel_to_world(uc, vc, depth_image[vc, uc]))
                            gt = env.get_ground_truth()
                            result = self._compute_metrics(seed, features, wc, gt, transformer,
                                                          depth_image, lat, rgb_image.shape)
                        finally:
                            env.close()

                        for obj_name, metrics in result.items():
                            obj_detections.setdefault(obj_name, []).append(metrics["detected"])
                            if metrics["pixel_error"] >= 0:
                                obj_px_err.setdefault(obj_name, []).append(metrics["pixel_error"])
                            if not np.isnan(metrics["error_total"]):
                                obj_3d_err.setdefault(obj_name, []).append(metrics["error_total"])
                    except Exception as e:
                        logger.warning("3b dist=%.1f rep=%d failed: %s", dist, rep, e)

                for obj_name in obj_detections:
                    writer.writerow({
                        "camera_distance_m": dist,
                        "object_class": obj_name,
                        "detection_rate": float(np.mean(obj_detections[obj_name])),
                        "mean_pixel_error": float(np.mean(obj_px_err.get(obj_name, [float("nan")]))),
                        "mean_3d_error": float(np.mean(obj_3d_err.get(obj_name, [float("nan")]))),
                    })
                f.flush()

        logger.info("Experiment 3b complete → %s", csv_path)

    def _run_3c_clutter(self) -> None:
        """Experiment 3c: multi-object clutter."""
        logger.info("=== Experiment 3c: Clutter Robustness ===")
        csv_path = os.path.join(self.output_dir, "experiment3c_clutter.csv")
        fieldnames = ["trial", "num_objects", "object_class", "detection_rate"]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for trial in range(20):
                seed = 5000 + trial
                logger.info("3c: trial %d/20 (seed=%d)", trial + 1, seed)
                try:
                    result = self._single_trial(seed)
                    num_objects = len(result)
                    for obj_name, metrics in result.items():
                        writer.writerow({
                            "trial": trial,
                            "num_objects": num_objects,
                            "object_class": obj_name,
                            "detection_rate": metrics["detected"],
                        })
                    f.flush()
                except Exception as e:
                    logger.warning("3c trial %d failed: %s", trial, e)

        logger.info("Experiment 3c complete → %s", csv_path)

    # ------------------------------------------------------------------ #
    #  Parameter Sensitivity
    # ------------------------------------------------------------------ #

    def run_parameter_sensitivity(self) -> None:
        """Sweep key algorithm parameters and measure detection rate.

        Sweeps:
            - Hough param2: [15, 20, 25, 30, 35, 40, 45]
            - Canny thresholds: [(30,100), (40,120), (50,150), (60,180), (70,200)]
            - Surface min_area: [200, 500, 1000, 2000, 3000]

        Outputs results/parameter_sensitivity.csv.
        """
        logger.info("=== Parameter Sensitivity Analysis ===")
        csv_path = os.path.join(self.output_dir, "parameter_sensitivity.csv")
        fieldnames = ["parameter_name", "parameter_value", "detection_rate", "object_class"]
        num_sweep_trials = 10  # trials per parameter value

        sweeps = [
            ("hough_param2", [15, 20, 25, 30, 35, 40, 45], {}),
            ("canny_thresholds", ["30-100", "40-120", "50-150", "60-180", "70-200"], {}),
            ("surface_min_area", [200, 500, 1000, 2000, 3000], {}),
        ]

        canny_map = {
            "30-100": (30, 100), "40-120": (40, 120), "50-150": (50, 150),
            "60-180": (60, 180), "70-200": (70, 200),
        }

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for param_name, values, _ in sweeps:
                logger.info("Sweeping %s: %s", param_name, values)
                for val in values:
                    if param_name == "hough_param2":
                        cfg = {"hough_param2": val}
                        val_str = str(val)
                    elif param_name == "canny_thresholds":
                        low, high = canny_map[val]
                        cfg = {"canny_low": low, "canny_high": high}
                        val_str = val
                    else:
                        cfg = {"surface_min_area": val}
                        val_str = str(val)

                    obj_detections: Dict[str, List] = {}
                    for trial in range(num_sweep_trials):
                        seed = 6000 + hash(param_name + str(val)) % 1000 + trial
                        seed = seed % (2 ** 31)
                        try:
                            result = self._single_trial(seed, detector_config=cfg)
                            for obj_name, metrics in result.items():
                                obj_detections.setdefault(obj_name, []).append(metrics["detected"])
                        except Exception as e:
                            logger.warning("Sensitivity %s=%s trial %d failed: %s",
                                           param_name, val_str, trial, e)

                    for obj_name, dets in obj_detections.items():
                        writer.writerow({
                            "parameter_name": param_name,
                            "parameter_value": val_str,
                            "detection_rate": float(np.mean(dets)),
                            "object_class": obj_name,
                        })
                    f.flush()

        logger.info("Parameter sensitivity complete → %s", csv_path)

    # ------------------------------------------------------------------ #
    #  Run All
    # ------------------------------------------------------------------ #

    def run_all(self) -> None:
        """Run all experiments sequentially."""
        logger.info("Starting full experiment suite...")
        self.run_experiment_1_detection_accuracy()
        self.run_experiment_2_coordinate_precision()
        self.run_experiment_3_robustness()
        self.run_parameter_sensitivity()
        logger.info("All experiments complete. Results in: %s", self.output_dir)


if __name__ == "__main__":
    runner = ExperimentRunner(num_trials=50, seed_start=0, output_dir="results")
    runner.run_all()
