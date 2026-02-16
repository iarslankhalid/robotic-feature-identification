"""
Main Entry Point
==================
Ties together the full pipeline:
  1. Launch simulation environment
  2. Capture RGB-D data from virtual camera
  3. Run feature detection algorithms
  4. Map detected features to 3D world coordinates
  5. Visualize and report results
"""

import argparse
import sys
import time

import numpy as np

from simulation.environment import SimulationEnvironment
from src.camera import VirtualCamera
from src.feature_detector import FeatureDetector
from src.coordinate_transform import CoordinateTransformer
from src.visualization import (
    draw_features_on_image,
    show_rgb_depth_side_by_side,
    show_detection_results,
    save_detection_report,
)


def run_pipeline(gui=True, save_report=True, show_plots=True):
    """Execute the full perception pipeline."""

    print("[1/5] Setting up simulation environment...")
    env = SimulationEnvironment(gui=gui)
    env.setup()

    # Let objects settle
    for _ in range(500):
        env.step()
        if gui:
            time.sleep(1 / 240.0)

    print("  Objects:", list(env.get_object_positions().keys()))

    # ------------------------------------------------------------------ #
    print("[2/5] Configuring virtual camera and capturing RGB-D frame...")
    camera = VirtualCamera(
        position=(0.0, -0.5, 0.9),
        target=(0.0, 0.0, 0.42),
        up_vector=(0, 0, 1),
        width=640,
        height=480,
        fov=60.0,
    )
    rgb_image, depth_image = camera.capture()
    cam_params = camera.get_camera_params()

    print(f"  RGB shape:   {rgb_image.shape}")
    print(f"  Depth shape: {depth_image.shape}")
    print(f"  Depth range: [{depth_image.min():.3f}, {depth_image.max():.3f}] m")

    # ------------------------------------------------------------------ #
    print("[3/5] Running feature detection...")
    detector = FeatureDetector()
    features = detector.detect_all(rgb_image)

    for feat_type, feat_list in features.items():
        print(f"  {feat_type}: {len(feat_list)} detected")

    # ------------------------------------------------------------------ #
    print("[4/5] Computing 3D world coordinates...")
    transformer = CoordinateTransformer(
        intrinsic_matrix=cam_params["intrinsic_matrix"],
        extrinsic_matrix=cam_params["extrinsic_matrix"],
    )

    world_coords = {}
    for feat_type, feat_list in features.items():
        coords = []
        for feat in feat_list:
            u, v = feat.pixel_coords
            wc = transformer.pixel_to_world(u, v, depth_image[v, u])
            coords.append(wc)
            print(f"  {feat.feature_type} at pixel ({u}, {v}) → "
                  f"world [{wc[0]:.4f}, {wc[1]:.4f}, {wc[2]:.4f}]")
        world_coords[feat_type] = coords

    # Ground truth comparison
    print("\n  Ground truth object positions:")
    for name, info in env.get_object_positions().items():
        pos = info["position"]
        print(f"    {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    # ------------------------------------------------------------------ #
    print("[5/5] Generating visualizations...")

    annotated = draw_features_on_image(rgb_image, features, world_coords)

    if save_report:
        report = save_detection_report(features, world_coords)
        print(report)

    if show_plots:
        show_rgb_depth_side_by_side(rgb_image, depth_image)
        show_detection_results(annotated)

    print("\nPipeline complete.")

    if gui:
        print("Simulation still running — press Ctrl+C to exit.")
        try:
            while True:
                env.step()
                time.sleep(1 / 240.0)
        except KeyboardInterrupt:
            pass

    env.close()
    return features, world_coords


def main():
    parser = argparse.ArgumentParser(
        description="Vision-Based Robotic Feature Identification"
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Run in headless mode (no PyBullet GUI)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip matplotlib visualization windows",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip saving detection report",
    )
    args = parser.parse_args()

    run_pipeline(
        gui=not args.no_gui,
        save_report=not args.no_report,
        show_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
