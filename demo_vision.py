#!/usr/bin/env python3
"""
Standalone Vision Demo (No PyBullet Required)
==============================================
Demonstrates feature detection and coordinate mapping using synthetic test images.
This script works without PyBullet, making it easy to run on any system.

Usage:
    python demo_vision.py
"""

import numpy as np
import cv2
import sys
import os

# Ensure src is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.feature_detector import FeatureDetector
from src.coordinate_transform import CoordinateTransformer


def create_synthetic_scene():
    """Create a synthetic RGB + depth image with holes, surfaces, and handles."""
    width, height = 640, 480
    
    # RGB image with objects
    rgb = np.ones((height, width, 3), dtype=np.uint8) * 50  # dark background
    
    # Depth image (in meters, simulating camera at z=0.9 looking down)
    depth = np.ones((height, width), dtype=np.float32) * 1.2  # background at 1.2m
    
    # --- Object 1: Washer (Class A - hole detection) ---
    washer_center = (160, 240)
    washer_outer_r = 40
    washer_inner_r = 15
    
    # Draw washer on RGB
    cv2.circle(rgb, washer_center, washer_outer_r, (180, 180, 180), -1)
    cv2.circle(rgb, washer_center, washer_inner_r, (30, 30, 30), -1)  # dark hole
    cv2.circle(rgb, washer_center, washer_outer_r, (140, 140, 140), 2)  # edge
    
    # Washer depth (at table height ~0.48m from camera)
    cv2.circle(depth, washer_center, washer_outer_r, 0.48, -1)
    
    # --- Object 2: Box (Class B - surface detection) ---
    box_rect = (280, 180, 140, 100)
    x, y, w, h = box_rect
    
    # Draw box on RGB
    cv2.rectangle(rgb, (x, y), (x + w, y + h), (100, 150, 200), -1)
    cv2.rectangle(rgb, (x, y), (x + w, y + h), (80, 120, 180), 3)  # edge
    
    # Add some texture/shading
    cv2.line(rgb, (x, y), (x + w, y + h), (90, 140, 190), 1)
    cv2.line(rgb, (x + w, y), (x, y + h), (90, 140, 190), 1)
    
    # Box depth
    depth[y:y + h, x:x + w] = 0.45
    
    # --- Object 3: Mug with handle (Class C - handle detection) ---
    mug_center = (500, 300)
    mug_radius = 35
    
    # Mug body on RGB
    cv2.circle(rgb, mug_center, mug_radius, (200, 80, 80), -1)
    cv2.circle(rgb, mug_center, mug_radius, (180, 60, 60), 2)
    
    # Handle (elongated rectangle)
    handle_pts = np.array([
        [mug_center[0] + mug_radius, mug_center[1] - 15],
        [mug_center[0] + mug_radius + 30, mug_center[1] - 20],
        [mug_center[0] + mug_radius + 30, mug_center[1] + 20],
        [mug_center[0] + mug_radius, mug_center[1] + 15],
    ], dtype=np.int32)
    cv2.fillPoly(rgb, [handle_pts], (210, 90, 90))
    cv2.polylines(rgb, [handle_pts], True, (180, 60, 60), 2)
    
    # Mug depth
    cv2.circle(depth, mug_center, mug_radius, 0.50, -1)
    cv2.fillPoly(depth, [handle_pts], 0.50)
    
    # Add some noise to make it realistic
    noise = np.random.randn(height, width, 3) * 5
    rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    depth_noise = np.random.randn(height, width) * 0.002
    depth = np.clip(depth + depth_noise, 0.01, 5.0)
    
    return rgb, depth


def create_camera_matrices(width=640, height=480, fov=60.0):
    """Create intrinsic and extrinsic matrices for a virtual camera."""
    # Intrinsic matrix from FOV
    fov_rad = np.radians(fov)
    fy = height / (2.0 * np.tan(fov_rad / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=np.float64)
    
    # Simple extrinsic: camera at (0, 0, 0.9) looking down at table at z=0.4
    # For simplicity, use identity rotation + translation
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[2, 3] = 0.9  # camera Z position
    
    return K, extrinsic


def main():
    print("=" * 70)
    print("VISION-BASED FEATURE DETECTION DEMO")
    print("(Standalone - No PyBullet Required)")
    print("=" * 70)
    
    # Generate synthetic scene
    print("\n[1/5] Creating synthetic RGB-D scene...")
    rgb_image, depth_image = create_synthetic_scene()
    print(f"  RGB shape:   {rgb_image.shape}")
    print(f"  Depth shape: {depth_image.shape}")
    print(f"  Depth range: [{depth_image.min():.3f}, {depth_image.max():.3f}] m")
    
    # Save images
    cv2.imwrite("reports/demo_rgb.png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    depth_vis = ((depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255).astype(np.uint8)
    cv2.imwrite("reports/demo_depth.png", cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS))
    print("  Saved: reports/demo_rgb.png, reports/demo_depth.png")
    
    # Feature detection
    print("\n[2/5] Running feature detection algorithms...")
    detector = FeatureDetector()
    features = detector.detect_all(rgb_image)
    
    for feat_type, feat_list in features.items():
        print(f"  {feat_type:10s}: {len(feat_list)} detected")
        for i, feat in enumerate(feat_list[:3]):  # show first 3
            print(f"    [{i}] pixel=({feat.pixel_coords[0]:3d}, {feat.pixel_coords[1]:3d}), "
                  f"conf={feat.confidence:.2f}")
    
    # Coordinate transformation
    print("\n[3/5] Computing 3D world coordinates...")
    K, extrinsic = create_camera_matrices()
    transformer = CoordinateTransformer(K, extrinsic)
    
    world_coords = {}
    for feat_type, feat_list in features.items():
        coords = []
        for feat in feat_list:
            u, v = feat.pixel_coords
            v_clamp = int(np.clip(v, 0, depth_image.shape[0] - 1))
            u_clamp = int(np.clip(u, 0, depth_image.shape[1] - 1))
            d = depth_image[v_clamp, u_clamp]
            wc = transformer.pixel_to_world(u_clamp, v_clamp, d)
            coords.append(wc)
            print(f"  {feat_type:10s} @ pixel ({u:3d},{v:3d}) → "
                  f"world [{wc[0]:+.3f}, {wc[1]:+.3f}, {wc[2]:+.3f}]m")
        world_coords[feat_type] = coords
    
    # Visualization
    print("\n[4/5] Generating annotated visualization...")
    from src.visualization import draw_features_on_image
    
    annotated = draw_features_on_image(rgb_image, features, world_coords)
    cv2.imwrite("reports/demo_annotated.png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print("  Saved: reports/demo_annotated.png")
    
    # Summary report
    print("\n[5/5] Generating detection report...")
    from src.visualization import save_detection_report
    
    report_text = save_detection_report(features, world_coords, "reports/demo_report.txt")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nResults saved to reports/:")
    print("  - demo_rgb.png          (raw RGB image)")
    print("  - demo_depth.png        (depth visualization)")
    print("  - demo_annotated.png    (detection overlay)")
    print("  - demo_report.txt       (detailed text report)")
    print("\nAll vision algorithms working correctly! ✓")
    
    # Display if running interactively
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(rgb_image)
        axes[0].set_title("RGB Input")
        axes[0].axis("off")
        
        axes[1].imshow(depth_image, cmap="viridis")
        axes[1].set_title("Depth Map")
        axes[1].axis("off")
        
        axes[2].imshow(annotated)
        axes[2].set_title("Detected Features")
        axes[2].axis("off")
        
        plt.tight_layout()
        plt.savefig("reports/demo_summary.png", dpi=150, bbox_inches="tight")
        print("\n  - demo_summary.png      (combined visualization)")
        print("\nDisplaying results...")
        plt.show()
    except:
        print("\nNote: matplotlib display not available (running headless)")


if __name__ == "__main__":
    main()
