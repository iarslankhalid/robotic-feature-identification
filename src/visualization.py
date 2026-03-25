"""
Visualization Module
=====================
Draw detection results as overlays on the camera feed.
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.logger import setup_logger

logger = setup_logger(__name__)


# Color map for feature types (BGR for OpenCV)
COLORS = {
    "hole": (0, 255, 0),       # green
    "surface": (255, 165, 0),  # orange
    "handle": (0, 0, 255),     # red
}

LABEL_COLORS = {
    "hole": (0, 200, 0),
    "surface": (200, 130, 0),
    "handle": (0, 0, 200),
}


def draw_approach_vectors(img_bgr: np.ndarray, features_dict: dict) -> np.ndarray:
    """Draw approach vector arrows on a BGR image.

    Arrows indicate the intended robot approach direction for each detected feature:
      - Holes: downward arrow (insertion approach)
      - Surfaces: upward arrow (suction approach)
      - Handles: lateral arrow (grasp approach)

    Args:
        img_bgr: np.ndarray (H, W, 3) in BGR format — modified in place and returned.
        features_dict: Dict from FeatureDetector.detect_all().

    Returns:
        The same img_bgr with arrows drawn.
    """
    arrow_len = 40  # pixels

    for feat_type, features in features_dict.items():
        color = COLORS.get(feat_type, (255, 255, 255))

        for feat in features:
            u, v = feat.pixel_coords

            if feat_type == "holes":
                # Insertion: arrow pointing down
                pt1 = (u, v - arrow_len // 2)
                pt2 = (u, v + arrow_len // 2)
            elif feat_type == "surfaces":
                # Suction: arrow pointing up (away from surface)
                pt1 = (u, v + arrow_len // 2)
                pt2 = (u, v - arrow_len // 2)
            else:  # handles
                # Lateral grasp: arrow pointing right
                pt1 = (u - arrow_len // 2, v)
                pt2 = (u + arrow_len // 2, v)

            cv2.arrowedLine(img_bgr, pt1, pt2, color, 2, tipLength=0.3)

    return img_bgr


def draw_features_on_image(rgb_image, features_dict, world_coords=None):
    """
    Draw detected features as overlays on an RGB image.

    Args:
        rgb_image: np.ndarray (H, W, 3) in RGB format.
        features_dict: Dict from FeatureDetector.detect_all().
        world_coords: Optional dict mapping feature_type to list of 3D coords.

    Returns:
        Annotated image (RGB).
    """
    # Work on a copy in BGR for OpenCV drawing
    img = cv2.cvtColor(rgb_image.copy(), cv2.COLOR_RGB2BGR)

    for feat_type, features in features_dict.items():
        color = COLORS.get(feat_type, (255, 255, 255))
        label_color = LABEL_COLORS.get(feat_type, (200, 200, 200))

        for i, feat in enumerate(features):
            u, v = feat.pixel_coords

            if feat.feature_type == "hole" and feat.radius is not None:
                # Draw circle
                cv2.circle(img, (u, v), int(feat.radius), color, 2)
                cv2.circle(img, (u, v), 3, color, -1)  # center dot
            elif feat.feature_type == "surface" and feat.pixel_region:
                x, y, w, h = feat.pixel_region[0]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.circle(img, (u, v), 4, color, -1)  # centroid
            elif feat.feature_type == "handle" and feat.pixel_region:
                x, y, w, h = feat.pixel_region[0]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.circle(img, (u, v), 4, color, -1)

            # Label
            label = f"{feat.feature_type} ({feat.confidence:.2f})"
            cv2.putText(
                img, label, (u + 5, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1,
            )

            # World coordinate annotation
            if world_coords and feat_type in world_coords and i < len(world_coords[feat_type]):
                wc = world_coords[feat_type][i]
                coord_str = f"[{wc[0]:.3f}, {wc[1]:.3f}, {wc[2]:.3f}]"
                cv2.putText(
                    img, coord_str, (u + 5, v + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1,
                )

    # Draw approach vectors
    draw_approach_vectors(img, features_dict)

    # Convert back to RGB
    annotated = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("draw_features_on_image: annotated %d feature types", len(features_dict))
    return annotated


def show_rgb_depth_side_by_side(rgb_image, depth_image, title="RGB-D View"):
    """Display RGB and depth images side by side using matplotlib."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(rgb_image)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    depth_display = axes[1].imshow(depth_image, cmap="viridis")
    axes[1].set_title("Depth Image")
    axes[1].axis("off")
    plt.colorbar(depth_display, ax=axes[1], label="Depth (m)")

    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/rgbd_view.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved RGB-D view to reports/rgbd_view.png")


def show_detection_results(annotated_image, title="Detection Results"):
    """Display annotated detection results."""
    plt.figure(figsize=(10, 7))
    plt.imshow(annotated_image)
    plt.title(title)
    plt.axis("off")
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/detection_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved detection results to reports/detection_results.png")


def create_summary_figure(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    features_dict: dict,
    world_coords: dict = None,
    output_path: str = "results/summary_figure.png",
) -> None:
    """Create a 2×2 publication-quality summary figure.

    Layout:
      - Top-left:  Raw RGB image
      - Top-right: Depth map with colorbar
      - Bottom-left: Annotated detections with approach vectors
      - Bottom-right: Detection confidence bar chart

    Args:
        rgb_image: np.ndarray (H, W, 3) RGB image.
        depth_image: np.ndarray (H, W) depth in meters.
        features_dict: Dict from FeatureDetector.detect_all().
        world_coords: Optional dict of world coordinates per feature type.
        output_path: File path for the saved PNG.
    """
    annotated = draw_features_on_image(rgb_image, features_dict, world_coords)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Raw RGB
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title("Raw RGB Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # Top-right: Depth map
    depth_plot = axes[0, 1].imshow(depth_image, cmap="viridis")
    axes[0, 1].set_title("Depth Map", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")
    plt.colorbar(depth_plot, ax=axes[0, 1], label="Depth (m)", fraction=0.046, pad=0.04)

    # Bottom-left: Annotated detections
    axes[1, 0].imshow(annotated)
    axes[1, 0].set_title("Detected Features + Approach Vectors", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    # Bottom-right: Confidence bar chart
    feat_labels = []
    confidences = []
    bar_colors = []
    color_map_rgb = {
        "holes": "#4CAF50",
        "surfaces": "#FF9800",
        "handles": "#F44336",
    }
    for feat_type, feat_list in features_dict.items():
        for i, feat in enumerate(feat_list):
            feat_labels.append(f"{feat_type[:-1]}{i}")
            confidences.append(feat.confidence)
            bar_colors.append(color_map_rgb.get(feat_type, "#9E9E9E"))

    if confidences:
        axes[1, 1].bar(feat_labels, confidences, color=bar_colors, edgecolor="black", linewidth=0.5)
        axes[1, 1].set_ylim(0, 1.15)
        axes[1, 1].set_ylabel("Confidence", fontsize=11)
        axes[1, 1].set_title("Detection Confidence", fontsize=12, fontweight="bold")
        axes[1, 1].tick_params(axis="x", rotation=30)
        axes[1, 1].grid(axis="y", alpha=0.4)
    else:
        axes[1, 1].text(0.5, 0.5, "No detections", ha="center", va="center",
                        transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].axis("off")

    fig.suptitle("Vision-Based Robotic Affordance Detection", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved summary figure → %s", output_path)


def save_detection_report(features_dict, world_coords, filepath="reports/detection_report.txt"):
    """Save a text report of all detected features and their world coordinates."""
    lines = ["=" * 60]
    lines.append("FEATURE DETECTION REPORT")
    lines.append("=" * 60)

    total = 0
    for feat_type, features in features_dict.items():
        lines.append(f"\n--- {feat_type.upper()} ---")
        lines.append(f"  Count: {len(features)}")
        for i, feat in enumerate(features):
            total += 1
            lines.append(f"  [{i}] pixel=({feat.pixel_coords[0]}, {feat.pixel_coords[1]})"
                         f"  confidence={feat.confidence:.3f}")
            if feat.radius is not None:
                lines.append(f"      radius={feat.radius:.1f}px")
            if feat.area is not None:
                lines.append(f"      area={feat.area:.1f}px²")
            if world_coords and feat_type in world_coords and i < len(world_coords[feat_type]):
                wc = world_coords[feat_type][i]
                lines.append(f"      world=[{wc[0]:.4f}, {wc[1]:.4f}, {wc[2]:.4f}]m")

    lines.append(f"\nTotal features detected: {total}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", filepath)
    return report
