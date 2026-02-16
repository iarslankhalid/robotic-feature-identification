"""
Visualization Module
=====================
Draw detection results as overlays on the camera feed.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


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

    # Convert back to RGB
    annotated = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    plt.savefig("reports/rgbd_view.png", dpi=150, bbox_inches="tight")
    plt.show()


def show_detection_results(annotated_image, title="Detection Results"):
    """Display annotated detection results."""
    plt.figure(figsize=(10, 7))
    plt.imshow(annotated_image)
    plt.title(title)
    plt.axis("off")
    plt.savefig("reports/detection_results.png", dpi=150, bbox_inches="tight")
    plt.show()


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
    with open(filepath, "w") as f:
        f.write(report)
    print(f"Report saved to {filepath}")
    return report
