"""
Results Visualization Module
==============================
Generates publication-quality charts from experiment CSV results.
"""

import csv
import os
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.logger import setup_logger

logger = setup_logger(__name__)


def _read_csv(csv_path: str) -> List[Dict]:
    """Read a CSV file into a list of row dicts.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of dicts, one per row.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def plot_detection_accuracy_table(csv_path: str, output_path: str) -> None:
    """Bar chart of detection rate per object class from Experiment 1.

    Args:
        csv_path: Path to experiment1_detection_accuracy.csv.
        output_path: Where to save the PNG.
    """
    if not os.path.exists(csv_path):
        logger.warning("CSV not found: %s — skipping chart", csv_path)
        return

    rows = _read_csv(csv_path)
    obj_detected: Dict[str, List] = {}
    for row in rows:
        obj = row["object_class"]
        obj_detected.setdefault(obj, []).append(int(row["detected"]))

    objects = sorted(obj_detected.keys())
    rates = [float(np.mean(obj_detected[o])) * 100.0 for o in objects]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    bars = ax.bar(objects, rates, color=colors[: len(objects)], edgecolor="black", linewidth=0.8)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 115)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_title("Feature Detection Accuracy by Object Class", fontsize=13, fontweight="bold")
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.4)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved detection accuracy chart → %s", output_path)


def plot_3d_error_boxplots(csv_path: str, output_path: str) -> None:
    """Box plots of per-axis 3D error distributions from Experiment 2.

    Args:
        csv_path: Path to experiment2_coordinate_precision.csv.
        output_path: Where to save the PNG.
    """
    if not os.path.exists(csv_path):
        logger.warning("CSV not found: %s — skipping chart", csv_path)
        return

    rows = _read_csv(csv_path)
    obj_errors: Dict[str, Dict[str, List]] = {}
    for row in rows:
        obj = row["object_class"]
        obj_errors.setdefault(obj, {"x": [], "y": [], "z": [], "total": []})
        for axis in ("x", "y", "z", "total"):
            key = f"error_{axis}"
            try:
                val = float(row[key])
                if not np.isnan(val):
                    obj_errors[obj][axis].append(val)
            except (ValueError, KeyError):
                pass

    objects = sorted(obj_errors.keys())
    axes_labels = ["X", "Y", "Z", "Total"]
    axes_keys = ["x", "y", "z", "total"]

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = []
    data_to_plot = []
    tick_labels = []
    n_axes = len(axes_keys)

    for i, obj in enumerate(objects):
        for j, key in enumerate(axes_keys):
            pos = i * (n_axes + 1) + j + 1
            positions.append(pos)
            data_to_plot.append(obj_errors[obj].get(key, [0]))
            tick_labels.append(f"{obj}\n{axes_labels[j]}")

    ax.boxplot(data_to_plot, positions=positions, patch_artist=True,
               boxprops=dict(facecolor="#90CAF9", alpha=0.7),
               medianprops=dict(color="red", linewidth=2))
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("Error (m)", fontsize=12)
    ax.set_title("3D Coordinate Precision — Per-Axis MAE", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved 3D error boxplots → %s", output_path)


def plot_robustness_curves(csv_dir: str, output_path: str) -> None:
    """Line plots of detection rate vs orientation and distance from Experiment 3.

    Args:
        csv_dir: Directory containing experiment3a and experiment3b CSVs.
        output_path: Where to save the PNG.
    """
    path_3a = os.path.join(csv_dir, "experiment3a_orientation.csv")
    path_3b = os.path.join(csv_dir, "experiment3b_distance.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Robustness Analysis", fontsize=14, fontweight="bold")

    # 3a: Orientation
    if os.path.exists(path_3a):
        rows = _read_csv(path_3a)
        obj_yaw: Dict[str, Dict] = {}
        for row in rows:
            obj = row["object_class"]
            yaw = float(row["yaw_deg"])
            rate = float(row["detection_rate"])
            obj_yaw.setdefault(obj, {"yaws": [], "rates": []})
            obj_yaw[obj]["yaws"].append(yaw)
            obj_yaw[obj]["rates"].append(rate * 100.0)

        for obj, data in sorted(obj_yaw.items()):
            axes[0].plot(data["yaws"], data["rates"], marker="o", label=obj)
        axes[0].set_xlabel("Object Yaw (°)", fontsize=11)
        axes[0].set_ylabel("Detection Rate (%)", fontsize=11)
        axes[0].set_title("Detection Rate vs Orientation")
        axes[0].legend()
        axes[0].grid(alpha=0.4)
        axes[0].set_ylim(-5, 115)
    else:
        axes[0].text(0.5, 0.5, "Data not available", ha="center", va="center",
                     transform=axes[0].transAxes)

    # 3b: Distance
    if os.path.exists(path_3b):
        rows = _read_csv(path_3b)
        obj_dist: Dict[str, Dict] = {}
        for row in rows:
            obj = row["object_class"]
            dist = float(row["camera_distance_m"])
            rate = float(row["detection_rate"])
            obj_dist.setdefault(obj, {"dists": [], "rates": []})
            obj_dist[obj]["dists"].append(dist)
            obj_dist[obj]["rates"].append(rate * 100.0)

        for obj, data in sorted(obj_dist.items()):
            axes[1].plot(data["dists"], data["rates"], marker="s", label=obj)
        axes[1].set_xlabel("Camera Distance (m)", fontsize=11)
        axes[1].set_ylabel("Detection Rate (%)", fontsize=11)
        axes[1].set_title("Detection Rate vs Camera Distance")
        axes[1].legend()
        axes[1].grid(alpha=0.4)
        axes[1].set_ylim(-5, 115)
    else:
        axes[1].text(0.5, 0.5, "Data not available", ha="center", va="center",
                     transform=axes[1].transAxes)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved robustness curves → %s", output_path)


def plot_parameter_sensitivity(csv_path: str, output_path: str) -> None:
    """Line plots of detection rate vs parameter value from sensitivity analysis.

    Args:
        csv_path: Path to parameter_sensitivity.csv.
        output_path: Where to save the PNG.
    """
    if not os.path.exists(csv_path):
        logger.warning("CSV not found: %s — skipping chart", csv_path)
        return

    rows = _read_csv(csv_path)
    params: Dict[str, Dict] = {}
    for row in rows:
        param = row["parameter_name"]
        val = row["parameter_value"]
        rate = float(row["detection_rate"])
        params.setdefault(param, {"vals": [], "rates": []})
        params[param]["vals"].append(val)
        params[param]["rates"].append(rate * 100.0)

    n = len(params)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(5 * max(n, 1), 5))
    if n == 1:
        axes = [axes]

    for ax, (param_name, data) in zip(axes, params.items()):
        ax.plot(data["vals"], data["rates"], marker="D", color="#673AB7")
        ax.set_xlabel(param_name, fontsize=10)
        ax.set_ylabel("Detection Rate (%)", fontsize=10)
        ax.set_title(f"Sensitivity: {param_name}", fontsize=11)
        ax.grid(alpha=0.4)
        ax.set_ylim(-5, 115)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Parameter Sensitivity Analysis", fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved parameter sensitivity plots → %s", output_path)


def generate_all_figures(results_dir: str = "results") -> None:
    """Generate all figures from experiment results CSVs.

    Args:
        results_dir: Directory containing the CSV files and where PNGs are saved.
    """
    logger.info("Generating all figures from %s ...", results_dir)

    plot_detection_accuracy_table(
        os.path.join(results_dir, "experiment1_detection_accuracy.csv"),
        os.path.join(results_dir, "detection_accuracy_chart.png"),
    )
    plot_3d_error_boxplots(
        os.path.join(results_dir, "experiment2_coordinate_precision.csv"),
        os.path.join(results_dir, "3d_error_boxplots.png"),
    )
    plot_robustness_curves(
        results_dir,
        os.path.join(results_dir, "robustness_curves.png"),
    )
    plot_parameter_sensitivity(
        os.path.join(results_dir, "parameter_sensitivity.csv"),
        os.path.join(results_dir, "parameter_sensitivity_plots.png"),
    )

    logger.info("All figures generated in %s", results_dir)


if __name__ == "__main__":
    generate_all_figures()
