"""
Experiment Report Generator
============================
Reads all experiment CSV files and generates a markdown summary report.
"""

import csv
import os
from typing import Dict, List

import numpy as np

from src.logger import setup_logger

logger = setup_logger(__name__)


def _read_csv(path: str) -> List[Dict]:
    """Read a CSV into a list of row dicts.

    Args:
        path: File path to the CSV.

    Returns:
        List of dicts; empty list if file does not exist.
    """
    if not os.path.exists(path):
        logger.warning("CSV missing: %s", path)
        return []
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _safe_float(val: str) -> float:
    """Parse a string to float, returning NaN on failure.

    Args:
        val: String value to parse.

    Returns:
        Float or NaN.
    """
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


def generate_summary_report(
    results_dir: str = "results",
    output_path: str = "results/EXPERIMENT_REPORT.md",
) -> str:
    """Auto-generate a markdown report from all experiment CSVs.

    Sections:
        - Experiment 1: Detection rates per class
        - Experiment 2: Mean 3D errors per class and axis
        - Experiment 3: Key robustness findings
        - Parameter sensitivity: Best configurations

    Args:
        results_dir: Directory containing experiment CSV files and figures.
        output_path: Destination for the markdown report.

    Returns:
        The generated report string.
    """
    logger.info("Generating experiment report from %s ...", results_dir)

    lines = [
        "# Experiment Report — Vision-Based Robotic Affordance Detection",
        "",
        "Auto-generated report from experiment results.",
        "",
        "---",
        "",
    ]

    # ------------------------------------------------------------------ #
    # Experiment 1: Detection Accuracy
    # ------------------------------------------------------------------ #
    lines.append("## Experiment 1: Feature Detection Accuracy")
    lines.append("")
    rows1 = _read_csv(os.path.join(results_dir, "experiment1_detection_accuracy.csv"))
    if rows1:
        obj_stats: Dict[str, Dict] = {}
        for row in rows1:
            obj = row["object_class"]
            detected = int(row["detected"])
            obj_stats.setdefault(obj, {"detected": [], "pixel_error": [], "iou": []})
            obj_stats[obj]["detected"].append(detected)
            px_err = _safe_float(row.get("pixel_error", "nan"))
            if px_err >= 0:
                obj_stats[obj]["pixel_error"].append(px_err)
            iou = _safe_float(row.get("iou", "0"))
            obj_stats[obj]["iou"].append(iou)

        lines.append("| Object Class | Feature Type | Detection Rate | Mean Pixel Error (px) | Mean IoU |")
        lines.append("|---|---|---|---|---|")
        for obj in sorted(obj_stats):
            data = obj_stats[obj]
            rate = float(np.mean(data["detected"])) * 100
            mean_px = float(np.nanmean(data["pixel_error"])) if data["pixel_error"] else float("nan")
            mean_iou = float(np.nanmean(data["iou"])) if data["iou"] else float("nan")
            # Look up feature type from any row for this object
            feat_type = next((r["feature_type"] for r in rows1 if r["object_class"] == obj), "—")
            lines.append(f"| {obj} | {feat_type} | {rate:.1f}% | {mean_px:.2f} | {mean_iou:.3f} |")
    else:
        lines.append("_No data available._")

    lines.append("")
    if os.path.exists(os.path.join(results_dir, "detection_accuracy_chart.png")):
        lines.append("![Detection Accuracy Chart](detection_accuracy_chart.png)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ------------------------------------------------------------------ #
    # Experiment 2: 3D Coordinate Precision
    # ------------------------------------------------------------------ #
    lines.append("## Experiment 2: 3D Coordinate Precision")
    lines.append("")
    rows2 = _read_csv(os.path.join(results_dir, "experiment2_coordinate_precision.csv"))
    if rows2:
        obj_errs: Dict[str, Dict] = {}
        for row in rows2:
            obj = row["object_class"]
            obj_errs.setdefault(obj, {"x": [], "y": [], "z": [], "total": []})
            for ax in ("x", "y", "z", "total"):
                val = _safe_float(row.get(f"error_{ax}", "nan"))
                if not np.isnan(val):
                    obj_errs[obj][ax].append(val)

        lines.append("| Object Class | MAE X (m) | MAE Y (m) | MAE Z (m) | Total Error (m) |")
        lines.append("|---|---|---|---|---|")
        for obj in sorted(obj_errs):
            d = obj_errs[obj]
            mae_x = float(np.nanmean(d["x"])) if d["x"] else float("nan")
            mae_y = float(np.nanmean(d["y"])) if d["y"] else float("nan")
            mae_z = float(np.nanmean(d["z"])) if d["z"] else float("nan")
            tot = float(np.nanmean(d["total"])) if d["total"] else float("nan")
            lines.append(f"| {obj} | {mae_x:.4f} | {mae_y:.4f} | {mae_z:.4f} | {tot:.4f} |")
    else:
        lines.append("_No data available._")

    lines.append("")
    if os.path.exists(os.path.join(results_dir, "3d_error_boxplots.png")):
        lines.append("![3D Error Boxplots](3d_error_boxplots.png)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ------------------------------------------------------------------ #
    # Experiment 3: Robustness
    # ------------------------------------------------------------------ #
    lines.append("## Experiment 3: Robustness Analysis")
    lines.append("")

    # 3a
    rows3a = _read_csv(os.path.join(results_dir, "experiment3a_orientation.csv"))
    if rows3a:
        all_rates = [_safe_float(r["detection_rate"]) for r in rows3a]
        worst_yaw_row = min(rows3a, key=lambda r: _safe_float(r["detection_rate"]))
        best_yaw_row = max(rows3a, key=lambda r: _safe_float(r["detection_rate"]))
        lines.append("### 3a — Orientation Robustness")
        lines.append(f"- Mean detection rate across all orientations: **{np.nanmean(all_rates)*100:.1f}%**")
        lines.append(f"- Best: yaw={best_yaw_row['yaw_deg']}° on {best_yaw_row['object_class']} "
                     f"({_safe_float(best_yaw_row['detection_rate'])*100:.1f}%)")
        lines.append(f"- Worst: yaw={worst_yaw_row['yaw_deg']}° on {worst_yaw_row['object_class']} "
                     f"({_safe_float(worst_yaw_row['detection_rate'])*100:.1f}%)")
    else:
        lines.append("### 3a — Orientation Robustness\n_No data available._")

    lines.append("")

    # 3b
    rows3b = _read_csv(os.path.join(results_dir, "experiment3b_distance.csv"))
    if rows3b:
        all_rates = [_safe_float(r["detection_rate"]) for r in rows3b]
        best_dist_row = max(rows3b, key=lambda r: _safe_float(r["detection_rate"]))
        lines.append("### 3b — Distance Robustness")
        lines.append(f"- Mean detection rate across all distances: **{np.nanmean(all_rates)*100:.1f}%**")
        lines.append(f"- Best distance: {best_dist_row['camera_distance_m']}m "
                     f"({_safe_float(best_dist_row['detection_rate'])*100:.1f}%)")
    else:
        lines.append("### 3b — Distance Robustness\n_No data available._")

    lines.append("")

    # 3c
    rows3c = _read_csv(os.path.join(results_dir, "experiment3c_clutter.csv"))
    if rows3c:
        all_rates = [_safe_float(r["detection_rate"]) for r in rows3c]
        lines.append("### 3c — Clutter Robustness (20 trials)")
        lines.append(f"- Mean detection rate in clutter: **{np.nanmean(all_rates)*100:.1f}%**")
    else:
        lines.append("### 3c — Clutter Robustness\n_No data available._")

    lines.append("")
    if os.path.exists(os.path.join(results_dir, "robustness_curves.png")):
        lines.append("![Robustness Curves](robustness_curves.png)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ------------------------------------------------------------------ #
    # Parameter Sensitivity
    # ------------------------------------------------------------------ #
    lines.append("## Parameter Sensitivity Analysis")
    lines.append("")
    rows_sens = _read_csv(os.path.join(results_dir, "parameter_sensitivity.csv"))
    if rows_sens:
        params: Dict[str, Dict] = {}
        for row in rows_sens:
            param = row["parameter_name"]
            val = row["parameter_value"]
            rate = _safe_float(row["detection_rate"])
            params.setdefault(param, {})
            params[param].setdefault(val, []).append(rate)

        lines.append("| Parameter | Best Value | Detection Rate |")
        lines.append("|---|---|---|")
        for param, vals in sorted(params.items()):
            avg_rates = {v: float(np.nanmean(r)) for v, r in vals.items()}
            best_val = max(avg_rates, key=avg_rates.get)
            lines.append(f"| {param} | {best_val} | {avg_rates[best_val]*100:.1f}% |")
    else:
        lines.append("_No data available._")

    lines.append("")
    if os.path.exists(os.path.join(results_dir, "parameter_sensitivity_plots.png")):
        lines.append("![Parameter Sensitivity](parameter_sensitivity_plots.png)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_Report generated automatically by `experiments/report_generator.py`._")

    report = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    logger.info("Report written to %s", output_path)
    return report


if __name__ == "__main__":
    generate_summary_report()
