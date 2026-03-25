"""
Tests for the experiments module — metrics, runner, and CSV output.
"""

import csv
import math
import os
import tempfile

import numpy as np
import pytest

from experiments.metrics import (
    MATCH_THRESHOLD_PX,
    circle_iou,
    detection_rate,
    per_axis_mae,
    pixel_localization_error,
    position_error_3d,
    processing_latency,
)


# ------------------------------------------------------------------ #
#  Metric unit tests
# ------------------------------------------------------------------ #

class TestDetectionRate:
    def test_all_matched(self):
        detections = [(100, 100), (200, 200)]
        ground_truths = [(105, 105), (198, 198)]  # within 40px
        assert detection_rate(detections, ground_truths) == pytest.approx(1.0)

    def test_none_matched(self):
        detections = [(0, 0)]
        ground_truths = [(400, 400)]  # far away
        assert detection_rate(detections, ground_truths) == pytest.approx(0.0)

    def test_empty_ground_truths(self):
        assert detection_rate([], []) == pytest.approx(1.0)

    def test_partial_match(self):
        detections = [(50, 50)]
        ground_truths = [(50, 50), (300, 300)]  # only first matches
        rate = detection_rate(detections, ground_truths)
        assert rate == pytest.approx(0.5)

    def test_returns_float(self):
        result = detection_rate([(10, 10)], [(10, 10)])
        assert isinstance(result, float)

    def test_threshold_exact(self):
        # Exactly at threshold should NOT match (strict <)
        detections = [(0, 0)]
        ground_truths = [(MATCH_THRESHOLD_PX, 0)]
        rate = detection_rate(detections, ground_truths)
        assert rate == pytest.approx(0.0)

    def test_threshold_just_inside(self):
        detections = [(0, 0)]
        ground_truths = [(MATCH_THRESHOLD_PX - 1, 0)]
        rate = detection_rate(detections, ground_truths)
        assert rate == pytest.approx(1.0)


class TestPositionError3D:
    def test_identical_points(self):
        assert position_error_3d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_known_distance(self):
        # 3-4-5 triangle in 3D
        result = position_error_3d([3.0, 4.0, 0.0], [0.0, 0.0, 0.0])
        assert result == pytest.approx(5.0)

    def test_returns_float(self):
        result = position_error_3d([0.1, 0.2, 0.3], [0.0, 0.0, 0.0])
        assert isinstance(result, float)

    def test_non_negative(self):
        result = position_error_3d([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0])
        assert result >= 0.0


class TestPerAxisMAE:
    def test_identical(self):
        result = per_axis_mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert result["x"] == pytest.approx(0.0)
        assert result["y"] == pytest.approx(0.0)
        assert result["z"] == pytest.approx(0.0)

    def test_known_values(self):
        result = per_axis_mae([1.0, 0.0, -1.0], [0.0, 0.0, 0.0])
        assert result["x"] == pytest.approx(1.0)
        assert result["y"] == pytest.approx(0.0)
        assert result["z"] == pytest.approx(1.0)

    def test_returns_dict_with_correct_keys(self):
        result = per_axis_mae([0, 0, 0], [0, 0, 0])
        assert set(result.keys()) == {"x", "y", "z"}

    def test_non_negative_values(self):
        result = per_axis_mae([-5.0, 3.0, -2.0], [0.0, 0.0, 0.0])
        assert result["x"] >= 0.0
        assert result["y"] >= 0.0
        assert result["z"] >= 0.0


class TestCircleIoU:
    def test_identical_circles(self):
        iou = circle_iou((100, 100), 30, (100, 100), 30, (480, 640))
        assert iou == pytest.approx(1.0, abs=0.01)

    def test_non_overlapping(self):
        iou = circle_iou((10, 10), 5, (400, 400), 5, (480, 640))
        assert iou == pytest.approx(0.0)

    def test_range(self):
        iou = circle_iou((100, 100), 40, (130, 100), 40, (480, 640))
        assert 0.0 <= iou <= 1.0

    def test_returns_float(self):
        result = circle_iou((50, 50), 20, (60, 60), 20, (480, 640))
        assert isinstance(result, float)


class TestProcessingLatency:
    def test_returns_tuple(self):
        result, ms = processing_latency(sum, [1, 2, 3])
        assert result == 6
        assert isinstance(ms, float)
        assert ms >= 0.0

    def test_latency_positive(self):
        import time
        _, ms = processing_latency(time.sleep, 0.01)
        assert ms >= 10.0  # at least 10 ms


# ------------------------------------------------------------------ #
#  Runner integration smoke test
# ------------------------------------------------------------------ #

class TestExperimentRunnerSingleTrial:
    @pytest.mark.slow
    def test_single_trial_completes(self):
        """Single trial should complete without errors and return metrics dict."""
        from experiments.runner import ExperimentRunner
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExperimentRunner(num_trials=1, seed_start=0, output_dir=tmp)
            result = runner._single_trial(seed=0)
        assert isinstance(result, dict)
        assert len(result) > 0
        for obj_name, metrics in result.items():
            assert "detected" in metrics
            assert "pixel_error" in metrics
            assert "latency_ms" in metrics
            assert metrics["latency_ms"] > 0


# ------------------------------------------------------------------ #
#  CSV output tests
# ------------------------------------------------------------------ #

class TestCSVOutput:
    @pytest.mark.slow
    def test_experiment1_csv_columns(self):
        from experiments.runner import ExperimentRunner
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExperimentRunner(num_trials=2, seed_start=0, output_dir=tmp)
            runner.run_experiment_1_detection_accuracy()
            csv_path = os.path.join(tmp, "experiment1_detection_accuracy.csv")
            assert os.path.exists(csv_path)
            rows = []
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) > 0
            required = {"trial", "seed", "object_class", "feature_type",
                        "detected", "pixel_error", "iou", "confidence", "latency_ms"}
            assert required.issubset(set(rows[0].keys()))

    @pytest.mark.slow
    def test_experiment2_csv_columns(self):
        from experiments.runner import ExperimentRunner
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExperimentRunner(num_trials=2, seed_start=0, output_dir=tmp)
            runner.run_experiment_2_coordinate_precision()
            csv_path = os.path.join(tmp, "experiment2_coordinate_precision.csv")
            assert os.path.exists(csv_path)
            rows = []
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) > 0
            required = {"trial", "seed", "object_class",
                        "detected_x", "detected_y", "detected_z",
                        "gt_x", "gt_y", "gt_z",
                        "error_x", "error_y", "error_z", "error_total"}
            assert required.issubset(set(rows[0].keys()))
