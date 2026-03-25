# CLAUDE.md — Vision-Based Robotic Affordance Detection

## Project Overview

This is an MSc Computer Science final project implementing a **vision-based robotic affordance detection system** in a PyBullet physics simulation. The system detects three types of geometric features on objects (holes, surfaces, handles), maps them to 3D world coordinates, and visualizes approach vectors — all using classical computer vision (no deep learning).

**Repository**: `robotic-feature-identification-main/`

---

## Current State of the Codebase

### What exists and works

| Module | File | Status |
|---|---|---|
| Feature Detector | `src/feature_detector.py` | ✅ Working — Hough circles, contour analysis, Canny edges |
| Coordinate Transform | `src/coordinate_transform.py` | ✅ Working — pixel→camera→world and inverse |
| Visualization | `src/visualization.py` | ✅ Basic — draws overlays, saves reports |
| Virtual Camera | `src/camera.py` | ✅ Working — RGB-D capture, depth linearization, intrinsic/extrinsic matrices |
| Simulation Environment | `simulation/environment.py` | ✅ Code complete — PyBullet tabletop with washer, box, mug |
| Standalone Demo | `demo_vision.py` | ✅ Working — synthetic images, no PyBullet needed |
| Full Pipeline | `main.py` | ✅ Code complete — requires PyBullet |
| Unit Tests | `tests/test_feature_detection.py` | ✅ 16 tests passing |
| Unit Tests | `tests/test_coordinate_transform.py` | ✅ 14 tests passing |
| Simulation Tests | `tests/test_simulation.py` | ⚠️ Requires PyBullet |
| Integration Tests | `tests/test_integration.py` | ⚠️ Requires PyBullet |

### Known issue: PyBullet on macOS

PyBullet fails to compile on macOS with recent XCode/SDK due to a `fdopen` macro clash in zlib headers. **The workaround**: use `--no-gui` / `DIRECT` mode, or run on Linux/Docker. On Linux systems (like this container), PyBullet installs fine via `pip install pybullet`.

### What is MISSING and needs to be built

1. **Experiments runner** — no `experiments/` module exists yet
2. **Quantitative evaluation** — no accuracy metrics, no ground-truth comparison pipeline
3. **Logging system** — no structured logging anywhere (just print statements)
4. **Approach vector visualization** — `visualization.py` only draws circles/rectangles, no 3D approach arrows
5. **Multi-trial randomized testing** — environment only spawns objects at fixed positions
6. **Parameter sensitivity analysis** — no sweep over detection thresholds
7. **Results export** — no CSV/JSON output for metrics, no auto-generated charts
8. **Demo video recording** — no frame-by-frame capture to video file
9. **requirements.txt** — file is misspelled as `requriements.txt`

---

## Task List

Complete these tasks in order. Each task should be a working, testable increment.

### Task 0: Project Setup and Fixes

**0.1** Rename `requriements.txt` → `requirements.txt`. Update any references.

**0.2** Create a proper Python package structure. Add a root `__init__.py` if missing. Ensure all imports work when running from the project root via `python -m`.

**0.3** Install dependencies:
```bash
pip install pybullet opencv-python-headless numpy matplotlib pytest --break-system-packages
```
Note: use `opencv-python-headless` (not `opencv-python`) since we're running headless. If already installed, skip.

**0.4** Verify the existing test suite passes:
```bash
python -m pytest tests/test_feature_detection.py tests/test_coordinate_transform.py -v
```
All 30 tests must pass before proceeding.

**0.5** Verify PyBullet works in DIRECT (headless) mode:
```python
import pybullet as p
cid = p.connect(p.DIRECT)
p.disconnect()
```

**0.6** Run `main.py` in headless mode to verify the full pipeline end-to-end:
```bash
python main.py --no-gui --no-plots --no-report
```
If this errors, fix the issue before continuing.

---

### Task 1: Add Structured Logging

Replace all `print()` calls across the codebase with Python's `logging` module.

**1.1** Create `src/logger.py`:
```python
import logging
import os
from datetime import datetime

def setup_logger(name="affordance", log_dir="logs", level=logging.INFO):
    """Configure logger with file + console handlers."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, f"run_{timestamp}.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d — %(message)s"
    ))
    logger.addHandler(fh)
    
    return logger
```

**1.2** Add `from src.logger import setup_logger` to every module. Replace `print(...)` with `logger.info(...)`, `logger.debug(...)`, or `logger.warning(...)` as appropriate. Use `logger.debug()` for per-feature verbose output and `logger.info()` for pipeline stage messages.

**1.3** Ensure the `logs/` directory is created automatically and add `logs/` to `.gitignore`.

---

### Task 2: Add Randomized Object Placement

The current `simulation/environment.py` spawns objects at fixed positions. For experiments, we need randomized placement.

**2.1** Add a `spawn_randomized(self, seed=None)` method to `SimulationEnvironment`:
- Accept an optional `seed` for reproducibility (`np.random.seed(seed)`)
- Randomize XY position within the table bounds: X ∈ [-0.25, 0.25], Y ∈ [-0.35, 0.35]
- Randomize Z-axis rotation (yaw) from 0° to 360° using `p.getQuaternionFromEuler([0, 0, yaw])`
- Ensure objects don't overlap (minimum 0.1m separation between centres)
- Log the spawned positions

**2.2** Add a `get_ground_truth(self)` method that returns a dict with precise feature locations per object:
```python
{
    "washer": {
        "type": "hole",
        "position": [x, y, z],  # center of the hole in world frame
        "feature_radius": 0.012,  # inner hole radius in meters
    },
    "box": {
        "type": "surface",
        "position": [x, y, z],  # centroid of top face
        "surface_dimensions": [0.10, 0.07],  # width, height of top face
    },
    "mug": {
        "type": "handle",
        "position": [x, y, z],  # center of handle protrusion
        "handle_dimensions": [0.01, 0.03, 0.04],  # half-extents
    },
}
```
Compute these from the current object positions using `p.getBasePositionAndOrientation()` plus the known object geometry offsets.

---

### Task 3: Build the Experiments Module

Create `experiments/` package with the full evaluation framework.

**3.1** Create `experiments/__init__.py` (empty).

**3.2** Create `experiments/metrics.py` with these metric functions:

```python
def detection_rate(detections, ground_truths):
    """Percentage of ground truth features that were detected."""
    # A detection matches a GT if its pixel projection is within 30px
    # Returns: float in [0, 1]

def pixel_localization_error(detected_pixel, ground_truth_world, transformer):
    """Euclidean pixel distance between detection and GT projection."""
    # Project GT world point to pixel, compute L2 distance
    # Returns: float (pixels)

def position_error_3d(detected_world, ground_truth_world):
    """Euclidean distance in 3D between detected and GT world coords."""
    # Returns: float (meters)

def per_axis_mae(detected_world, ground_truth_world):
    """Mean absolute error per axis (X, Y, Z)."""
    # Returns: dict {"x": float, "y": float, "z": float} in meters

def circle_iou(detected_center, detected_radius, gt_center, gt_radius, image_shape):
    """IoU between two circles on the image plane."""
    # Render both circles as binary masks, compute intersection/union
    # Returns: float in [0, 1]

def processing_latency(func, *args, **kwargs):
    """Time a function call in milliseconds."""
    # Returns: (result, elapsed_ms)
```

**3.3** Create `experiments/runner.py` — the main experiment executor:

```python
class ExperimentRunner:
    def __init__(self, num_trials=50, seed_start=0, output_dir="results"):
        self.num_trials = num_trials
        self.seed_start = seed_start
        self.output_dir = output_dir
    
    def run_experiment_1_detection_accuracy(self):
        """
        Experiment 1: Feature Detection Accuracy
        - 50 trials per object class
        - Randomized object pose each trial
        - Metrics: detection rate, pixel error, IoU (for holes), F1 (for edges)
        - Output: per-class accuracy table as CSV
        """
    
    def run_experiment_2_coordinate_precision(self):
        """
        Experiment 2: 3D Coordinate Precision
        - Same 50 trials
        - Metrics: per-axis MAE (X, Y, Z), total Euclidean error
        - Compare detected 3D coords vs PyBullet ground truth
        - Output: error distributions as box plots (saved to results/)
        """
    
    def run_experiment_3_robustness(self):
        """
        Experiment 3: Robustness Testing
        - 3a: Varying orientation (0°-360° in 30° steps = 12 orientations × 3 objects)
        - 3b: Varying distance (camera at 0.3m-1.0m in 0.1m steps = 8 distances)
        - 3c: Multi-object clutter (2-3 objects simultaneously, 20 random layouts)
        - Output: detection rate vs perturbation variable plots
        """
    
    def run_parameter_sensitivity(self):
        """
        Sweep key algorithm parameters and measure detection rate:
        - Hough param2: [15, 20, 25, 30, 35, 40, 45]
        - Canny thresholds: [(30,100), (40,120), (50,150), (60,180), (70,200)]
        - Surface min_area: [200, 500, 1000, 2000, 3000]
        Output: parameter vs detection_rate line plots
        """
    
    def run_all(self):
        """Run all experiments sequentially."""
        self.run_experiment_1_detection_accuracy()
        self.run_experiment_2_coordinate_precision()
        self.run_experiment_3_robustness()
        self.run_parameter_sensitivity()
    
    def _single_trial(self, seed, camera_pos=None, camera_target=None):
        """
        Execute one complete trial:
        1. Create SimulationEnvironment(gui=False)
        2. Call spawn_randomized(seed=seed)
        3. Step simulation 500 times to settle
        4. Capture RGB-D from VirtualCamera
        5. Run FeatureDetector.detect_all()
        6. Run CoordinateTransformer on all detections
        7. Get ground truth from env.get_ground_truth()
        8. Compute all metrics
        9. Close environment
        10. Return results dict
        """
```

**3.4** Create `experiments/visualize_results.py` — generates all charts:

```python
def plot_detection_accuracy_table(csv_path, output_path):
    """Bar chart: detection rate per object class."""

def plot_3d_error_boxplots(csv_path, output_path):
    """Box plots: per-axis MAE distributions."""

def plot_robustness_curves(csv_path, output_path):
    """Line plots: detection rate vs orientation/distance/clutter."""

def plot_parameter_sensitivity(csv_path, output_path):
    """Line plots: detection rate vs parameter value for each algorithm."""

def generate_all_figures(results_dir="results"):
    """Generate all figures for the report."""
```

---

### Task 4: Improve Visualization with Approach Vectors

The current `src/visualization.py` only draws 2D overlays. Add 3D approach vector arrows.

**4.1** Add `draw_approach_vectors()` function to `src/visualization.py`:
- For **holes**: draw an arrow pointing downward into the hole (insertion direction)
- For **surfaces**: draw an arrow pointing perpendicular to the surface (suction approach)
- For **handles**: draw an arrow pointing laterally toward the handle (grasp approach)
- Each arrow should be drawn as a line + arrowhead on the 2D image, colored by feature type
- The arrow direction should be computed from the surface normal (for simulation objects, use known geometry; approach is always -Z for top-down camera)

**4.2** Update `draw_features_on_image()` to call `draw_approach_vectors()` and include them in the annotated output.

**4.3** Add a `create_summary_figure()` function that produces a publication-quality 2×2 matplotlib figure:
- Top-left: Raw RGB image
- Top-right: Depth map with colorbar
- Bottom-left: Annotated detections with approach vectors
- Bottom-right: Detection confidence bar chart
- Save to `results/summary_figure.png` at 300 DPI

---

### Task 5: Demo Video Recording

**5.1** Create `experiments/video_recorder.py`:

```python
class VideoRecorder:
    def __init__(self, output_path="results/demo_video.mp4", fps=20, resolution=(640, 480)):
        # Use cv2.VideoWriter with XVID or mp4v codec
    
    def record_demo(self, num_frames=200):
        """
        Record a demo video showing:
        1. Objects spawning on the table
        2. Camera capturing RGB-D
        3. Feature detection running (show annotated frames)
        4. Objects being moved to new positions (respawn with different seed)
        5. Re-detection on new positions
        
        Each frame = annotated RGB from the pipeline.
        Move objects every 50 frames by calling env.reset_objects() 
        or spawn_randomized() with a new seed.
        """
    
    def add_frame(self, annotated_rgb):
        """Write one frame to the video."""
    
    def close(self):
        """Finalize and save the video file."""
```

---

### Task 6: Results Export and Reporting

**6.1** All experiment results must be saved as CSV files in `results/`:
- `results/experiment1_detection_accuracy.csv` — columns: `trial, seed, object_class, feature_type, detected, pixel_error, iou, confidence, latency_ms`
- `results/experiment2_coordinate_precision.csv` — columns: `trial, seed, object_class, detected_x, detected_y, detected_z, gt_x, gt_y, gt_z, error_x, error_y, error_z, error_total`
- `results/experiment3a_orientation.csv` — columns: `yaw_deg, object_class, detection_rate, mean_pixel_error, mean_3d_error`
- `results/experiment3b_distance.csv` — columns: `camera_distance_m, object_class, detection_rate, mean_pixel_error, mean_3d_error`
- `results/experiment3c_clutter.csv` — columns: `trial, num_objects, object_class, detection_rate`
- `results/parameter_sensitivity.csv` — columns: `parameter_name, parameter_value, detection_rate, object_class`

**6.2** Create `experiments/report_generator.py` that reads all CSVs and produces a summary:
```python
def generate_summary_report(results_dir="results", output_path="results/EXPERIMENT_REPORT.md"):
    """
    Auto-generate a markdown report with:
    - Table of detection rates per class (Exp 1)
    - Table of mean 3D errors per class and per axis (Exp 2)
    - Key findings from robustness analysis (Exp 3)
    - Best parameter configurations from sensitivity analysis
    - All figures embedded as image references
    """
```

---

### Task 7: Add Missing Tests

**7.1** Create `tests/test_experiments.py`:
- Test that `ExperimentRunner._single_trial()` completes without error on one trial
- Test that metric functions return correct types and ranges
- Test CSV output is well-formed
- Test that `detection_rate()` returns 1.0 when all features are found
- Test that `position_error_3d()` returns 0.0 for identical points

**7.2** Update `tests/test_simulation.py` and `tests/test_integration.py`:
- These already exist and are correct. Just ensure they pass now that PyBullet is available:
```bash
python -m pytest tests/ -v
```
All tests (existing 30 + new experiment tests) must pass.

**7.3** Create `tests/test_visualization.py`:
- Test that `draw_features_on_image()` returns an image of correct shape and dtype
- Test that `draw_approach_vectors()` doesn't crash on empty feature lists
- Test that `create_summary_figure()` produces a file on disk

---

### Task 8: Run Everything

**8.1** Execute the full experiment suite:
```bash
python -m experiments.runner
```
This should run all 50+ trials per experiment and save results to `results/`.

**8.2** Generate all visualizations:
```bash
python -m experiments.visualize_results
```

**8.3** Generate the summary report:
```bash
python -m experiments.report_generator
```

**8.4** Record the demo video:
```bash
python -m experiments.video_recorder
```

**8.5** Run the complete test suite:
```bash
python -m pytest tests/ -v --tb=short
```
Report the total pass/fail count.

---

## Implementation Constraints

### MUST follow these rules

1. **No deep learning** — only classical OpenCV algorithms (Hough, Canny, contour analysis, morphology). No PyTorch, TensorFlow, or pretrained models.
2. **PyBullet DIRECT mode only** — always use `gui=False`. No GUI windows. All visualization saved to files.
3. **Headless matplotlib** — use `matplotlib.use('Agg')` at the top of any file that imports matplotlib. Never call `plt.show()` — always `plt.savefig()` then `plt.close()`.
4. **All outputs to `results/`** — CSVs, PNGs, MP4, markdown report. Create the directory if it doesn't exist.
5. **All logs to `logs/`** — structured logging with timestamps.
6. **Reproducibility** — every randomized trial must use an explicit integer seed. Seeds should be `seed_start + trial_index`.
7. **No hardcoded absolute paths** — use `os.path.join()` and relative paths from project root.
8. **Type hints** on all new function signatures.
9. **Docstrings** on all new classes and functions (Google style).
10. **No network access needed** — everything runs locally with synthetic/simulated data.

### Performance expectations

- Each single trial (sim setup → capture → detect → transform → metrics → teardown) should take < 5 seconds
- Full 50-trial experiment should take < 5 minutes
- Video recording (200 frames) should take < 3 minutes

---

## File Structure After Completion

```
robotic-feature-identification-main/
├── main.py                          # Full pipeline entry point
├── demo_vision.py                   # Standalone demo (no PyBullet)
├── requirements.txt                 # Fixed filename
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── camera.py                    # Virtual RGB-D camera
│   ├── coordinate_transform.py      # 2D↔3D transforms
│   ├── feature_detector.py          # Hole/surface/handle detection
│   ├── visualization.py             # Drawing, approach vectors, summary figures
│   └── logger.py                    # Structured logging setup
│
├── simulation/
│   ├── __init__.py
│   └── environment.py               # PyBullet environment + randomized spawning + ground truth
│
├── experiments/
│   ├── __init__.py
│   ├── metrics.py                   # All evaluation metrics
│   ├── runner.py                    # Experiment executor (Exp 1, 2, 3 + sensitivity)
│   ├── visualize_results.py         # Chart generation from CSVs
│   ├── report_generator.py          # Auto-generate EXPERIMENT_REPORT.md
│   └── video_recorder.py            # Demo video capture
│
├── tests/
│   ├── __init__.py
│   ├── test_feature_detection.py    # 16 tests (existing)
│   ├── test_coordinate_transform.py # 14 tests (existing)
│   ├── test_simulation.py           # Simulation tests (existing, now runnable)
│   ├── test_integration.py          # Integration tests (existing, now runnable)
│   ├── test_experiments.py          # NEW: experiment/metric tests
│   └── test_visualization.py        # NEW: visualization tests
│
├── results/                         # All experiment outputs
│   ├── experiment1_detection_accuracy.csv
│   ├── experiment2_coordinate_precision.csv
│   ├── experiment3a_orientation.csv
│   ├── experiment3b_distance.csv
│   ├── experiment3c_clutter.csv
│   ├── parameter_sensitivity.csv
│   ├── summary_figure.png
│   ├── detection_accuracy_chart.png
│   ├── 3d_error_boxplots.png
│   ├── robustness_curves.png
│   ├── parameter_sensitivity_plots.png
│   ├── demo_video.mp4
│   └── EXPERIMENT_REPORT.md
│
├── logs/                            # Runtime logs
│   └── run_YYYYMMDD_HHMMSS.log
│
├── reports/                         # Demo outputs (existing)
│   ├── demo_rgb.png
│   ├── demo_depth.png
│   ├── demo_annotated.png
│   ├── demo_summary.png
│   └── demo_report.txt
│
└── documentation/
    ├── QUICKSTART.md
    ├── TEST_RESULTS.md
    └── proposal-v1.md
```

---

## Ground Truth Matching Logic

When comparing detections to ground truth, use this matching procedure:

1. For each ground truth feature, project its 3D world position to pixel coordinates using `transformer.world_to_pixel()`
2. For each detection, find the nearest ground truth projection within 40 pixels
3. A detection **matches** a ground truth if the pixel distance is < 40px
4. If a ground truth has no match, it is a **miss** (false negative)
5. If a detection has no matching ground truth, it is a **false positive**
6. Detection rate = matched_gt / total_gt

For **holes specifically**, also compute circle IoU between detected circle and projected GT circle.

---

## Camera Configuration for Experiments

Default camera (used in most trials):
```python
camera = VirtualCamera(
    position=(0.0, -0.5, 0.9),
    target=(0.0, 0.0, 0.42),
    up_vector=(0, 0, 1),
    width=640, height=480, fov=60.0
)
```

For **Experiment 3b** (varying distance), adjust the camera position:
```python
# distance_m ranges from 0.3 to 1.0
# Camera looks straight down at table center
position = (0.0, 0.0, 0.42 + distance_m)
target = (0.0, 0.0, 0.42)
up_vector = (0, 1, 0)
```

---

## How to Verify Success

After all tasks are complete, these commands should all succeed:

```bash
# 1. All tests pass
python -m pytest tests/ -v
# Expected: 50+ tests, 0 failures

# 2. Experiments produce results
ls results/*.csv
# Expected: 6 CSV files

# 3. Figures generated
ls results/*.png
# Expected: 4+ PNG chart files

# 4. Report exists
cat results/EXPERIMENT_REPORT.md
# Expected: Markdown with tables and figure references

# 5. Video exists
ls results/demo_video.mp4
# Expected: Playable MP4 file

# 6. Logs captured
ls logs/
# Expected: At least one .log file with structured output
```