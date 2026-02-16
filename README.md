# Vision-Based Robotic Affordance Detection

A complete computer vision system for detecting geometric features (holes, surfaces, handles) in RGB-D images and mapping them to 3D world coordinates for robotic manipulation.

## 🎯 Features

- **Hole Detection**: Circular apertures using Hough Circle Transform
- **Surface Detection**: Planar regions via contour analysis
- **Handle Detection**: Elongated protrusions using Canny edges + morphology
- **3D Mapping**: Pixel → world coordinate transformation
- **Visualization**: Annotated overlays and detailed reports
- **Fully Tested**: 30 unit tests, 100% passing

## 🚀 Quick Start

### Python 3.13 (Current)

Works with all vision components (no PyBullet):

```bash
# Activate venv
source venv/bin/activate

# Run standalone demo
python3 demo_vision.py

# Run tests
python3 -m pytest tests/test_feature_detection.py tests/test_coordinate_transform.py -v
```

### Python 3.11 (With `.venv311`)

Tested and working (PyBullet compilation issues on macOS):

```bash
# Use the Python 3.11 environment
.venv311/bin/python3 demo_vision.py

# Run tests
.venv311/bin/python3 -m pytest tests/test_feature_detection.py tests/test_coordinate_transform.py -v
```

## 📦 Installation

### Core Dependencies (Working)
```bash
pip install numpy opencv-python matplotlib pytest
```

### Optional: PyBullet Simulation
⚠️ PyBullet has compilation issues on macOS with XCode 17. See [TEST_RESULTS.md](TEST_RESULTS.md) for details.

```bash
pip install pybullet  # May fail on newer macOS
```

## 📊 Test Results

**30/30 tests passing** ✅

See [TEST_RESULTS.md](TEST_RESULTS.md) for detailed test coverage.

## 📁 Project Structure

```
src/
  camera.py               - Virtual RGB-D camera
  feature_detector.py     - Hole/surface/handle detection algorithms
  coordinate_transform.py - 2D↔3D coordinate transformations
  visualization.py        - Annotated overlays and reports

simulation/
  environment.py          - PyBullet physics simulation (requires pybullet)

tests/
  test_feature_detection.py      - 16 tests ✅
  test_coordinate_transform.py   - 14 tests ✅
  test_simulation.py             - Requires pybullet
  test_integration.py            - Requires pybullet

demo_vision.py           - Standalone demo (no pybullet needed)
main.py                  - Full pipeline with simulation (requires pybullet)
```

## 🎨 Example Output

The demo generates:
- `reports/demo_rgb.png` - Input RGB image
- `reports/demo_depth.png` - Depth visualization
- `reports/demo_annotated.png` - Detection overlay
- `reports/demo_summary.png` - Combined view
- `reports/demo_report.txt` - Detailed 3D coordinates

## 🧪 Running Tests

```bash
# All vision tests (no PyBullet needed)
pytest tests/test_feature_detection.py tests/test_coordinate_transform.py -v

# Specific test class
pytest tests/test_feature_detection.py::TestHoleDetection -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 📖 Documentation

- [QUICKSTART.md](QUICKSTART.md) - Quick reference guide
- [TEST_RESULTS.md](TEST_RESULTS.md) - Detailed test results
- [documentation/proposal-v1.md](documentation/proposal-v1.md) - Original project proposal

## 🔧 Module APIs

### Feature Detector

```python
from src.feature_detector import FeatureDetector

detector = FeatureDetector()
features = detector.detect_all(rgb_image)

# Returns: {"holes": [..], "surfaces": [..], "handles": [..]}
```

### Coordinate Transformer

```python
from src.coordinate_transform import CoordinateTransformer

transformer = CoordinateTransformer(intrinsic_matrix, extrinsic_matrix)
world_coords = transformer.pixel_to_world(u, v, depth)

# Returns: np.array([x, y, z]) in meters
```

## 🐛 Known Issues

### PyBullet Compilation
PyBullet fails to compile on macOS with recent XCode/SDK versions due to header conflicts. The vision system works independently of PyBullet.

**Workarounds**:
- Use the standalone demo with synthetic data
- Use Docker with older build environment
- Use alternative physics engines (MuJoCo, IsaacGym)

See [TEST_RESULTS.md](TEST_RESULTS.md) for details.

## 📝 License

Research/Educational Project

## 🙏 Acknowledgments

Based on classical computer vision techniques:
- Hough Circle Transform (Duda & Hart, 1972)
- Canny Edge Detection (Canny, 1986)
- PyBullet Physics Simulation (Coumans)