# Project Proposal: Vision-Based Robotic Affordance Detection and Feature Identification

## 1. Introduction & Motivation

For robots to interact effectively with the physical world, they must possess more than simple object classification capabilities. While identifying an object as a "mug" or a "tool" is useful for inventory, it is insufficient for physical manipulation. To grasp, insert, or sort objects, a robot must identify specific affordances — functional geometric features such as handles, graspable rims, flat suction surfaces, or insertion holes.

This project proposes the development of a **Vision-Based Feature Identification System**. By utilizing a high-fidelity physics simulation to generate synthetic RGB-D (Red, Green, Blue + Depth) data, the system will bridge the gap between visual perception and robotic action. The system will autonomously detect meaningful geometric features on various rigid bodies and calculate their 3D spatial coordinates for potential robotic interaction.

---

## 2. Problem Statement

Traditional robotic manipulation relies heavily on hard-coded coordinates or known object poses. If an object is unstructured or randomly placed, standard “blind” robots fail. Current computer vision systems often stop at 2D bounding boxes (detecting where an object is in an image), failing to provide the specific grasp points required for interaction.

This project addresses the need for a perception system that can extract specific geometric features (holes, handles, surfaces) from a monocular camera feed and translate them into actionable 3D world coordinates relative to a robot’s base.

---

## 3. Project Objectives

1. **Simulation Environment**  
   Design a physics-based simulation environment containing a robotic workspace and a set of test objects (industrial parts and household items).

2. **Synthetic Data Acquisition**  
   Implement a virtual RGB-D camera system capable of streaming synchronized color and depth data.

3. **Feature Extraction**  
   Develop computer vision algorithms capable of identifying specific geometric features:
   - Apertures (holes) for insertion tasks
   - Planar surfaces for suction/picking tasks
   - Protrusions (handles) for grasping tasks

4. **3D Coordinate Mapping**  
   Implement mathematical transformation logic that converts 2D pixel coordinates from the vision system into 3D world coordinates.

---

## 4. Proposed Methodology

The project follows a four-stage pipeline, implemented entirely within a simulated environment to ensure safety, repeatability, and precise ground-truth validation.

### Phase 1: Environment Setup (Simulation Layer)

The simulation will be built using a physics engine widely used in robotics research.

- **Workspace:** Simulated tabletop environment
- **Objects:** Three object classes representing manipulation challenges:
  - Class A (Washer/Nut): insertion tasks requiring center-point detection
  - Class B (Box/Container): logistics tasks requiring surface segmentation
  - Class C (Mug/Tool): complex geometry requiring handle/edge detection

---

### Phase 2: Perception System (Data Layer)

A virtual camera will be calibrated and positioned within the simulation.

- RGB extraction for high-resolution color data
- Depth buffer processing to obtain true Euclidean distance values per pixel

---

### Phase 3: Algorithm Implementation (Processing Layer)

Core logic uses classical computer vision techniques to remain explainable and efficient.

- **Aperture Detection:** Hough Circle Transform for circular geometry and centroid detection
- **Surface Detection:** Contour analysis and area thresholding for planar surface identification
- **Handle/Edge Detection:** Canny edge detection + morphological operations for protrusion isolation

---

### Phase 4: Coordinate Transformation (Action Layer)

Pixel-to-world projection pipeline:

1. Identify the feature in the 2D image
2. Sample the depth at the detected pixel
3. Apply camera intrinsic matrix to project into 3D space
4. Apply camera extrinsic matrix to map into world coordinates

---

## 5. System Architecture

The system is composed of modular blocks:

1. **Input Module:** Synthetic camera stream
2. **Preprocessing Module:** Noise reduction, grayscale conversion, ROI masking
3. **Feature Logic Module:** Detection algorithms for holes, surfaces, and handles
4. **Math Module:** 2D → 3D matrix transformations
5. **Visualization Module:** Overlay drawing target vectors on the live feed

---

## 6. Tools and Technologies

- Programming Language: Python 3.13
- Simulation Engine: PyBullet
- Computer Vision Library: OpenCV (cv2)
- Numerical Computation: NumPy
- Visualization: Matplotlib / PyBullet Debug GUI

---

## 7. Expected Deliverables

1. Functional Python-based simulation script
2. Demonstration video showing real-time feature detection
3. Technical report including:
   - Algorithmic methodology
   - Coordinate transformation math
   - Accuracy analysis

---
