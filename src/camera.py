"""
Virtual RGB-D Camera Module
============================
Provides a simulated camera that captures synchronized RGB and depth images
from the PyBullet simulation environment.
"""

import pybullet as p
import numpy as np


class VirtualCamera:
    """Virtual RGB-D camera using PyBullet's rendering engine."""

    def __init__(
        self,
        position=(0.0, 0.0, 0.9),
        target=(0.0, 0.0, 0.42),
        up_vector=(0, 1, 0),
        width=640,
        height=480,
        fov=60.0,
        near=0.01,
        far=5.0,
    ):
        """
        Configure the virtual camera.

        Args:
            position: Camera position in world coordinates [x, y, z].
            target: Point the camera looks at.
            up_vector: Camera up direction.
            width: Image width in pixels.
            height: Image height in pixels.
            fov: Vertical field of view in degrees.
            near: Near clipping plane distance.
            far: Far clipping plane distance.
        """
        self.position = list(position)
        self.target = list(target)
        self.up_vector = list(up_vector)
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far

        # Compute matrices
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.position,
            cameraTargetPosition=self.target,
            cameraUpVector=self.up_vector,
        )
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.near,
            farVal=self.far,
        )

        # Pre-compute intrinsic matrix from FOV
        self._compute_intrinsics()

    def _compute_intrinsics(self):
        """Compute the 3x3 camera intrinsic matrix from FOV and image size."""
        fov_rad = np.radians(self.fov)
        fy = self.height / (2.0 * np.tan(fov_rad / 2.0))
        fx = fy  # square pixels
        cx = self.width / 2.0
        cy = self.height / 2.0

        self.intrinsic_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1],
        ], dtype=np.float64)

    def get_extrinsic_matrix(self):
        """
        Return the 4x4 extrinsic matrix (world-to-camera transform).
        PyBullet's view matrix is column-major; we reshape accordingly.
        """
        view = np.array(self.view_matrix).reshape(4, 4, order="F")
        return view

    def capture(self):
        """
        Capture an RGB-D frame from the simulation.

        Returns:
            rgb_image: np.ndarray of shape (H, W, 3), uint8
            depth_image: np.ndarray of shape (H, W), float32, true depth in meters
        """
        _, _, rgba, depth_buffer, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_TINY_RENDERER,
        )

        # RGB (drop alpha)
        rgba = np.array(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)
        rgb_image = rgba[:, :, :3]

        # Convert depth buffer to true Euclidean depth in meters
        depth_buffer = np.array(depth_buffer, dtype=np.float32).reshape(
            self.height, self.width
        )
        depth_image = self._linearize_depth(depth_buffer)

        return rgb_image, depth_image

    def _linearize_depth(self, depth_buffer):
        """
        Convert PyBullet's non-linear depth buffer to true depth in meters.

        PyBullet depth buffer: d = far * near / (far - (far - near) * buffer_value)
        """
        depth = self.far * self.near / (
            self.far - (self.far - self.near) * depth_buffer
        )
        return depth

    def get_camera_params(self):
        """Return a dict of all camera parameters for downstream modules."""
        return {
            "position": self.position,
            "target": self.target,
            "width": self.width,
            "height": self.height,
            "fov": self.fov,
            "near": self.near,
            "far": self.far,
            "intrinsic_matrix": self.intrinsic_matrix,
            "extrinsic_matrix": self.get_extrinsic_matrix(),
            "view_matrix": np.array(self.view_matrix).reshape(4, 4, order="F"),
            "projection_matrix": np.array(self.projection_matrix).reshape(4, 4, order="F"),
        }
