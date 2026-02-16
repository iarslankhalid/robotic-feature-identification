"""
Coordinate Transformation Module
==================================
Converts 2D pixel coordinates + depth into 3D world coordinates
using camera intrinsic and extrinsic matrices.

Pipeline:
  1. Get (u, v) pixel coordinates of a detected feature
  2. Sample depth at that pixel
  3. Back-project to 3D camera coordinates using intrinsics
  4. Transform to world coordinates using extrinsics
"""

import numpy as np


class CoordinateTransformer:
    """2D pixel → 3D world coordinate mapper."""

    def __init__(self, intrinsic_matrix, extrinsic_matrix):
        """
        Args:
            intrinsic_matrix: 3x3 camera intrinsic matrix (K).
            extrinsic_matrix: 4x4 camera extrinsic matrix (world-to-camera).
        """
        self.K = np.array(intrinsic_matrix, dtype=np.float64)
        self.extrinsic = np.array(extrinsic_matrix, dtype=np.float64)

        # Inverse of intrinsic for back-projection
        self.K_inv = np.linalg.inv(self.K)

        # Inverse of extrinsic for camera-to-world
        self.extrinsic_inv = np.linalg.inv(self.extrinsic)

    def pixel_to_camera(self, u, v, depth):
        """
        Back-project a 2D pixel to 3D camera coordinates.

        Args:
            u: Pixel column (x).
            v: Pixel row (y).
            depth: True depth at this pixel in meters.

        Returns:
            np.ndarray of shape (3,) — point in camera frame [X_c, Y_c, Z_c].
        """
        pixel_homogeneous = np.array([u, v, 1.0], dtype=np.float64)
        # Normalized camera coordinates
        point_norm = self.K_inv @ pixel_homogeneous
        # Scale by depth
        point_camera = point_norm * depth
        return point_camera

    def camera_to_world(self, point_camera):
        """
        Convert a 3D point from camera frame to world frame.

        Args:
            point_camera: np.ndarray of shape (3,) in camera coordinates.

        Returns:
            np.ndarray of shape (3,) in world coordinates.
        """
        point_h = np.append(point_camera, 1.0)  # homogeneous
        point_world_h = self.extrinsic_inv @ point_h
        return point_world_h[:3]

    def pixel_to_world(self, u, v, depth):
        """
        Full pipeline: pixel (u, v) + depth → world (X, Y, Z).

        Args:
            u: Pixel column.
            v: Pixel row.
            depth: True depth in meters.

        Returns:
            np.ndarray of shape (3,) — world coordinates.
        """
        cam_point = self.pixel_to_camera(u, v, depth)
        world_point = self.camera_to_world(cam_point)
        return world_point

    def world_to_pixel(self, world_point):
        """
        Project a 3D world point back to 2D pixel coordinates.

        Args:
            world_point: np.ndarray of shape (3,) in world coordinates.

        Returns:
            (u, v, depth) — pixel coordinates and depth.
        """
        point_h = np.append(np.array(world_point, dtype=np.float64), 1.0)
        cam_h = self.extrinsic @ point_h
        cam_point = cam_h[:3]

        depth = cam_point[2]
        if abs(depth) < 1e-8:
            return None

        pixel_h = self.K @ cam_point
        u = pixel_h[0] / pixel_h[2]
        v = pixel_h[1] / pixel_h[2]
        return (u, v, depth)

    def batch_pixel_to_world(self, pixels, depth_image):
        """
        Convert multiple pixels to world coordinates.

        Args:
            pixels: List of (u, v) tuples.
            depth_image: Full depth image array.

        Returns:
            List of np.ndarray world coordinates, one per pixel.
        """
        results = []
        for u, v in pixels:
            # Clamp to image bounds
            v_clamped = int(np.clip(v, 0, depth_image.shape[0] - 1))
            u_clamped = int(np.clip(u, 0, depth_image.shape[1] - 1))
            depth = depth_image[v_clamped, u_clamped]
            world_pt = self.pixel_to_world(u_clamped, v_clamped, depth)
            results.append(world_pt)
        return results
