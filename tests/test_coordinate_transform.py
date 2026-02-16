"""
Tests for the coordinate transformation module.
Validates the pixel ↔ world coordinate conversions using known
camera parameters and synthetic data.
"""

import pytest
import numpy as np

from src.coordinate_transform import CoordinateTransformer


@pytest.fixture
def identity_transformer():
    """Transformer with identity extrinsic (camera = world frame)."""
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0,   0,   1],
    ], dtype=np.float64)
    extrinsic = np.eye(4, dtype=np.float64)
    return CoordinateTransformer(K, extrinsic)


@pytest.fixture
def offset_transformer():
    """Transformer with camera offset from origin."""
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0,   0,   1],
    ], dtype=np.float64)
    # Camera at (0, 0, 1) looking down the -Z axis (identity rotation)
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[2, 3] = 1.0  # translate along Z
    return CoordinateTransformer(K, extrinsic)


class TestPixelToCamera:

    def test_center_pixel_projects_along_optical_axis(self, identity_transformer):
        """
        The image center (cx, cy) should project to (0, 0, depth) in camera frame.
        """
        depth = 2.0
        result = identity_transformer.pixel_to_camera(320, 240, depth)
        np.testing.assert_allclose(result, [0.0, 0.0, 2.0], atol=1e-6)

    def test_depth_scaling(self, identity_transformer):
        """Doubling depth should double the projected camera coordinates."""
        r1 = identity_transformer.pixel_to_camera(400, 300, 1.0)
        r2 = identity_transformer.pixel_to_camera(400, 300, 2.0)
        np.testing.assert_allclose(r2, r1 * 2.0, atol=1e-6)

    def test_off_center_pixel(self, identity_transformer):
        """An off-center pixel should have non-zero X and Y components."""
        result = identity_transformer.pixel_to_camera(420, 340, 1.0)
        assert result[0] > 0, "X should be positive for pixel right of center"
        assert result[1] > 0, "Y should be positive for pixel below center"


class TestCameraToWorld:

    def test_identity_extrinsic(self, identity_transformer):
        """With identity extrinsic, camera coords == world coords."""
        cam_pt = np.array([0.5, -0.3, 1.5])
        world_pt = identity_transformer.camera_to_world(cam_pt)
        np.testing.assert_allclose(world_pt, cam_pt, atol=1e-6)

    def test_translation_offset(self, offset_transformer):
        """Camera translated by (0,0,1) should shift world Z."""
        cam_pt = np.array([0.0, 0.0, 0.0])
        world_pt = offset_transformer.camera_to_world(cam_pt)
        # The inverse of translating camera by (0,0,1) means
        # camera origin is at (0, 0, -1) in world
        np.testing.assert_allclose(world_pt[2], -1.0, atol=1e-6)


class TestPixelToWorld:

    def test_roundtrip_consistency(self, identity_transformer):
        """pixel_to_world → world_to_pixel should recover original pixel."""
        u, v, depth = 350, 200, 1.5
        world_pt = identity_transformer.pixel_to_world(u, v, depth)
        recovered = identity_transformer.world_to_pixel(world_pt)
        assert recovered is not None
        np.testing.assert_allclose(recovered[0], u, atol=1.0)
        np.testing.assert_allclose(recovered[1], v, atol=1.0)
        np.testing.assert_allclose(recovered[2], depth, atol=0.01)

    def test_center_maps_correctly(self, identity_transformer):
        """Image center at depth 1.0 should map to (0, 0, 1) with identity."""
        world_pt = identity_transformer.pixel_to_world(320, 240, 1.0)
        np.testing.assert_allclose(world_pt, [0.0, 0.0, 1.0], atol=1e-6)


class TestWorldToPixel:

    def test_origin_projects_to_center(self, identity_transformer):
        """World origin at depth 1.0 should project to image center."""
        result = identity_transformer.world_to_pixel([0.0, 0.0, 1.0])
        assert result is not None
        np.testing.assert_allclose(result[0], 320, atol=1.0)
        np.testing.assert_allclose(result[1], 240, atol=1.0)

    def test_zero_depth_returns_none(self, identity_transformer):
        """A point at depth 0 should return None (division by zero guard)."""
        result = identity_transformer.world_to_pixel([0.0, 0.0, 0.0])
        assert result is None


class TestBatchConversion:

    def test_batch_matches_individual(self, identity_transformer):
        """Batch conversion should match calling pixel_to_world individually."""
        depth_image = np.ones((480, 640), dtype=np.float32) * 2.0
        pixels = [(100, 200), (320, 240), (500, 400)]

        batch_results = identity_transformer.batch_pixel_to_world(pixels, depth_image)
        for i, (u, v) in enumerate(pixels):
            individual = identity_transformer.pixel_to_world(u, v, 2.0)
            np.testing.assert_allclose(batch_results[i], individual, atol=1e-6)

    def test_batch_empty_list(self, identity_transformer):
        """Empty pixel list should return empty results."""
        depth_image = np.ones((480, 640), dtype=np.float32)
        results = identity_transformer.batch_pixel_to_world([], depth_image)
        assert results == []


class TestEdgeCases:

    def test_very_large_depth(self, identity_transformer):
        """Should handle large depth values without overflow."""
        result = identity_transformer.pixel_to_world(320, 240, 1000.0)
        assert np.isfinite(result).all()

    def test_very_small_depth(self, identity_transformer):
        """Should handle very small positive depth."""
        result = identity_transformer.pixel_to_world(320, 240, 0.001)
        assert np.isfinite(result).all()

    def test_intrinsic_matrix_invertible(self, identity_transformer):
        """K_inv should be the true inverse of K."""
        identity = identity_transformer.K @ identity_transformer.K_inv
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-10)
