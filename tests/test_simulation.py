"""
Tests for the simulation environment.
Verifies that PyBullet initializes correctly, objects spawn at expected
positions, and the environment can be reset cleanly.
"""

import pytest
import numpy as np
import pybullet as p

from simulation.environment import SimulationEnvironment


@pytest.fixture
def env():
    """Create a headless simulation environment for testing."""
    sim = SimulationEnvironment(gui=False)
    sim.setup()
    yield sim
    sim.close()


class TestEnvironmentSetup:
    """Tests for basic environment initialization."""

    def test_physics_client_connected(self, env):
        """Physics client should be active after setup."""
        assert env.physics_client is not None

    def test_table_created(self, env):
        """Table body should exist."""
        assert env.table_id is not None
        pos, _ = p.getBasePositionAndOrientation(env.table_id)
        assert abs(pos[2] - 0.4) < 0.01, "Table should be at z=0.4"

    def test_all_objects_spawned(self, env):
        """All three object classes should be present."""
        positions = env.get_object_positions()
        assert "washer" in positions, "Washer (Class A) not found"
        assert "box" in positions, "Box (Class B) not found"
        assert "mug" in positions, "Mug (Class C) not found"

    def test_objects_above_table(self, env):
        """All objects should be resting on or above the table surface."""
        positions = env.get_object_positions()
        table_surface_z = 0.42  # table top + half thickness
        for name, info in positions.items():
            z = info["position"][2]
            assert z >= table_surface_z - 0.05, (
                f"{name} is below table at z={z:.3f}"
            )

    def test_object_count(self, env):
        """There should be exactly 3 visible objects (excluding metadata)."""
        positions = env.get_object_positions()
        assert len(positions) == 3


class TestEnvironmentDynamics:
    """Tests for simulation stepping and resets."""

    def test_step_does_not_crash(self, env):
        """Stepping the simulation should not raise errors."""
        env.step(n=100)

    def test_objects_stable_after_settling(self, env):
        """Objects should not move significantly after settling."""
        positions_before = env.get_object_positions()
        env.step(n=500)
        positions_after = env.get_object_positions()

        for name in positions_before:
            pos_a = np.array(positions_before[name]["position"])
            pos_b = np.array(positions_after[name]["position"])
            drift = np.linalg.norm(pos_b - pos_a)
            assert drift < 0.01, (
                f"{name} drifted by {drift:.4f}m after settling"
            )

    def test_reset_objects(self, env):
        """After reset, objects should be back near their spawn positions."""
        original = env.get_object_positions()
        env.reset_objects()
        reset_pos = env.get_object_positions()

        assert set(original.keys()) == set(reset_pos.keys())
        for name in original:
            pos_o = np.array(original[name]["position"])
            pos_r = np.array(reset_pos[name]["position"])
            diff = np.linalg.norm(pos_r - pos_o)
            assert diff < 0.05, (
                f"{name} position changed by {diff:.4f}m after reset"
            )

    def test_close_and_reconnect(self):
        """Environment should be closable and re-creatable."""
        env1 = SimulationEnvironment(gui=False)
        env1.setup()
        env1.close()
        assert env1.physics_client is None

        env2 = SimulationEnvironment(gui=False)
        env2.setup()
        positions = env2.get_object_positions()
        assert len(positions) == 3
        env2.close()
