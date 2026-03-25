"""
Simulation Environment Module
=============================
PyBullet-based simulation with a tabletop workspace and test objects
for robotic affordance detection.

Object Classes:
  - Class A (Washer/Nut): Cylindrical with central hole for insertion tasks
  - Class B (Box/Container): Rectangular with flat surfaces for picking tasks
  - Class C (Mug/Tool): Complex geometry with handle for grasping tasks
"""

import pybullet as p
import pybullet_data
import numpy as np
import time

from src.logger import setup_logger

logger = setup_logger(__name__)


class SimulationEnvironment:
    """Physics-based simulation environment with tabletop workspace and test objects."""

    def __init__(self, gui=True):
        """
        Initialize the simulation environment.

        Args:
            gui: If True, launch with visual GUI; otherwise use DIRECT mode.
        """
        self.gui = gui
        self.physics_client = None
        self.objects = {}
        self.table_id = None

    def setup(self):
        """Create the physics world, table, and spawn test objects."""
        mode = p.GUI if self.gui else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        logger.info("Simulation environment initializing (gui=%s)", self.gui)

        # Ground plane
        p.loadURDF("plane.urdf")

        # Table (a flat box)
        self.table_id = self._create_table()

        # Spawn test objects on the table
        self._spawn_objects()

        # Step simulation to let objects settle
        for _ in range(240):
            p.stepSimulation()

        logger.info("Simulation setup complete. Objects: %s", list(self.objects.keys()))
        return self

    def _create_table(self):
        """Create a tabletop surface at z=0.4m."""
        table_half_extents = [0.4, 0.6, 0.02]
        table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents)
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=table_half_extents,
            rgbaColor=[0.6, 0.4, 0.2, 1.0],
        )
        table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0, 0, 0.4],
        )
        return table_id

    def _spawn_objects(self):
        """Spawn the three object classes on the table."""
        table_z = 0.42  # just above the table surface

        # --- Class A: Washer (torus-like, approximated as a short cylinder with hole) ---
        washer_outer_r = 0.03
        washer_inner_r = 0.012
        washer_height = 0.006
        self.objects["washer"] = self._create_washer(
            position=[0.0, -0.15, table_z + washer_height],
            outer_radius=washer_outer_r,
            inner_radius=washer_inner_r,
            height=washer_height,
        )

        # --- Class B: Box/Container ---
        box_half = [0.05, 0.035, 0.03]
        box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half)
        box_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=box_half, rgbaColor=[0.2, 0.5, 0.8, 1.0]
        )
        self.objects["box"] = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=box_collision,
            baseVisualShapeIndex=box_visual,
            basePosition=[0.0, 0.0, table_z + box_half[2]],
        )

        # --- Class C: Mug (cylinder body + small box handle) ---
        self.objects["mug"] = self._create_mug(
            position=[0.0, 0.15, table_z + 0.04]
        )

    def _create_washer(self, position, outer_radius, inner_radius, height, orientation=None):
        """
        Create a washer-like object (solid cylinder — hole is detected visually).
        PyBullet doesn't support torus shapes natively, so we approximate with a cylinder
        and rely on the vision system to detect the central feature.
        """
        if orientation is None:
            orientation = p.getQuaternionFromEuler([0, 0, 0])
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=outer_radius, height=height)
        vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=outer_radius,
            length=height,
            rgbaColor=[0.7, 0.7, 0.7, 1.0],
        )
        body = p.createMultiBody(
            baseMass=0.02,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=position,
            baseOrientation=orientation,
        )
        # Store metadata for ground-truth validation
        self.objects.setdefault("_metadata", {})
        self.objects["_metadata"]["washer"] = {
            "hole_center": position,
            "hole_radius": inner_radius,
            "outer_radius": outer_radius,
        }
        return body

    def _create_mug(self, position, orientation=None):
        """
        Create a mug approximated as a cylinder body + a box handle.
        Returns the body ID of the main cylinder.
        """
        if orientation is None:
            orientation = p.getQuaternionFromEuler([0, 0, 0])
        mug_radius = 0.03
        mug_height = 0.07

        # Mug body
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=mug_radius, height=mug_height)
        vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=mug_radius,
            length=mug_height,
            rgbaColor=[0.9, 0.2, 0.2, 1.0],
        )
        body = p.createMultiBody(
            baseMass=0.15,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=position,
            baseOrientation=orientation,
        )

        # Handle (small box attached to the side)
        handle_half = [0.005, 0.015, 0.02]
        handle_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=handle_half)
        handle_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=handle_half,
            rgbaColor=[0.9, 0.2, 0.2, 1.0],
        )
        handle_pos = [
            position[0] + mug_radius + handle_half[0],
            position[1],
            position[2],
        ]
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=handle_col,
            baseVisualShapeIndex=handle_vis,
            basePosition=handle_pos,
        )

        return body

    def spawn_randomized(self, seed: int = None) -> None:
        """Spawn objects at random positions on the table.

        Args:
            seed: Optional integer seed for reproducible randomization.
        """
        if seed is not None:
            np.random.seed(seed)

        table_z = 0.42

        # Remove existing objects
        for name, obj_id in list(self.objects.items()):
            if name.startswith("_"):
                continue
            p.removeBody(obj_id)
        self.objects.clear()

        # Generate non-overlapping positions
        positions = []
        min_sep = 0.1
        max_attempts = 200

        for _ in range(3):
            for attempt in range(max_attempts):
                x = np.random.uniform(-0.25, 0.25)
                y = np.random.uniform(-0.35, 0.35)
                candidate = np.array([x, y])

                # Check separation from already-placed objects
                too_close = False
                for prev in positions:
                    if np.linalg.norm(candidate - prev) < min_sep:
                        too_close = True
                        break

                if not too_close:
                    positions.append(candidate)
                    break
            else:
                # Fallback: place without overlap guarantee
                positions.append(np.array([
                    np.random.uniform(-0.25, 0.25),
                    np.random.uniform(-0.35, 0.35),
                ]))

        yaws = [np.random.uniform(0, 2 * np.pi) for _ in range(3)]

        # --- Washer ---
        washer_height = 0.006
        washer_outer_r = 0.03
        washer_inner_r = 0.012
        wx, wy = positions[0]
        w_orn = p.getQuaternionFromEuler([0, 0, yaws[0]])
        self.objects["washer"] = self._create_washer(
            position=[wx, wy, table_z + washer_height],
            outer_radius=washer_outer_r,
            inner_radius=washer_inner_r,
            height=washer_height,
            orientation=w_orn,
        )

        # --- Box ---
        box_half = [0.05, 0.035, 0.03]
        bx, by = positions[1]
        b_orn = p.getQuaternionFromEuler([0, 0, yaws[1]])
        box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half)
        box_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=box_half, rgbaColor=[0.2, 0.5, 0.8, 1.0]
        )
        self.objects["box"] = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=box_collision,
            baseVisualShapeIndex=box_visual,
            basePosition=[bx, by, table_z + box_half[2]],
            baseOrientation=b_orn,
        )

        # --- Mug ---
        mx, my = positions[2]
        m_orn = p.getQuaternionFromEuler([0, 0, yaws[2]])
        self.objects["mug"] = self._create_mug(
            position=[mx, my, table_z + 0.04],
            orientation=m_orn,
        )

        # Settle
        for _ in range(500):
            p.stepSimulation()

        logger.info("spawn_randomized(seed=%s): washer=(%.3f,%.3f) box=(%.3f,%.3f) mug=(%.3f,%.3f)",
                    seed, positions[0][0], positions[0][1],
                    positions[1][0], positions[1][1],
                    positions[2][0], positions[2][1])

    def get_ground_truth(self) -> dict:
        """Return ground-truth feature locations for all spawned objects.

        Returns:
            Dict keyed by object name, each containing feature type, world
            position, and geometry dimensions derived from current physics state.
        """
        gt = {}

        # --- Washer ---
        if "washer" in self.objects:
            pos, orn = p.getBasePositionAndOrientation(self.objects["washer"])
            meta = self.objects.get("_metadata", {}).get("washer", {})
            inner_r = meta.get("hole_radius", 0.012)
            gt["washer"] = {
                "type": "hole",
                "position": list(pos),
                "feature_radius": inner_r,
            }
            logger.debug("GT washer hole at %s r=%.4f", pos, inner_r)

        # --- Box ---
        if "box" in self.objects:
            pos, orn = p.getBasePositionAndOrientation(self.objects["box"])
            box_half_z = 0.03
            surface_pos = [pos[0], pos[1], pos[2] + box_half_z]
            gt["box"] = {
                "type": "surface",
                "position": surface_pos,
                "surface_dimensions": [0.10, 0.07],
            }
            logger.debug("GT box surface at %s", surface_pos)

        # --- Mug ---
        if "mug" in self.objects:
            pos, orn = p.getBasePositionAndOrientation(self.objects["mug"])
            mug_radius = 0.03
            handle_half_x = 0.005
            handle_pos = [pos[0] + mug_radius + handle_half_x, pos[1], pos[2]]
            gt["mug"] = {
                "type": "handle",
                "position": handle_pos,
                "handle_dimensions": [0.01, 0.03, 0.04],
            }
            logger.debug("GT mug handle at %s", handle_pos)

        return gt

    def get_object_positions(self):
        """Return current positions of all objects."""
        positions = {}
        for name, obj_id in self.objects.items():
            if name.startswith("_"):
                continue
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            positions[name] = {"position": list(pos), "orientation": list(orn)}
        return positions

    def step(self, n=1):
        """Advance the simulation by n steps."""
        for _ in range(n):
            p.stepSimulation()

    def close(self):
        """Disconnect from the physics server."""
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None

    def reset_objects(self):
        """Remove all objects and re-spawn them."""
        for name, obj_id in list(self.objects.items()):
            if name.startswith("_"):
                continue
            p.removeBody(obj_id)
        self.objects.clear()
        self._spawn_objects()
        for _ in range(240):
            p.stepSimulation()


if __name__ == "__main__":
    env = SimulationEnvironment(gui=True)
    env.setup()
    logger.info("Object positions: %s", env.get_object_positions())
    logger.info("Simulation running — press Ctrl+C to exit.")
    try:
        while True:
            env.step()
            time.sleep(1 / 240.0)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
