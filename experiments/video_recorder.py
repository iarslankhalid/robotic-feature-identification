"""
Demo Video Recorder Module
===========================
Records a demo video showing feature detection running on a PyBullet simulation,
with objects being repositioned every 50 frames.
"""

import os
from typing import Tuple

import cv2
import numpy as np

from simulation.environment import SimulationEnvironment
from src.camera import VirtualCamera
from src.coordinate_transform import CoordinateTransformer
from src.feature_detector import FeatureDetector
from src.logger import setup_logger
from src.visualization import draw_features_on_image

logger = setup_logger(__name__)

DEFAULT_CAM_POS = (0.0, -0.5, 0.9)
DEFAULT_CAM_TARGET = (0.0, 0.0, 0.42)
DEFAULT_CAM_UP = (0, 0, 1)


class VideoRecorder:
    """Records a demo video of the vision pipeline running on a simulation.

    Attributes:
        output_path: File path for the output MP4.
        fps: Frames per second.
        resolution: (width, height) of each frame.
        writer: cv2.VideoWriter instance.
    """

    def __init__(
        self,
        output_path: str = "results/demo_video.mp4",
        fps: int = 20,
        resolution: Tuple[int, int] = (640, 480),
    ) -> None:
        """Initialize the video recorder.

        Args:
            output_path: Destination path for the MP4 file.
            fps: Frames per second.
            resolution: (width, height) tuple for the video.
        """
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.writer = None
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    def _init_writer(self) -> None:
        """Lazily initialize the VideoWriter on first frame."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, self.resolution
        )
        if not self.writer.isOpened():
            logger.warning("VideoWriter failed to open — trying XVID codec")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.writer = cv2.VideoWriter(
                self.output_path.replace(".mp4", ".avi"),
                fourcc, self.fps, self.resolution,
            )
        logger.info("VideoWriter initialized: %s  fps=%d  res=%s",
                    self.output_path, self.fps, self.resolution)

    def add_frame(self, annotated_rgb: np.ndarray) -> None:
        """Write one annotated RGB frame to the video.

        Args:
            annotated_rgb: np.ndarray (H, W, 3) in RGB format.
        """
        if self.writer is None:
            self._init_writer()
        # Resize if needed
        h, w = annotated_rgb.shape[:2]
        if (w, h) != self.resolution:
            frame = cv2.resize(annotated_rgb, self.resolution)
        else:
            frame = annotated_rgb.copy()
        # Convert RGB → BGR for OpenCV
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(bgr)

    def close(self) -> None:
        """Finalize and save the video file."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            logger.info("Video saved to %s", self.output_path)

    def record_demo(self, num_frames: int = 200) -> None:
        """Record a demo video of the full pipeline.

        Shows objects spawning, camera capture, and feature detection.
        Objects are respawned every 50 frames with a new seed.

        Args:
            num_frames: Total number of annotated frames to capture.
        """
        logger.info("Recording demo video: %d frames → %s", num_frames, self.output_path)

        env = SimulationEnvironment(gui=False)
        try:
            env.setup()

            camera = VirtualCamera(
                position=DEFAULT_CAM_POS,
                target=DEFAULT_CAM_TARGET,
                up_vector=DEFAULT_CAM_UP,
                width=self.resolution[0],
                height=self.resolution[1],
                fov=60.0,
            )
            cam_params = camera.get_camera_params()
            transformer = CoordinateTransformer(
                intrinsic_matrix=cam_params["intrinsic_matrix"],
                extrinsic_matrix=cam_params["extrinsic_matrix"],
            )
            detector = FeatureDetector()

            current_seed = 0
            env.spawn_randomized(seed=current_seed)

            for frame_idx in range(num_frames):
                # Respawn every 50 frames
                if frame_idx > 0 and frame_idx % 50 == 0:
                    current_seed += 1
                    logger.info("Frame %d: respawning with seed=%d", frame_idx, current_seed)
                    env.spawn_randomized(seed=current_seed)

                # Advance simulation a few steps to animate settling
                env.step(n=5)

                # Capture and detect
                rgb_image, depth_image = camera.capture()
                features = detector.detect_all(rgb_image)

                world_coords = {}
                for feat_type, feat_list in features.items():
                    coords = []
                    for feat in feat_list:
                        u, v = feat.pixel_coords
                        vc = int(np.clip(v, 0, depth_image.shape[0] - 1))
                        uc = int(np.clip(u, 0, depth_image.shape[1] - 1))
                        coords.append(
                            transformer.pixel_to_world(uc, vc, depth_image[vc, uc])
                        )
                    world_coords[feat_type] = coords

                annotated = draw_features_on_image(rgb_image, features, world_coords)

                # Add frame counter overlay
                bgr_ann = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    bgr_ann,
                    f"Frame {frame_idx + 1}/{num_frames}  Seed {current_seed}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                frame_rgb = cv2.cvtColor(bgr_ann, cv2.COLOR_BGR2RGB)
                self.add_frame(frame_rgb)

            logger.info("Recording complete: %d frames", num_frames)
        finally:
            env.close()
            self.close()


if __name__ == "__main__":
    recorder = VideoRecorder(output_path="results/demo_video.mp4", fps=20, resolution=(640, 480))
    recorder.record_demo(num_frames=200)
