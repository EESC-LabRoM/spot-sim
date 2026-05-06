"""
Camera factory and initialization for Spot robot in Isaac Sim.
"""

from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class CameraConfig:
    """RGB camera configuration for Isaac Sim."""

    name: str
    prim_path: str  # Relative to robot prim, e.g., "arm_link_wr1/hand_cam"
    resolution: Tuple[int, int]  # (width, height)
    frequency: int = 10  # Hz
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Local offset from parent
    orientation_rpy: Tuple[float, float, float] = (0.0, -np.pi/2, 0.0)  # Roll, Pitch, Yaw in radians
    focal_length: float = 18.0  # cm
    horizontal_aperture: float = 20.955  # cm
    clipping_range: Tuple[float, float] | None = (0.01, 100.00)  # near, far in meters


def create_all_cameras(
    robot_prim_path: str,
    camera_configs: List["CameraConfig"],
) -> Dict[str, "Camera"]:
    """
    Create all cameras from a list of configs.

    Cameras are not yet initialized — call initialize_cameras() after world.reset().

    Args:
        robot_prim_path: USD path to the robot prim (e.g., "/World/Robot")
        camera_configs: List of CameraConfig dataclasses

    Returns:
        Dictionary mapping camera names to Camera objects
    """
    from isaacsim.sensors.camera import Camera
    import isaacsim.core.utils.numpy.rotations as rot_utils

    cameras = {}
    for config in camera_configs:
        orientation = rot_utils.euler_angles_to_quats(
            np.array(config.orientation_rpy), degrees=False
        )
        camera = Camera(
            prim_path=f"{robot_prim_path}/{config.prim_path}",
            name=config.name,
            resolution=config.resolution,
            frequency=config.frequency,
            translation=np.array(config.translation),
            orientation=orientation,
        )
        camera.set_focal_length(config.focal_length)
        camera.set_horizontal_aperture(config.horizontal_aperture)
        if config.clipping_range is not None:
            camera.set_clipping_range(*config.clipping_range)
        cameras[config.name] = camera
    return cameras


def initialize_cameras(cameras: List["Camera"], enable_depth: bool = True) -> None:
    """
    Initialize cameras after world.reset().

    Args:
        cameras: List of Camera objects to initialize
        enable_depth: Whether to enable depth output for all cameras
    """
    for camera in cameras:
        camera.initialize()
        if enable_depth:
            camera.add_distance_to_image_plane_to_frame()
