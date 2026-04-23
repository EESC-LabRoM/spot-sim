"""
spot_isaacsim - Minimal Isaac Sim environment for Spot robot.

A clean, simplified environment for spawning the Boston Dynamics Spot robot
in Isaac Sim without Isaac Lab overhead.

Usage:
    # As a script
    python scripts/spot_isaacsim/play.py

    # As a module (after SimulationApp is initialized)
    from scripts.spot_isaacsim import StaticViewer, get_default_config
    viewer = StaticViewer()
    while True:
        viewer.step()
"""

from .spot_config import (
    # Robot config
    RobotConfig,
    SPOT_STANDING_JOINT_POSITIONS,
    SPOT_DEFAULT_JOINT_POSITIONS,
    PROJECT_ROOT,
    # Robot loading
    load_robot_from_urdf,
    load_robot_from_usd,
    # Physics
    apply_all_physics,
    # Camera configs
    CameraConfig,
    hand_camera_config,
    frontleft_camera_config,
    frontright_camera_config,
    left_camera_config,
    right_camera_config,
    back_camera_config,
    body_camera_configs,
    all_rgb_camera_configs,
)
from .scene import (
    SceneBuilder,
    SceneConfig,
    create_dome_light,
    create_dynamic_cube,
    create_ground_plane,
    create_static_cube,
    create_all_cameras,
    initialize_cameras,
)

__all__ = [
    # Robot/Sim Config
    "SceneConfig",
    "RobotConfig",
    "SPOT_STANDING_JOINT_POSITIONS",
    "PROJECT_ROOT",
    # Camera Config
    "CameraConfig",
    "hand_camera_config",
    "frontleft_camera_config",
    "frontright_camera_config",
    "left_camera_config",
    "right_camera_config",
    "back_camera_config",
    "body_camera_configs",
    "all_rgb_camera_configs",
    # Robot loading
    "load_robot_from_urdf",
    "load_robot_from_usd",
    # Physics
    "apply_all_physics",
    # Scene
    "SceneBuilder",
    "create_ground_plane",
    "create_dome_light",
    "create_static_cube",
    "create_dynamic_cube",
    # Cameras
    "create_all_cameras",
    "initialize_cameras",
]
