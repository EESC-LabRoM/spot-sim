"""
Spot robot configuration package for Isaac Sim.

Organizes configurations for:
- Cameras (cameras/rgbd_cameras.py, cameras/mounted_zed.py)
- Robot (robot.py)
- Physics parameters (physics/)
"""

from .cameras import (
    hand_camera_config,
    frontleft_camera_config,
    frontright_camera_config,
    left_camera_config,
    right_camera_config,
    back_camera_config,
    body_camera_configs,
    all_rgb_camera_configs,
    ZedConfig,
    add_zed_to_stage,
)

from .cfg.constants import (
    SPOT_STANDING_ARM_JOINT_POSITION,
    SPOT_RESTING_ARM_JOINT_POSITION,
    SPOT_STANDING_JOINT_POSITIONS,
    SPOT_DEFAULT_JOINT_POSITIONS,
)

from .robot import (
    PROJECT_ROOT,
    RobotConfig,
    load_robot_from_urdf,
    load_robot_from_usd,
)

from .physics import apply_all_physics

__all__ = [
    # Cameras
    "hand_camera_config",
    "frontleft_camera_config",
    "frontright_camera_config",
    "left_camera_config",
    "right_camera_config",
    "back_camera_config",
    "body_camera_configs",
    "all_rgb_camera_configs",
    "ZedConfig",
    "add_zed_to_stage",
    # Robot constants
    "SPOT_STANDING_ARM_JOINT_POSITION",
    "SPOT_RESTING_ARM_JOINT_POSITION",
    "SPOT_STANDING_JOINT_POSITIONS",
    "SPOT_DEFAULT_JOINT_POSITIONS",
    # Robot config
    "PROJECT_ROOT",
    "RobotConfig",
    # Robot loading
    "load_robot_from_urdf",
    "load_robot_from_usd",
    # Physics
    "apply_all_physics",
]
