"""
Scene configuration for Spot simulation
Clean wrapper that combines robot, cameras, and assets
"""
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg

from .spot_configs.robot_config import create_spot_with_arm_config
from .spot_configs.cameras_config import (
    hand_camera_config,
    frontleft_camera_config,
    frontright_camera_config,
    left_camera_config,
    right_camera_config,
    back_camera_config,
)
from .assets_config import (
    create_ground_config,
    create_light_config,
    create_table_config,
    create_target_object_config,
    create_blue_cube_config,
    create_multilamp_config,
    create_multilamp_collider_config,
)


@configclass
class SpotManipulationSceneCfg(InteractiveSceneCfg):
    """Base scene for Spot manipulation"""

    # Environment
    ground = create_ground_config()
    light = create_light_config()

    # Robot with cameras
    robot = create_spot_with_arm_config()
    robot_camera_hand = hand_camera_config
    robot_camera_frontleft = frontleft_camera_config
    robot_camera_frontright = frontright_camera_config
    robot_camera_left = left_camera_config
    robot_camera_right = right_camera_config
    robot_camera_back = back_camera_config

    # Objects
    table = create_table_config()
    target_object = create_target_object_config()


@configclass
class SpotManipulationWithObjectsSceneCfg(SpotManipulationSceneCfg):
    """Extended scene with multiple objects"""

    blue_cube = create_blue_cube_config()
    multilamp = create_multilamp_config()
    multilamp_collider = create_multilamp_collider_config()
