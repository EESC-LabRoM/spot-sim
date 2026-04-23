"""
Camera configuration package for spot_isaacsim.

- rgbd_cameras: Isaac Sim Camera sensor configs for Spot's onboard cameras
- mounted_zed: ZED X USD asset mounting (physical camera on robot body)
"""

from .rgbd_cameras import (
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
from .mounted_zed import ZedConfig, add_zed_to_stage

__all__ = [
    # RGBD cameras
    "CameraConfig",
    "hand_camera_config",
    "frontleft_camera_config",
    "frontright_camera_config",
    "left_camera_config",
    "right_camera_config",
    "back_camera_config",
    "body_camera_configs",
    "all_rgb_camera_configs",
    # ZED
    "ZedConfig",
    "add_zed_to_stage",
]
