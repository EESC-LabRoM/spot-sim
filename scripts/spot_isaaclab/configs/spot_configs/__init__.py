"""Spot robot configuration module"""
from .robot_config import (
    create_spot_with_arm_config,
    arm_prefix,
    ARM_JOINT_NAMES,
    ARM_JOINT_LIMITS,
    LEG_JOINT_SUFFIXES,
)

__all__ = [
    "create_spot_with_arm_config",
    "create_hand_camera_config",
    "arm_prefix",
    "ARM_JOINT_NAMES",
    "ARM_JOINT_LIMITS",
    "LEG_JOINT_SUFFIXES",
]
