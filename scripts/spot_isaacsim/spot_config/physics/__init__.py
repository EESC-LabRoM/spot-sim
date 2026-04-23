"""
Centralized physics configuration for Spot robot.

All drive gains, friction, and material properties are defined here.
Call apply_all_physics() once after world.reset() to configure the robot.
"""

from .drive_gains import LEG_DRIVE_PARAMS, ARM_DRIVE_PARAMS, apply_drive_gains, apply_leg_drive_multiplier
from .materials import (
    FEET_STATIC_FRICTION,
    FEET_DYNAMIC_FRICTION,
    FEET_RESTITUTION,
    apply_feet_friction,
    GRIPPER_STATIC_FRICTION,
    GRIPPER_DYNAMIC_FRICTION,
    GRIPPER_RESTITUTION,
    apply_gripper_friction,
)


def apply_all_physics(robot_prim_path: str) -> None:
    """Apply ALL physics parameters to the robot.

    Call once after world.reset(). This is the single entry point
    for configuring drive gains, friction, and other physics properties.

    Args:
        robot_prim_path: USD root prim of the robot (e.g. '/World/Robot').
    """
    apply_drive_gains(robot_prim_path)
    apply_feet_friction(robot_prim_path)
    apply_gripper_friction(robot_prim_path)


__all__ = [
    "LEG_DRIVE_PARAMS",
    "ARM_DRIVE_PARAMS",
    "FEET_STATIC_FRICTION",
    "FEET_DYNAMIC_FRICTION",
    "FEET_RESTITUTION",
    "apply_drive_gains",
    "apply_leg_drive_multiplier",
    "apply_feet_friction",
    "GRIPPER_STATIC_FRICTION",
    "GRIPPER_DYNAMIC_FRICTION",
    "GRIPPER_RESTITUTION",
    "apply_gripper_friction",
    "apply_all_physics",
]
