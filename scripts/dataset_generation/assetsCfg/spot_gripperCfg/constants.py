# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""Constant data for Boston Dynamics Spot robot."""

SPOT_DEFAULT_POS: tuple[float, float, float] = (0.0, 0.0, 0.65)

ARM_JOINT_NAMES = [
    "arm_wr1",
    "arm_f1x",
]


SPOT_DEFAULT_JOINT_POS: dict[str, float] = {
    "arm_wr1": 0.0,
    "arm_f1x": -1.54,
}

SPOT_DEFAULT_JOINT_VEL: dict[str, float] = {
    "arm_wr1": 0.0,
    "arm_f1x": 0.0,
}

ARM_EFFORT_LIMIT: tuple[float, ...] = (30.3, 15.32)

ARM_STIFFNESS: tuple[float, ...] = (100.0, 16.0)

ARM_DAMPING: tuple[float, ...] = (2.0, 0.32)

ARM_ARMATURE: tuple[float, ...] = (0.01, 0.001)
