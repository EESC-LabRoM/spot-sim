"""
Joint-position constants for Spot.

Shared across control/, spot_config/, and simulation layers.
"""

import torch

_c = {
    "shoulder": torch.pi / 180.0 * 110.0,
    "elbow":    torch.pi / 180.0 *  45.0,
}
_c["wrist"] = 180 - ((180 - _c["shoulder"]) + _c["elbow"])

SPOT_STANDING_ARM_JOINT_POSITION = {
    "arm_sh0": 0.0,
    "arm_sh1": -3.141 + _c["shoulder"],  # Shoulder 0/1
    "arm_el0":  3.141 - _c["elbow"],
    "arm_el1": 0.0,                       # Elbow 0/1
    "arm_wr0": 0.0 - _c["wrist"],
    "arm_wr1": 0.0,                       # Wrist 0/1
}

SPOT_RESTING_ARM_JOINT_POSITION = {
    "arm_sh0": 0.0,
    "arm_sh1": -3.141,  # Shoulder 0/1
    "arm_el0":  3.141,
    "arm_el1": 0.0,     # Elbow 0/1
    "arm_wr0": 0.0,
    "arm_wr1": 0.0,     # Wrist 0/1
}

SPOT_GRIPPER_CLOSED = {
    "arm_f1x": 0.0,
}

SPOT_GRIPPER_OPEN = {
    "arm_f1x": -1.5,
}

# Default standing pose — used without locomotion
SPOT_STANDING_JOINT_POSITIONS = {
    # Front left leg
    "fl_hx":  0.1,
    "fl_hy":  0.9,
    "fl_kn": -1.503,
    # Front right leg
    "fr_hx": -0.1,
    "fr_hy":  0.9,
    "fr_kn": -1.503,
    # Hind left leg
    "hl_hx":  0.1,
    "hl_hy":  1.1,
    "hl_kn": -1.503,
    # Hind right leg
    "hr_hx": -0.1,
    "hr_hy":  1.1,
    "hr_kn": -1.503,
}

SPOT_DEFAULT_JOINT_POSITIONS = SPOT_STANDING_JOINT_POSITIONS | SPOT_RESTING_ARM_JOINT_POSITION | SPOT_GRIPPER_OPEN

# ---------------------------------------------------------------------------
# Control constants — migrated from control/ files
# ---------------------------------------------------------------------------

import numpy as np

# Grasp state machine distances / thresholds
APPROACH_DISTANCE_M  = 0.15   # metres to back off from grasp for pre-grasp pose
RETRIEVE_LIFT_M      = 0.20   # metres to lift above grasp during retrieve
APPROACH_THRESH      = 0.06   # EE error (m) to consider pre-grasp reached
GRASPING_THRESH      = 0.04
RETRIEVING_THRESH    = 0.05
HOME_JOINT_THRESH    = 0.05   # joint error (rad) to consider arm homed

# Gripper positions
GRIPPER_OPEN   = -1.5
GRIPPER_CLOSED =  0.0

# State machine step counts
WAIT_STEPS  = 60
CLOSE_STEPS = 50

# Navigation
BASE_REACH_M          = 1.0    # metres — base must be within this to start grasp
WARMUP_STEPS          = 100    # physics steps before locomotion policy activates
NAV_CMD_ZERO_THRESH   = 0.05   # command magnitude considered zero
NAV_ARRIVED_STEPS     = 150    # steps at zero cmd before declaring arrived
NAV_IDLE_RESET_STEPS  = 150    # steps after arrived before resetting to IDLE

# Arm joints
ARM_JOINT_NAMES = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"]

# End-effector offset applied in the grasp frame (metres)
EE_TARGET_OFFSET = np.array([-0.15, 0.0, -0.02], dtype=np.float64)
