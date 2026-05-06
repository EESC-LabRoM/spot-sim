"""
Drive gain parameters for all Spot joints.

Single source of truth for stiffness, damping, and armature values.
Applied once after world.reset() via apply_drive_gains().

Leg gains stored in N*m/deg (USD native unit) — written directly to DriveAPI.
Arm gains stored in N*m/rad — converted to N*m/deg (* pi/180) before writing.

Reference values from relic constants.py:
  Leg: HIP_STIFFNESS=60.0, HIP_DAMPING=1.5, KNEE_STIFFNESS=60.0, KNEE_DAMPING=1.5
  Arm: ARM_STIFFNESS=(120,120,120,100,100,100,16), ARM_DAMPING=(2,2,2,2,2,2,0.32)
  Arm effort limits: (90.9, 181.8, 90.9, 30.3, 30.3, 30.3, 15.32)
"""

import math
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Leg drive gains — (stiffness, damping) in N*m/deg (USD DriveAPI native unit)
# Active values: base 60.0, 1.50
# ---------------------------------------------------------------------------
LEG_DRIVE_PARAMS: Dict[str, Tuple[float, float]] = {
    "fl_hx": (90.0, 1.80),  "fl_hy": (100.0, 1.80),  "fl_kn": (100.0, 1.80),
    "fr_hx": (90.0, 1.80),  "fr_hy": (100.0, 1.80),  "fr_kn": (100.0, 1.80),
    "hl_hx": (90.0, 1.80),  "hl_hy": (90.0 , 1.80),  "hl_kn": (90.0 , 1.80),
    "hr_hx": (90.0, 1.80),  "hr_hy": (90.0 , 1.80),  "hr_kn": (90.0 , 1.80),
}

# ---------------------------------------------------------------------------
# Arm drive gains — (stiffness N*m/rad, damping N*m*s/rad, armature kg*m^2)
# Converted to N*m/deg (* pi/180) before writing to USD.
# ---------------------------------------------------------------------------
ARM_DRIVE_PARAMS: Dict[str, Tuple[float, float]] = {
    "arm_sh0": (690.0, 280.0),
    "arm_sh1": (690.0, 280.0),
    "arm_el0": (690.0, 280.0),
    "arm_el1": (573.0, 200.0),
    "arm_wr0": (573.0 ,200.0),
    "arm_wr1": (573.0 ,200.0),
    "arm_f1x": (200.0 ,23.50),
}

# ARM_DRIVE_PARAMS: Dict[str, Tuple[float, float]] = {
#     "arm_sh0": (220.0, 80.0),
#     "arm_sh1": (220.0, 80.0),
#     "arm_el0": (220.0, 80.0),
#     "arm_wr0": (220.0, 80.0),
#     "arm_wr1": (220.0, 80.0),
#     "arm_el1": (220.0, 80.0),
#     "arm_f1x": (16.00 ,0.32),
# }

def apply_drive_gains(robot_prim_path: str) -> None:
    """Apply drive stiffness/damping to all leg and arm joints.

    Call after world.reset() so physics drives are initialised.

    Args:
        robot_prim_path: USD root prim of the robot (e.g. '/World/Robot').
    """
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(robot_prim_path)

    all_joint_names = set(LEG_DRIVE_PARAMS) | set(ARM_DRIVE_PARAMS)
    joint_prim_map = {}
    for prim in Usd.PrimRange(robot_prim):
        if prim.GetName() in all_joint_names:
            joint_prim_map[prim.GetName()] = prim

    rad_to_usd = math.pi / 180.0  # multiply N*m/rad values by this for N*m/deg

    # --- Leg joints (values already in N*m/deg) ---
    leg_applied = 0
    for joint_name, (stiffness, damping) in LEG_DRIVE_PARAMS.items():
        prim = joint_prim_map.get(joint_name)
        if prim is None:
            print(f"[WARN] apply_drive_gains: leg joint '{joint_name}' not found under {robot_prim_path}")
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            print(f"[WARN] apply_drive_gains: no DriveAPI on {prim.GetPath()}")
            continue
        drive.GetTypeAttr().Set("force")
        drive.GetStiffnessAttr().Set(float(stiffness * rad_to_usd))
        drive.GetDampingAttr().Set(float(damping * rad_to_usd))
        leg_applied += 1

    # --- Arm joints (convert N*m/rad -> N*m/deg) ---
    arm_applied = 0
    for joint_name, (stiffness, damping) in ARM_DRIVE_PARAMS.items():
        prim = joint_prim_map.get(joint_name)
        if prim is None:
            print(f"[WARN] apply_drive_gains: arm joint '{joint_name}' not found under {robot_prim_path}")
            continue
        # Apply() is idempotent: creates DriveAPI if absent, returns existing one otherwise.
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.GetTypeAttr().Set("force")
        drive.GetStiffnessAttr().Set(float(stiffness * rad_to_usd))
        drive.GetDampingAttr().Set(float(damping * rad_to_usd))
        arm_applied += 1

    print(f"[Physics] Drive gains applied: {leg_applied}/{len(LEG_DRIVE_PARAMS)} leg, "
          f"{arm_applied}/{len(ARM_DRIVE_PARAMS)} arm joints")


def apply_leg_drive_multiplier(robot_prim_path: str, multiplier: float, silent: bool = False) -> None:
    """Scale LEG drive stiffness and damping by multiplier relative to LEG_DRIVE_PARAMS.

    Call at lock/unlock transitions — does NOT touch arm joints.

    Args:
        robot_prim_path: USD root prim of the robot (e.g. '/World/Robot').
        multiplier: Scale factor applied to both stiffness and damping (e.g. 10.0 to lock).
        silent: If True, suppress the status print (use during per-frame ramp ticks).
    """
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(robot_prim_path)

    joint_prim_map = {}
    for prim in Usd.PrimRange(robot_prim):
        if prim.GetName() in LEG_DRIVE_PARAMS:
            joint_prim_map[prim.GetName()] = prim

    rad_to_usd = math.pi / 180.0
    applied = 0
    for joint_name, (stiffness, damping) in LEG_DRIVE_PARAMS.items():
        prim = joint_prim_map.get(joint_name)
        if prim is None:
            print(f"[WARN] apply_leg_drive_multiplier: joint '{joint_name}' not found under {robot_prim_path}")
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            print(f"[WARN] apply_leg_drive_multiplier: no DriveAPI on {prim.GetPath()}")
            continue
        drive.GetStiffnessAttr().Set(float(stiffness * multiplier * rad_to_usd))
        drive.GetDampingAttr().Set(float(damping * multiplier * rad_to_usd))
        applied += 1

    if not silent:
        print(f"[Physics] Leg drive gains \u00d7{multiplier:.1f}: {applied}/{len(LEG_DRIVE_PARAMS)} joints")

