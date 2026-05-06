"""
ZED X camera mounting configuration for Isaac Sim.

Adds the Stereolabs ZED X USD asset to the simulation stage,
parented to a configurable reference frame with a local transform.
"""

import math
from dataclasses import dataclass, field
from typing import Tuple

DEFAULT_ZED_USD_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usdc"
)


@dataclass
class ZedConfig:
    """Configuration for a ZED X camera mounted on the robot."""

    reference_frame: str = "/World/Robot/body"  # USD prim path to parent to
    translation: Tuple[float, float, float] = (0.38, 0.0, 0.09)  # Local XYZ offset (m)
    rotation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Roll, pitch, yaw (rad)
    usd_url: str = field(default=DEFAULT_ZED_USD_URL)
    prim_name: str = "ZED_X"  # Prim name under reference_frame
    streaming_port: int = 30000  # Port for ZED ROS2 Wrapper (sim_port=)
    transport_mode: str = "NETWORK"  # "IPC" (same machine), "NETWORK" (remote), "BOTH"
    ros_frame_id: str = (
        "zed_camera_link"  # ROS TF child frame name expected by the wrapper
    )


def add_zed_to_stage(config: ZedConfig) -> str:
    """
    Add the ZED X USD asset to the stage, parented to config.reference_frame.

    Applies translation and rotation_rpy as a local transform relative to the
    reference frame. Returns the full USD prim path of the ZED prim.
    """
    import omni.usd
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, Usd, UsdGeom

    prim_path = f"{config.reference_frame}/{config.prim_name}"

    add_reference_to_stage(usd_path=config.usd_url, prim_path=prim_path)

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(
            f"[ZED] Prim not found after add_reference_to_stage: {prim_path}"
        )

    # The ZED USD has physics APIs that conflict with the parent body link.
    # Disabling rigidBodyEnabled is not enough — PhysX still validates the xform
    # hierarchy for any prim with RigidBodyAPI applied, producing "missing xformstack
    # reset" errors. Remove all physics APIs from the ZED prim and every descendant
    # via USD list-edit ops on the current override layer (works for referenced prims).
    physics_apis = ("PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI")
    for p in Usd.PrimRange(prim):
        for api in physics_apis:
            p.RemoveAppliedSchema(api)

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()

    tx, ty, tz = config.translation
    xform.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))

    r, p, y = config.rotation_rpy
    xform.AddRotateXOp().Set(math.degrees(r))
    xform.AddRotateYOp().Set(math.degrees(p))
    xform.AddRotateZOp().Set(math.degrees(y))

    print(
        f"[ZED] Mounted ZED X at '{prim_path}' "
        f"(ref: '{config.reference_frame}', "
        f"t={config.translation}, rpy={config.rotation_rpy})"
    )
    return prim_path
