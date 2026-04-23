"""
Factory functions for creating scene objects in Isaac Sim.

Each function is a thin wrapper around Isaac Sim / USD APIs.
The builder.py module calls these with values extracted from scene_cfg.yaml.
"""

import math
from pathlib import Path
from typing import Tuple

import numpy as np


def _rpy_deg_to_quat(rpy_deg):
    """Convert RPY (roll, pitch, yaw) in degrees to quaternion (w, x, y, z).

    Uses ZYX extrinsic Euler convention (standard robotics / ROS convention):
    rotate by yaw around Z, then pitch around Y, then roll around X.
    """
    r, p, y = np.radians(rpy_deg)
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    w  = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([w, qx, qy, qz])


def create_ground_plane(
    world,
    prim_path: str = "/World/GroundPlane",
    size: float = 10.0,
    z_position: float = 0.0,
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    # static_friction: float = 1.0,
    # dynamic_friction: float = 1.0,
    # restitution: float = 0.0,
) -> None:
    """Create a static ground plane."""
    from isaacsim.core.api.objects import GroundPlane

    ground = GroundPlane(
        prim_path=prim_path,
        size=size,
        z_position=z_position,
        color=np.array(color),
        # static_friction=static_friction,
        # dynamic_friction=dynamic_friction,
        # restitution=restitution,
    )
    world.scene.add(ground)
    print(f"[Scene] Ground plane created at {prim_path}")


def create_dome_light(
    stage,
    prim_path: str = "/World/DomeLight",
    intensity: float = 4000.0,
    color: Tuple[float, float, float] = (0.9, 0.9, 0.9),
) -> None:
    """Create a dome light for ambient illumination."""
    from pxr import Gf, UsdLux

    light = UsdLux.DomeLight.Define(stage, prim_path)
    light.GetIntensityAttr().Set(intensity)
    light.GetColorAttr().Set(Gf.Vec3f(*color))
    print(f"[Scene] Dome light created at {prim_path}")


def create_static_cube(
    world,
    prim_path: str,
    size: Tuple[float, float, float],
    position: Tuple[float, float, float],
    color: Tuple[float, float, float],
    orientation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    static_friction: float = 0.7,
    dynamic_friction: float = 0.7,
):
    """
    Create a static (non-moving) cube.

    Args:
        size: Cube dimensions (x, y, z) applied as scale
        position: Position (x, y, z)
        color: RGB color
        orientation_rpy: Rotation (roll, pitch, yaw) in degrees — ZYX extrinsic convention
        static_friction: Static friction coefficient (default: 0.7)
        dynamic_friction: Dynamic friction coefficient (default: 0.7)

    Returns:
        FixedCuboid object
    """
    from isaacsim.core.api.objects import FixedCuboid

    cube = FixedCuboid(
        prim_path=prim_path,
        name=prim_path.split("/")[-1],
        size=1.0,
        scale=np.array(size),
        position=np.array(position),
        orientation=_rpy_deg_to_quat(orientation_rpy),
        color=np.array(color),
    )
    world.scene.add(cube)
    _set_friction(cube.prim, static_friction, dynamic_friction)
    print(f"[Scene] Static cube '{prim_path.split('/')[-1]}' created at {prim_path}")
    return cube


def create_dynamic_cube(
    world,
    prim_path: str,
    size: Tuple[float, float, float],
    position: Tuple[float, float, float],
    color: Tuple[float, float, float],
    mass: float = 0.1,
    orientation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """
    Create a dynamic (physics-enabled) cube.

    Args:
        size: Cube dimensions (x, y, z) applied as scale
        position: Position (x, y, z)
        color: RGB color
        mass: Object mass in kg
        orientation_rpy: Rotation (roll, pitch, yaw) in degrees — ZYX extrinsic convention

    Returns:
        DynamicCuboid object
    """
    from isaacsim.core.api.objects import DynamicCuboid

    cube = DynamicCuboid(
        prim_path=prim_path,
        name=prim_path.split("/")[-1],
        size=1.0,
        scale=np.array(size),
        position=np.array(position),
        orientation=_rpy_deg_to_quat(orientation_rpy),
        color=np.array(color),
        mass=mass,
    )
    world.scene.add(cube)
    print(f"[Scene] Dynamic cube '{prim_path.split('/')[-1]}' created at {prim_path}")
    return cube


def load_usd_asset(
    world,
    prim_path: str,
    usd_path: str,
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    gravity_enabled: bool = True,
    mass: float = None,
    static_friction: float = 1.0,
    dynamic_friction: float = 1.0,
):
    """
    Load a USD asset from the assets/ folder (or any path) into the scene.

    Does NOT use SingleXFormPrim — position/scale/rotation are applied via the
    USD Xformable API so that the asset's internal xform ops are preserved.
    This is critical for USD assets that contain physics rigid bodies.

    Supports URLs (https://, http://, omniverse://), absolute paths, and paths
    relative to the project root.

    Args:
        world: Isaac Sim World object (kept for API compatibility)
        prim_path: USD prim path where the asset will be placed (e.g. /World/BallValve)
        usd_path: Path to the USD file (URL, absolute path, or relative to project root)
        position: Initial position (x, y, z)
        orientation_rpy: Rotation (roll, pitch, yaw) in degrees — ZYX extrinsic convention
        scale: Scale factor (x, y, z)
        gravity_enabled: Set to False to disable gravity for all rigid bodies in this asset.
        mass: Override mass in kg for the first rigid body found in this asset.
              If None, the asset's baked-in mass is used unchanged.
        static_friction: Static friction coefficient applied to all collision prims in the asset (default: 0.7).
        dynamic_friction: Dynamic friction coefficient applied to all collision prims in the asset (default: 0.7).

    Returns:
        Usd.Prim at prim_path
    """
    import omni.usd
    from pxr import Gf, UsdGeom
    from isaacsim.core.utils.stage import add_reference_to_stage

    # Resolve path: URLs are used as-is; relative paths are resolved from project root
    if usd_path.startswith(("https://", "http://", "omniverse://")):
        usd_path_resolved = usd_path  # URL — use as-is
    else:
        asset_path = Path(usd_path)
        if asset_path.is_absolute():
            usd_path_resolved = str(asset_path)
        else:
            # 4 levels up: scene/ -> spot_isaacsim/ -> scripts/ -> project root
            project_root = Path(__file__).parent.parent.parent.parent
            usd_path_resolved = str((project_root / asset_path).resolve())

    print(f"[Scene] Loading asset '{prim_path.split('/')[-1]}' from: {usd_path_resolved}")

    add_reference_to_stage(usd_path=usd_path_resolved, prim_path=prim_path)

    # Set position/scale via USD API — does NOT clear existing xform ops
    # (important for physics assets that have their own internal transforms)
    stage_instance = omni.usd.get_context().get_stage()
    prim = stage_instance.GetPrimAtPath(prim_path)

    if prim.IsValid():
        xformable = UsdGeom.Xformable(prim)
        if any(abs(p) > 1e-9 for p in position):
            xformable.AddTranslateOp(opSuffix="worldOffset").Set(Gf.Vec3d(*position))
        if any(abs(r) > 1e-9 for r in orientation_rpy):
            xformable.AddRotateXYZOp(opSuffix="worldRot").Set(Gf.Vec3f(*orientation_rpy))
        if any(abs(s - 1.0) > 1e-9 for s in scale):
            xformable.AddScaleOp(opSuffix="worldScale").Set(Gf.Vec3f(*scale))

    if not gravity_enabled:
        _set_gravity_enabled(prim, enabled=False)

    if mass is not None:
        _set_mass(prim, mass)

    if static_friction is not None or dynamic_friction is not None:
        _set_friction(prim, static_friction or 0.5, dynamic_friction or 0.5)

    print(f"[Scene] Asset '{prim_path.split('/')[-1]}' added at {prim_path}")
    return prim


def _set_gravity_enabled(prim, enabled: bool):
    """BFS over prim subtree — sets disableGravity on every RigidBodyAPI prim."""
    from pxr import PhysxSchema, UsdPhysics

    queue = [prim]
    while queue:
        p = queue.pop(0)
        if UsdPhysics.RigidBodyAPI(p):
            physx_api = PhysxSchema.PhysxRigidBodyAPI(p)
            if not physx_api:
                physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
            physx_api.GetDisableGravityAttr().Set(not enabled)
        queue.extend(p.GetChildren())


def _set_mass(prim, mass_kg: float):
    """BFS over prim subtree — sets mass on the first RigidBodyAPI prim found."""
    from pxr import UsdPhysics

    queue = [prim]
    while queue:
        p = queue.pop(0)
        if UsdPhysics.RigidBodyAPI(p):
            mass_api = UsdPhysics.MassAPI.Get(p.GetStage(), p.GetPath())
            if not mass_api:
                mass_api = UsdPhysics.MassAPI.Apply(p)
            mass_api.GetMassAttr().Set(mass_kg)
            print(f"[Scene] Mass set to {mass_kg} kg on {p.GetPath()}")
            return
        queue.extend(p.GetChildren())
    print(f"[Scene] Warning: no RigidBodyAPI prim found under {prim.GetPath()}, mass not set.")


def _set_friction(prim, static_friction: float, dynamic_friction: float):
    """Apply friction in-place on every CollisionAPI prim in the subtree."""
    from pxr import Usd, UsdPhysics

    applied = 0
    for p in Usd.PrimRange(prim):
        if UsdPhysics.CollisionAPI(p):
            mat = UsdPhysics.MaterialAPI.Apply(p)
            mat.CreateStaticFrictionAttr().Set(static_friction)
            mat.CreateDynamicFrictionAttr().Set(dynamic_friction)
            applied += 1
    print(f"[Scene] Friction (st={static_friction}, dy={dynamic_friction}) "
          f"applied to {applied} collision prims under {prim.GetPath()}")


# Mapping from named stage types to Isaac Sim nucleus asset paths
_NAMED_STAGES = {
    "warehouse_simple": "Isaac/Environments/Simple_Warehouse/warehouse.usd",
    "warehouse_full": "Isaac/Environments/Full_Warehouse/full_warehouse.usd",
}


def load_warehouse_stage(
    world,
    prim_path: str = "/World/Warehouse",
    stage_type: str = None,
    usd_path: str = None,
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """
    Load a pre-built stage environment (warehouse, room, etc.) into the scene.

    Resolves the USD path in this order:
      1. ``usd_path`` if provided — used as-is (supports omniverse:// URLs, absolute paths,
         and paths relative to the project root)
      2. ``stage_type`` key in _NAMED_STAGES — resolved via ``get_assets_root_path()``

    Args:
        world: Isaac Sim World object
        prim_path: USD prim path where the stage will be placed (e.g. /World/Warehouse)
        stage_type: Named built-in type. One of: ``warehouse_simple``, ``warehouse_full``
        usd_path: Direct path to a USD file (overrides stage_type)
        position: Initial position (x, y, z)
        scale: Scale factor (x, y, z)

    Returns:
        XFormPrim wrapping the loaded stage
    """
    if usd_path is None:
        from isaacsim.storage.native import get_assets_root_path

        assets_root = get_assets_root_path()
        rel = _NAMED_STAGES.get(stage_type)
        if rel is None:
            raise ValueError(
                f"[Scene] Unknown stage type '{stage_type}'. "
                f"Known types: {list(_NAMED_STAGES)}. "
                f"Or provide a usd_path directly."
            )
        usd_path = f"{assets_root}/{rel}"

    print(f"[Scene] Loading stage '{stage_type or prim_path.split('/')[-1]}'")
    return load_usd_asset(
        world,
        prim_path=prim_path,
        usd_path=usd_path,
        position=position,
        scale=scale,
        static_friction=None,
        dynamic_friction=None,
    )

