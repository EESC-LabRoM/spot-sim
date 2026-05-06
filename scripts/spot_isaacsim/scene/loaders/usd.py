"""
USD asset loaders for Isaac Sim scenes.

Functions:
    load_usd_asset      — load any USD/mesh asset with optional physics config
    load_scene_usd      — load a static scene/environment USD (no physics)
    load_warehouse_stage — load a named built-in warehouse stage
"""

from pathlib import Path
from typing import Tuple

import numpy as np

from .mesh import MESH_EXTENSIONS, convert_mesh_to_usd
from .utils import rpy_deg_to_quat, apply_physics_cfg, _strip_physics
from scripts.spot_isaacsim.utils.path import resolve_asset_path

# Mapping from named stage types to Isaac Sim nucleus asset paths
_NAMED_STAGES = {
    "warehouse_simple": "Isaac/Environments/Simple_Warehouse/warehouse.usd",
    "warehouse_full":   "Isaac/Environments/Full_Warehouse/full_warehouse.usd",
}


def load_usd_asset(
    world,
    prim_path: str,
    usd_path: str,
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    physics: dict | None = None,
):
    """Load a USD asset (or mesh file) into the scene.

    Routing by ``physics`` value:

    ``None``
        Stage/environment path — position + scale via Xformable API, returns raw Usd.Prim.
        Does not register with the world scene.

    ``{"enabled": False, …}``
        Free-pose object — strips ALL baked physics from the subtree, then wraps in
        SingleXFormPrim for pose control. Registered with the world scene.

    ``{"enabled": True, …}``  (default when key absent)
        Normal physics object — applies physics config, wraps in
        SingleRigidPrim(reset_xform_properties=True). Registered with the world scene.

    If ``usd_path`` has a mesh extension (.glb, .fbx, .obj, .stl, .dae) the file is
    auto-converted to USD on first call and the result is cached next to the source.

    Args:
        world: Isaac Sim World object.
        prim_path: USD prim path where the asset will be placed (e.g. /World/Drill).
        usd_path: Path to USD/mesh file. Supports omniverse:// URLs, absolute paths,
            and paths relative to the project root.
        position: Initial position (x, y, z).
        orientation_rpy: Rotation (roll, pitch, yaw) in degrees — ZYX extrinsic.
        scale: Scale factor (x, y, z).
        physics: Physics config dict or None. See module docstring for shape.

    Returns:
        Usd.Prim when physics=None, SingleXFormPrim when physics.enabled=False,
        SingleRigidPrim when physics.enabled=True.
    """
    import omni.usd
    from isaacsim.core.utils.stage import add_reference_to_stage

    # Auto-convert mesh files to USD
    if Path(usd_path).suffix.lower() in MESH_EXTENSIONS:
        usd_path = convert_mesh_to_usd(usd_path)

    usd_path_resolved = resolve_asset_path(usd_path)
    if not usd_path_resolved.startswith(("https://", "http://", "omniverse://")):
        if not Path(usd_path_resolved).exists():
            raise FileNotFoundError(f"[Scene] Asset not found: {usd_path_resolved}")
    print(f"[Scene] Loading '{prim_path.split('/')[-1]}' from: {usd_path_resolved}")

    add_reference_to_stage(usd_path=usd_path_resolved, prim_path=prim_path)

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    # --- Stage/environment path: no physics, no scene registration ---
    if physics is None:
        from pxr import Gf, UsdGeom
        if prim.IsValid():
            xformable = UsdGeom.Xformable(prim)
            if any(abs(p) > 1e-9 for p in position):
                xformable.AddTranslateOp(opSuffix="worldOffset").Set(Gf.Vec3d(*position))
            if any(abs(s - 1.0) > 1e-9 for s in scale):
                xformable.AddScaleOp(opSuffix="worldScale").Set(Gf.Vec3f(*scale))
        print(f"[Scene] Asset '{prim_path.split('/')[-1]}' added (scene stage) at {prim_path}")
        return prim

    physics_enabled = physics.get("enabled", True)

    if not physics_enabled:
        # Strip all baked physics from the entire subtree before making it pose-only
        _strip_physics(prim)
        from isaacsim.core.prims import SingleXFormPrim
        quat = rpy_deg_to_quat(orientation_rpy)
        xform_prim = SingleXFormPrim(
            prim_path=prim_path,
            name=prim_path.split("/")[-1],
            position=np.array(position),
            orientation=quat,
            scale=np.array(scale),
            reset_xform_properties=True,
        )
        world.scene.add(xform_prim)
        print(f"[Scene] Asset '{prim_path.split('/')[-1]}' added (free-pose) at {prim_path}")
        return xform_prim

    # Normal physics object
    apply_physics_cfg(prim, physics)

    from isaacsim.core.prims import SingleRigidPrim
    quat = rpy_deg_to_quat(orientation_rpy)
    rigid_prim = SingleRigidPrim(
        prim_path=prim_path,
        name=prim_path.split("/")[-1],
        position=np.array(position),
        orientation=quat,
        scale=np.array(scale),
        reset_xform_properties=True,
    )
    world.scene.add(rigid_prim)
    print(f"[Scene] Asset '{prim_path.split('/')[-1]}' added (physics) at {prim_path}")
    return rigid_prim


def load_scene_usd(
    world,
    prim_path: str,
    usd_path: str,
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """Load a static scene/environment USD — no physics, no scene registration.

    Use this for warehouse stages, rooms, and any environment asset that has no
    rigid body. Returns the raw Usd.Prim.

    Args:
        world: Isaac Sim World object (unused, kept for consistent call signature).
        prim_path: USD prim path (e.g. /World/Warehouse).
        usd_path: Path to the USD file.
        position: Initial position (x, y, z).
        scale: Scale factor (x, y, z).

    Returns:
        Raw Usd.Prim at prim_path.
    """
    return load_usd_asset(world, prim_path, usd_path, position=position, scale=scale)


def load_warehouse_stage(
    world,
    prim_path: str = "/World/Warehouse",
    stage_type: str = None,
    usd_path: str = None,
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """Load a pre-built stage environment (warehouse, room, etc.) into the scene.

    Resolves the USD path in this order:
      1. ``usd_path`` if provided — used as-is (supports omniverse://, absolute,
         and project-relative paths)
      2. ``stage_type`` key in _NAMED_STAGES — resolved via ``get_assets_root_path()``

    Args:
        world: Isaac Sim World object.
        prim_path: USD prim path (e.g. /World/Warehouse).
        stage_type: Named built-in type. One of: ``warehouse_simple``, ``warehouse_full``.
        usd_path: Direct path to a USD file (overrides stage_type).
        position: Initial position (x, y, z).
        scale: Scale factor (x, y, z).

    Returns:
        Raw Usd.Prim at prim_path.
    """
    if usd_path is None:
        from isaacsim.storage.native import get_assets_root_path

        assets_root = get_assets_root_path()
        rel = _NAMED_STAGES.get(stage_type)
        if rel is None:
            raise ValueError(
                f"[Scene] Unknown stage type '{stage_type}'. "
                f"Known types: {list(_NAMED_STAGES)}. "
                f"Or provide usd_path directly."
            )
        usd_path = f"{assets_root}/{rel}"

    print(f"[Scene] Loading stage '{stage_type or prim_path.split('/')[-1]}'")
    return load_scene_usd(world, prim_path, usd_path, position=position, scale=scale)
