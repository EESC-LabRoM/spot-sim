"""
Mesh-to-USD conversion with correct physics prim hierarchy.

Public API:
    MESH_EXTENSIONS      — set of supported mesh file extensions
    convert_mesh_to_usd  — convert a mesh file to USD, cache result next to source

Physics applied after conversion (correct hierarchy):
    Xform (default prim) → RigidBodyAPI, MassAPI (density-based)
    Mesh child(ren)       → CollisionAPI, ConvexHull, identity local transform
"""

import asyncio
from pathlib import Path

MESH_EXTENSIONS = frozenset({".glb", ".fbx", ".obj", ".stl", ".dae"})


def convert_mesh_to_usd(
    input_path: str,
    output_path: str | None = None,
    density: float = 100.0,
    collision_offset: float = 0.001,
) -> str:
    """Convert a mesh file to a physics-ready USD and cache the result.

    On first call, runs ``omni.kit.asset_converter`` then post-processes the USD:
    - RigidBodyAPI + MassAPI (density-based) applied to the root Xform
    - CollisionAPI + ConvexHull applied to every Mesh child
    - xformOps cleared on Mesh children (identity local transform)

    On subsequent calls with the same path the cached ``.usd`` is returned immediately.

    Args:
        input_path: Path to source mesh (.glb, .fbx, .obj, .stl, .dae).
        output_path: Destination USD path. Defaults to same folder, same stem, .usd extension.
        density: Rigid body density in kg/m³ used to compute initial mass.
        collision_offset: Rest offset applied to CollisionAPI prims.

    Returns:
        Absolute path to the converted (or cached) USD file.
    """
    src = Path(input_path)
    dst = Path(output_path) if output_path else src.with_suffix(".usd")

    if dst.exists():
        print(f"[Scene] Using cached USD for '{src.name}': {dst}")
        return str(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    asyncio.get_event_loop().run_until_complete(_convert_async(src, dst))
    _post_process(dst, density=density, collision_offset=collision_offset)
    return str(dst)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

async def _convert_async(src: Path, dst: Path) -> None:
    """Run omni.kit.asset_converter — raw geometry only, no physics."""
    import omni.kit.asset_converter

    ctx = omni.kit.asset_converter.AssetConverterContext()
    ctx.ignore_materials = False
    ctx.merge_all_meshes = True
    ctx.use_meter_as_world_unit = True
    ctx.baking_scales = True
    ctx.use_double_precision_to_usd_transform_op = True

    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(str(src), str(dst), None, ctx)
    success = await task.wait_until_finished()
    if not success:
        raise RuntimeError(
            f"[Scene] Mesh conversion failed for '{src}': {task.get_error_message()}"
        )
    print(f"[Scene] Converted '{src.name}' → {dst}")


def _post_process(usd_path: Path, density: float, collision_offset: float) -> None:
    """Apply correct physics hierarchy and enforce identity transforms on Mesh children."""
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"[Scene] Could not open converted stage: {usd_path}")

    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise RuntimeError(f"[Scene] No default prim in converted USD: {usd_path}")

    # --- RigidBodyAPI + MassAPI on the root Xform ---
    UsdPhysics.RigidBodyAPI.Apply(root)

    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateDensityAttr().Set(density)

    # --- CollisionAPI + ConvexHull on each Mesh child, clear local xformOps ---
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        UsdPhysics.CollisionAPI.Apply(prim)

        mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_col.CreateApproximationAttr().Set("convexHull")

        col_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        col_api.CreateRestOffsetAttr().Set(collision_offset)

        _clear_xform_ops(prim)

    stage.GetRootLayer().Save()
    print(f"[Scene] Physics post-processing done: {usd_path}")


def _clear_xform_ops(prim) -> None:
    """Remove all xformOp attributes and clear xformOpOrder on a prim.

    Enforces identity local transform so the Mesh stays locked to its Xform parent.
    """
    from pxr import UsdGeom

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    for attr in list(prim.GetAttributes()):
        if attr.GetName().startswith("xformOp:"):
            prim.RemoveProperty(attr.GetName())
