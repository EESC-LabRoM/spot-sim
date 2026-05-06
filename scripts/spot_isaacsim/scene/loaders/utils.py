"""
Math and USD-physics utilities shared across the loaders package.

Public API:
    rpy_deg_to_quat    — ZYX-extrinsic RPY (degrees) → quaternion (w, x, y, z)
    apply_physics_cfg  — apply a physics: YAML subtree to a prim

USD physics hierarchy enforced here:
    Xform (root) → RigidBodyAPI, MassAPI, PhysxRigidBodyAPI   (direct, not BFS)
    Mesh children → CollisionAPI, friction material             (BFS)
"""

import numpy as np
from typing import Tuple

from scripts.spot_isaacsim.utils.math import rpy_deg_to_quat  # re-exported for loader consumers


def apply_physics_cfg(prim, physics: dict) -> None:
    """Apply a physics config dict to a prim.

    ``prim`` must be the root Xform. Rigid-body properties (mass, gravity) are applied
    directly to ``prim`` — NOT via BFS — because RigidBodyAPI belongs on the Xform.
    Friction is applied via BFS to CollisionAPI mesh children.

    Expected shape (all sub-keys optional):
        {
            "enabled": true,
            "rigid_body": {"mass": float, "gravity": bool},
            "material":   {"static_friction": float, "dynamic_friction": float},
        }

    No-op when ``physics["enabled"]`` is False. Caller must call ``_strip_physics``
    before this when disabling physics on a prim that has baked-in schemas.
    """
    if not physics.get("enabled", True):
        return

    rb = physics.get("rigid_body", {})
    if "mass" in rb:
        _set_mass(prim, rb["mass"])
    if not rb.get("gravity", True):
        _set_gravity_disabled(prim)

    mat = physics.get("material", {})
    if mat:
        _set_friction(prim, mat.get("static_friction"), mat.get("dynamic_friction"))


# ---------------------------------------------------------------------------
# Private helpers — not part of the public API
# ---------------------------------------------------------------------------

_PHYSICS_SCHEMA_NAMES = frozenset({
    "PhysicsRigidBodyAPI",
    "PhysicsCollisionAPI",
    "PhysicsMassAPI",
    "PhysicsMeshCollisionAPI",
    "PhysxRigidBodyAPI",
    "PhysxCollisionAPI",
})


def _strip_physics(prim) -> None:
    """Remove ALL applied physics schemas from the entire prim subtree (BFS).

    Handles multi-mesh USDs where schemas may be applied at different tree levels.
    Schemas removed: PhysicsRigidBodyAPI, PhysicsCollisionAPI, PhysicsMassAPI,
    PhysicsMeshCollisionAPI, PhysxRigidBodyAPI, PhysxCollisionAPI.

    Uses GetAppliedSchemas() + RemoveAppliedSchema() — avoids the UsdPhysics Python
    binding truthiness bug where API wrapper objects are always truthy.
    """
    from pxr import Usd

    removed = 0
    for p in Usd.PrimRange(prim):
        for schema in list(p.GetAppliedSchemas()):
            if schema in _PHYSICS_SCHEMA_NAMES:
                p.RemoveAppliedSchema(schema)
                removed += 1
    print(f"[Scene] Stripped {removed} physics schema(s) from {prim.GetPath()}")


def _set_mass(prim, mass_kg: float) -> None:
    """Apply MassAPI directly on prim (the Xform root) and set mass value.

    Applied directly — NOT via BFS — because MassAPI belongs on the rigid body
    Xform, not on a Mesh child.
    """
    from pxr import UsdPhysics

    mass_api = UsdPhysics.MassAPI.Get(prim.GetStage(), prim.GetPath())
    if not mass_api:
        mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.GetMassAttr().Set(mass_kg)
    print(f"[Scene] Mass set to {mass_kg} kg on {prim.GetPath()}")


def _set_gravity_disabled(prim) -> None:
    """Apply PhysxRigidBodyAPI directly on prim (the Xform root) and disable gravity.

    Applied directly — NOT via BFS — because PhysxRigidBodyAPI belongs on the Xform.
    """
    from pxr import PhysxSchema

    physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)
    if not physx_api:
        physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_api.GetDisableGravityAttr().Set(True)
    print(f"[Scene] Gravity disabled on {prim.GetPath()}")


def _set_friction(prim, static_friction: float | None, dynamic_friction: float | None) -> None:
    """Apply friction via BFS to every CollisionAPI prim in the subtree.

    BFS is correct here — friction/material belongs on the Mesh children
    (CollisionAPI prims), not on the Xform root.
    """
    from pxr import Usd, UsdPhysics

    if static_friction is None and dynamic_friction is None:
        return

    applied = 0
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            mat = UsdPhysics.MaterialAPI.Apply(p)
            if static_friction is not None:
                mat.CreateStaticFrictionAttr().Set(static_friction)
            if dynamic_friction is not None:
                mat.CreateDynamicFrictionAttr().Set(dynamic_friction)
            applied += 1
    print(f"[Scene] Friction (st={static_friction}, dy={dynamic_friction}) "
          f"applied to {applied} collision prim(s) under {prim.GetPath()}")
