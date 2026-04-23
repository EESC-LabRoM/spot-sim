"""
Physics material parameters for Spot robot.

Feet friction and other contact material properties.
Applied once after world.reset() via apply_feet_friction() / apply_gripper_friction().
"""

FEET_STATIC_FRICTION: float = 1.0
FEET_DYNAMIC_FRICTION: float = 1.0
FEET_RESTITUTION: float = 0.0

FOOT_NAMES = {"fl_foot", "fr_foot", "hl_foot", "hr_foot"}

# Gripper finger friction — matches dataset_generation randomisation midpoint (0.3–1.0)
GRIPPER_STATIC_FRICTION: float = 2.5
GRIPPER_DYNAMIC_FRICTION: float = 2.5
GRIPPER_RESTITUTION: float = 0.0

GRIPPER_LINK_NAMES = {"arm_link_fngr", "arm0_link_fngr", "arm_link_jaw", "arm0_link_jaw"}


def apply_feet_friction(
    robot_prim_path: str,
    static_friction: float = FEET_STATIC_FRICTION,
    dynamic_friction: float = FEET_DYNAMIC_FRICTION,
    restitution: float = FEET_RESTITUTION,
) -> None:
    """Apply a high-friction physics material to Spot's feet collision meshes.

    Args:
        robot_prim_path: USD root prim of the robot (e.g. '/World/Robot').
        static_friction: Static friction coefficient.
        dynamic_friction: Dynamic friction coefficient.
        restitution: Restitution (bounciness) coefficient.
    """
    import omni.usd
    from pxr import Usd, UsdPhysics, UsdShade, UsdGeom

    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        print(f"[WARN] apply_feet_friction: Robot prim not found at {robot_prim_path}")
        return

    # Create Material Prim container
    material_scope_path = f"{robot_prim_path}/physics_materials"
    if not stage.GetPrimAtPath(material_scope_path).IsValid():
        UsdGeom.Scope.Define(stage, material_scope_path)

    # Define physics material
    mat_path = f"{material_scope_path}/high_friction"
    UsdShade.Material.Define(stage, mat_path)
    material_prim = stage.GetPrimAtPath(mat_path)

    physics_material = UsdPhysics.MaterialAPI.Apply(material_prim)
    physics_material.CreateStaticFrictionAttr().Set(static_friction)
    physics_material.CreateDynamicFrictionAttr().Set(dynamic_friction)
    physics_material.CreateRestitutionAttr().Set(restitution)

    # Bind to feet
    applied_count = 0
    material = UsdShade.Material(material_prim)

    for prim in Usd.PrimRange(robot_prim):
        if prim.GetName() in FOOT_NAMES:
            binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
            binding_api.Bind(material, "physics")
            binding_api.Bind(material, UsdShade.Tokens.allPurpose)
            applied_count += 1

    print(f"[Physics] Feet friction (st={static_friction}, dy={dynamic_friction}) "
          f"applied to {applied_count} collision bodies")


def apply_gripper_friction(
    robot_prim_path: str,
    static_friction: float = GRIPPER_STATIC_FRICTION,
    dynamic_friction: float = GRIPPER_DYNAMIC_FRICTION,
    restitution: float = GRIPPER_RESTITUTION,
) -> None:
    """Apply a friction physics material to Spot's gripper collision prims.

    Creates a shared material prim and binds it to the Xform collision prims
    (left_finger, left_tooth, right_finger, right_tooth, front_jaw, etc.) —
    same approach as feet friction.

    Args:
        robot_prim_path: USD root prim of the robot (e.g. '/World/Robot').
        static_friction: Static friction coefficient.
        dynamic_friction: Dynamic friction coefficient.
        restitution: Restitution (bounciness) coefficient.
    """
    import omni.usd
    from pxr import Usd, UsdPhysics, UsdShade, UsdGeom

    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        print(f"[WARN] apply_gripper_friction: Robot prim not found at {robot_prim_path}")
        return

    # Create material prim (reuse same scope as feet)
    material_scope_path = f"{robot_prim_path}/physics_materials"
    if not stage.GetPrimAtPath(material_scope_path).IsValid():
        UsdGeom.Scope.Define(stage, material_scope_path)

    mat_path = f"{material_scope_path}/gripper_friction"
    UsdShade.Material.Define(stage, mat_path)
    material_prim = stage.GetPrimAtPath(mat_path)

    physics_material = UsdPhysics.MaterialAPI.Apply(material_prim)
    physics_material.CreateStaticFrictionAttr().Set(static_friction)
    physics_material.CreateDynamicFrictionAttr().Set(dynamic_friction)
    physics_material.CreateRestitutionAttr().Set(restitution)

    material = UsdShade.Material(material_prim)

    # Bind to all Xform collision prims under gripper links
    applied_count = 0
    for prim in Usd.PrimRange(robot_prim):
        if prim.GetName() in GRIPPER_LINK_NAMES:
            for descendant in Usd.PrimRange(prim):
                if descendant == prim:
                    continue
                if UsdPhysics.CollisionAPI(descendant):
                    binding_api = UsdShade.MaterialBindingAPI.Apply(descendant)
                    binding_api.Bind(material, "physics")
                    applied_count += 1

    print(f"[Physics] Gripper friction (st={static_friction}, dy={dynamic_friction}) "
          f"applied to {applied_count} collision prims")

