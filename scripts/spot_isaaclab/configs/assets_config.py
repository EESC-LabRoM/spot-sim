"""
Environment assets configuration
Defines ground, lights, table, and objects for the simulation scene
"""
import torch
import numpy as np
from pathlib import Path
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_from_euler_xyz


def create_ground_config() -> AssetBaseCfg:
    """Ground plane with physics material"""
    return AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
                friction_combine_mode="average",
                restitution_combine_mode="min",
            )
        )
    )


def create_light_config() -> AssetBaseCfg:
    """Dome light configuration"""
    return AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=4000.0,
            color=(0.9, 0.9, 0.9)
        )
    )


def create_table_config() -> AssetBaseCfg:
    """Table asset"""
    return AssetBaseCfg(
        prim_path="/World/Objects/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 1.0, 0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.4),
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.2, 0.0, 0.2))
    )


def create_target_object_config() -> RigidObjectCfg:
    """Red cube target object - KINEMATIC for interactive UI manipulation"""
    return RigidObjectCfg(
        prim_path="/World/Objects/Target",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Enable kinematic mode for UI dragging
                disable_gravity=True,     # Prevent falling when released
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, -0.3, 0.85))
    )


def create_blue_cube_config() -> RigidObjectCfg:
    """Blue cube object (for extended scene)"""
    return RigidObjectCfg(
        prim_path="/World/Objects/BlueCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 0.3, 0.85))
    )


def create_multilamp_config() -> AssetBaseCfg:
    """Multilamp USDZ asset (for extended scene)"""
    return AssetBaseCfg(
        prim_path="/World/Objects/Multilamp",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path("assets/honeycomb/Multilamps_Config.usdz")),
            scale=(1.0/1000, 1.0/1000, 1.0/1000),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(1.5, 0.0, 0.85),
            rot=quat_from_euler_xyz(
                torch.tensor(0.0),
                torch.tensor(np.pi/2),
                torch.tensor(np.pi/2)
            ).tolist()
        )
    )


def create_multilamp_collider_config() -> RigidObjectCfg:
    """Collider proxy for multilamp"""
    return RigidObjectCfg(
        prim_path="/World/Objects/MultilampCollider",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.15, 0.3),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.8, 0.8),
                opacity=0.3
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, -0.6, 0.7))
    )
