"""
Configuração da cena de simulação
"""
import torch
import numpy as np
from pathlib import Path

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_from_euler_xyz

from .robot_config import create_spot_with_arm_config,arm_prefix


@configclass
class SpotManipulationSceneCfg(InteractiveSceneCfg):
    """Configuração da cena para manipulação do Spot"""

    # Chão
    ground = AssetBaseCfg(
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

    # Iluminação
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=4000.0,
            color=(0.9, 0.9, 0.9)
        )
    )

    # Robô
    robot: ArticulationCfg = create_spot_with_arm_config()

    # Câmera no Gripper
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/"+arm_prefix+"_link_wr1/hand_cam",
        update_period=0,
        height=336,
        width=336,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.02),
            rot=quat_from_euler_xyz(
                torch.tensor(np.pi/2),
                torch.tensor(np.pi),
                torch.tensor(np.pi/2)
            ).tolist()
        )
    )

    # Mesa
    table = AssetBaseCfg(
        prim_path="/World/Objects/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 1.0, 0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.4),
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.2, 0.0, 0.2))
    )

    # Objeto alvo (cubo vermelho)
    target_object = RigidObjectCfg(
        prim_path="/World/Objects/Target",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, -0.3, 0.85))
    )


@configclass
class SpotManipulationWithObjectsSceneCfg(SpotManipulationSceneCfg):
    """Cena estendida com múltiplos objetos"""

    # Cubo azul
    blue_cube = RigidObjectCfg(
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

    # Multilamp (exemplo de objeto USDZ)
    multilamp = AssetBaseCfg(
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

    # Colisor proxy para multilamp
    multilamp_collider = RigidObjectCfg(
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
