import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils.math import quat_apply, quat_conjugate
from dataset_generation import ASSET_DIR

#  .usd paths
BOTTLE_USD_PATH = f"{ASSET_DIR}/converted_usd/bottle.usd"
DRILL_USD_PATH = f"{ASSET_DIR}/converted_usd/drill.usd"

# Configuration
# NOTE: fbx/.glb/.step must be converted to .usd first!
BOTTLE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/MyAsset",
    spawn=UsdFileCfg(
        usd_path=BOTTLE_USD_PATH,
        semantic_tags=[("TargetObject", "bottle")],
        activate_contact_sensors=True,
        scale=(1.0, 1.0, 1.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        # Rigid Body Properties
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        # Collision properties
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
    ),
    # Initial State
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(1.0, 0.0, 0.5),
        rot=(0.707, 0.707, 0.0, 0.0),
    ),
)


# NOTE: fbx/.glb/.step must be converted to .usd first!
DRILL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/MyAsset",
    spawn=UsdFileCfg(
        usd_path=DRILL_USD_PATH,
        semantic_tags=[("TargetObject", "drill")],
        activate_contact_sensors=True,
        scale=(0.01, 0.01, 0.01),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        # Rigid Body Properties
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        # Collision properties
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
    ),
    # Initial State
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(1.0, 0.0, 0.5),
        rot=(0.707, 0.707, 0.0, 0.0),
    ),
)
