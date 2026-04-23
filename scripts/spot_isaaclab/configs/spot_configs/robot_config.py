"""
Robot configuration for Spot with arm
Defines robot articulation, actuators, and joint limits
"""
import torch
import numpy as np
from pathlib import Path
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from ..global_config import USE_URDF
arm_prefix = "arm" if USE_URDF else "arm0"

try:
    from relic.assets.spot.spot import SPOT_ARM_CFG
except ImportError:
    raise ImportError("Não foi possível carregar SPOT_ARM_CFG")

try:
    from isaaclab_assets.robots.spot import SPOT_CFG
except ImportError:
    raise ImportError("Não foi possível carregar SPOT_CFG")

relic_urdf_multiplier_s = 1000.0
relic_urdf_multiplier_d = 1000.0


def create_spot_with_arm_config():
    """
    Create Spot robot configuration with arm and integrated hand camera.
    Adapts joint names based on USE_URDF global variable.

    Returns:
        ArticulationCfg: Complete Spot configuration with arm and camera
    """

    if USE_URDF:
        spot_cfg = SPOT_ARM_CFG.copy()
        # Override with local URDF that has fisheye cameras
        # Navigate from this file to project root: configs/spot_configs/ -> configs/ -> spot_isaac/ -> scripts/ -> project_root/
        project_root = Path(__file__).parent.parent.parent.parent.parent
        local_urdf_path = str(project_root / "assets" / "spot" / "spot_with_arm.urdf")
        spot_cfg.spawn.asset_path = local_urdf_path
        spot_cfg.spawn.joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type='force',
            target_type='position',
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            )
        )
        print(f"[INFO] Configuração do Spot com braço criada a partir de {spot_cfg.spawn.asset_path}")
    else:
        spot_cfg = SPOT_CFG.copy()
        # USD with arm
        spot_cfg.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/BostonDynamics/spot/spot_with_arm.usd"
        print(f"[INFO] Configuração do Spot com braço criada a partir de {spot_cfg.spawn.usd_path}")

    spot_cfg.init_state.pos = (0.0, 0.0, 0.4)

    print(f"[INFO] Usando prefixo de braço: '{arm_prefix}_'")

    # Initial arm positions (retracted)
    spot_cfg.init_state.joint_pos.update({
        f"{arm_prefix}_sh0": 0.0,
        f"{arm_prefix}_sh1": -3.141,
        f"{arm_prefix}_el0": 2.8,
        f"{arm_prefix}_el1": 0.0,
        f"{arm_prefix}_wr0": -0.2,
        f"{arm_prefix}_wr1": 0.0,
        f"{arm_prefix}_f1x": 0.0,
    })

    spot_cfg.init_state.joint_vel = {
        f"{arm_prefix}_sh0": 0.0, f"{arm_prefix}_sh1": 0.0, f"{arm_prefix}_el0": 0.0,
        f"{arm_prefix}_el1": 0.0, f"{arm_prefix}_wr0": 0.0, f"{arm_prefix}_wr1": 0.0, f"{arm_prefix}_f1x": 0.0,
    }

    # --- ARM ACTUATORS ---
    # 1. Heavy joints (Base, Shoulder, Elbow-Joint)
    # Real limit ~90Nm
    spot_cfg.actuators["arm_heavy"] = ImplicitActuatorCfg(
        joint_names_expr=[f"{arm_prefix}_sh0", f"{arm_prefix}_sh1", f"{arm_prefix}_el0"],
        effort_limit=90.0,
        velocity_limit=10.0,
        stiffness=120.0 * relic_urdf_multiplier_s,
        damping=4.0 * relic_urdf_multiplier_d,
        armature=0.01,
        friction=0.5,
    )

    # 2. Light joints (Elbow-Rotation and Wrists)
    # Real limit ~23Nm
    spot_cfg.actuators["arm_light"] = ImplicitActuatorCfg(
        joint_names_expr=[f"{arm_prefix}_el1", f"{arm_prefix}_wr0", f"{arm_prefix}_wr1"],
        effort_limit=15.0,
        velocity_limit=15.0,
        stiffness=60.0 * relic_urdf_multiplier_s,
        damping=4.0 * relic_urdf_multiplier_d,
        armature=0.01,
        friction=0.3,
    )

    # 3. Gripper - Keep closed (not used in IK for now)
    spot_cfg.actuators["gripper"] = ImplicitActuatorCfg(
        joint_names_expr=[f"{arm_prefix}_f1x"],
        effort_limit=10.0,
        velocity_limit=5.0,
        stiffness=100.0 * relic_urdf_multiplier_s,
        damping=5.0 * relic_urdf_multiplier_d,
    )

    # --- LEG ACTUATORS ---
    # 3. Hips - Weaker than knees
    # Real: ~45Nm
    spot_cfg.actuators["legs_hips"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hx", ".*_hy"],
        effort_limit=60.0,
        velocity_limit=20.0,
        stiffness=120.0 * relic_urdf_multiplier_s,
        damping=2.0 * relic_urdf_multiplier_d,
    )

    # 4. Knees - Strongest leg motor
    # Real: Peak ~115Nm
    spot_cfg.actuators["legs_knees"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_kn"],
        effort_limit=140.0,
        velocity_limit=20.0,
        stiffness=150.0 * relic_urdf_multiplier_s,
        damping=2.0 * relic_urdf_multiplier_d,
    )

    spot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"

    return spot_cfg


# Constants
# Standard joint suffixes (independent of prefix)
_ARM_SUFFIXES = ['_sh0', '_sh1', '_el0', '_el1', '_wr0', '_wr1', '_f1x']

# Generate complete list using defined prefix
ARM_JOINT_NAMES = [f"{arm_prefix}{suffix}" for suffix in _ARM_SUFFIXES]

LEG_JOINT_SUFFIXES = ['_hx', '_hy', '_kn']

# Joint limits (kept the same, using integer indices)
ARM_JOINT_LIMITS = {
    0: (-3.14, 3.14),   # sh0
    1: (-3.14, 0.0),    # sh1
    2: (0.0, 3.14),     # el0
    3: (-3.14, 3.14),   # el1
    4: (-3.14, 3.14),   # wr0
    5: (-1.57, 1.57),   # wr1
}
