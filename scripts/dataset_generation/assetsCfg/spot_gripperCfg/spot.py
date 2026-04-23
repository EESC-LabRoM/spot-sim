# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""Configuration for the Boston Dynamics robot.

The following configuration parameters are available:

* :obj:`SPOT_ARM_CFG`: The Spot Arm robot with delay PD and remote PD actuators.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from dataset_generation.assetsCfg import ASSETCFG_DIR
from dataset_generation.assetsCfg.spot_gripperCfg.constants import (
    ARM_ARMATURE,
    ARM_DAMPING,
    ARM_EFFORT_LIMIT,
    ARM_STIFFNESS,
    SPOT_DEFAULT_JOINT_POS,
    SPOT_DEFAULT_POS,
    SPOT_DEFAULT_JOINT_VEL,
)

##
# Configuration
##s


SPOT_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=False,
        link_density=1.0e-8,
        asset_path=f"{ASSETCFG_DIR}/spot_gripperCfg/spot_gripper.urdf",
        semantic_tags=[("Robot", "spot_gripper")],
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=SPOT_DEFAULT_POS,
        joint_pos=SPOT_DEFAULT_JOINT_POS,
        joint_vel=SPOT_DEFAULT_JOINT_VEL,
    ),
    actuators={
        "spot_arm_wr1": ImplicitActuatorCfg(
            joint_names_expr=["arm_wr1"],
            effort_limit_sim=ARM_EFFORT_LIMIT[0],
            stiffness=ARM_STIFFNESS[0],
            damping=ARM_DAMPING[0],
            armature=ARM_ARMATURE[0],
        ),
        "spot_arm_f1x": ImplicitActuatorCfg(
            joint_names_expr=["arm_f1x"],
            effort_limit_sim=ARM_EFFORT_LIMIT[1],
            stiffness=ARM_STIFFNESS[1],
            damping=ARM_DAMPING[1],
            armature=ARM_ARMATURE[1],
        ),
    },
)
