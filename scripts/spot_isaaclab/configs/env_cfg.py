"""
Environment configurations for Spot Isaac Lab environments.
Wires up ActionTerms, ObservationTerms, and EventTerms.
"""
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import ObservationTermCfg, ObservationGroupCfg, ActionTermCfg, EventTermCfg
from isaaclab.utils import configclass

from configs import global_config as config

# Import local scene configs
from configs.scene_config import (
    SpotManipulationSceneCfg,
    SpotManipulationWithObjectsSceneCfg,
)

# Import Isaac Lab compliance layer
from ..isaac_lab import actions, observations, events


@configclass
class SpotObservationsCfg:
    """Observation configuration for Spot environments"""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observations"""

        # Arm state
        arm_joint_pos = ObservationTermCfg(func=observations.arm_joint_positions)
        arm_joint_vel = ObservationTermCfg(func=observations.arm_joint_velocities)

        # End-effector state
        ee_position = ObservationTermCfg(func=observations.ee_position)
        ee_orientation = ObservationTermCfg(func=observations.ee_orientation)

        # Base state
        base_position = ObservationTermCfg(func=observations.base_position)
        base_orientation = ObservationTermCfg(func=observations.base_orientation)

        # Camera data (optional, can be disabled for performance)
        # camera_rgb = ObservationTermCfg(func=observations.camera_rgb)
        # camera_depth = ObservationTermCfg(func=observations.camera_depth)

        # VLM tracking state
        vlm_state = ObservationTermCfg(func=observations.vlm_tracking_state)

    policy: PolicyCfg = PolicyCfg()


@configclass
class SpotActionsCfg:
    """Action configuration for Spot environments"""

    # Use position control by default
    arm_action = ActionTermCfg(
        class_type=actions.SpotArmPositionAction,
        asset_name="robot",
    )

    # Alternative: Use velocity control
    # arm_action = ActionTermCfg(
    #     class_type=actions.SpotArmVelocityAction,
    #     asset_name="robot",
    # )


@configclass
class SpotEventsCfg:
    """Event configuration for Spot environments"""

    # Reset events
    reset_scene = EventTermCfg(
        func=events.reset_scene_to_default,
        mode="reset",
    )

    # Domain randomization (optional)
    # randomize_arm = EventTermCfg(
    #     func=events.randomize_arm_joint_positions,
    #     mode="reset",
    #     params={"position_range": (-0.05, 0.05)},
    # )

    # randomize_base = EventTermCfg(
    #     func=events.randomize_base_position,
    #     mode="reset",
    #     params={"position_range": (-0.1, 0.1)},
    # )


@configclass
class PositionTrackingEnvCfg(ManagerBasedEnvCfg):
    """Configuration for position tracking environment (3D control)"""

    # Scene configuration
    if config.USE_EXTENDED_SCENE:
        scene: SpotManipulationWithObjectsSceneCfg = SpotManipulationWithObjectsSceneCfg(
            num_envs=1, env_spacing=2.0
        )
    else:
        scene: SpotManipulationSceneCfg = SpotManipulationSceneCfg(num_envs=1, env_spacing=2.0)

    # Manager configurations
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()
    events: SpotEventsCfg = SpotEventsCfg()

    def __post_init__(self):
        """Post-initialization to set runtime parameters"""
        # Episode settings
        self.decimation = 1
        self.episode_length_s = 100.0

        # Viewer settings
        self.viewer.eye = (2.0, 2.0, 1.0)
        self.viewer.lookat = (0.3, 0.0, 0.0)

        # Simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = config.RENDER_EVERY_N_STEPS
        self.sim.device = "cuda:0"

        # PhysX settings
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_max_rigid_contact_count = 2**20
        self.sim.physx.enable_stabilization = True
        self.sim.physx.solver_type = 1
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**20
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**20
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005


@configclass
class PoseTrackingEnvCfg(ManagerBasedEnvCfg):
    """Configuration for pose tracking environment (6DOF control)"""

    # Scene configuration
    if config.USE_EXTENDED_SCENE:
        scene: SpotManipulationWithObjectsSceneCfg = SpotManipulationWithObjectsSceneCfg(
            num_envs=1, env_spacing=2.0
        )
    else:
        scene: SpotManipulationSceneCfg = SpotManipulationSceneCfg(num_envs=1, env_spacing=2.0)

    # Manager configurations
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()
    events: SpotEventsCfg = SpotEventsCfg()

    def __post_init__(self):
        """Post-initialization to set runtime parameters"""
        # Episode settings
        self.decimation = 1
        self.episode_length_s = 100.0

        # Viewer settings
        self.viewer.eye = (2.0, 2.0, 1.0)
        self.viewer.lookat = (0.3, 0.0, 0.0)

        # Simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = config.RENDER_EVERY_N_STEPS
        self.sim.device = "cuda:0"

        # PhysX settings
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_max_rigid_contact_count = 2**20
        self.sim.physx.enable_stabilization = True
        self.sim.physx.solver_type = 1
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**20
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**20
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005
