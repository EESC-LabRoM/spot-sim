from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.utils import configclass

from dataset_generation.spot_grasping.configs.actions import ActionsCfg
from dataset_generation.spot_grasping.configs.events import EventCfg
from dataset_generation.spot_grasping.configs.observations import ObservationsCfg
from dataset_generation.spot_grasping.configs.scene import SpotGripperSceneCfg


@configclass
class SpotEndEffectorEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the Spot end-effector environment."""

    scene: SpotGripperSceneCfg = SpotGripperSceneCfg(env_spacing=2.5, num_envs=10)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # 50 Hz
        self.episode_length_s = 20.0

        # simulation settings
        self.sim.dt = 0.005  # 200 Hz
        self.sim.render_interval = self.decimation
        self.scene.robot.spawn.joint_drive.gains.stiffness = None
