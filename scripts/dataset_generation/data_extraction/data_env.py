from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.utils import configclass

from dataset_generation.data_extraction.configs.actions import ActionsCfg
from dataset_generation.data_extraction.configs.events import EventCfg
from dataset_generation.data_extraction.configs.observations import ObservationsCfg
from dataset_generation.data_extraction.configs.scene import RawDataExtractionSceneCfg


@configclass
class RawDataExtractionEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the Raw Data Extraction environment."""

    scene: RawDataExtractionSceneCfg = RawDataExtractionSceneCfg(
        env_spacing=2.5, num_envs=10
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.sim.dt = 0.005

        super().__post_init__()
