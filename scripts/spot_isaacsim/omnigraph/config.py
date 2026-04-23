"""Configuration dataclass for the ROS 2 OmniGraph bridge."""
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class BridgeConfig:
    """Configuration for ROS bridge topics."""

    enable_joint_state: bool = True
    enable_tf: bool = True
    cameras: Optional[List[Dict[str, str]]] = None
    joint_state_topic: str = "/joint_states"
    tf_topic: str = "/tf"
    graph_path: str = "/ActionGraph_ROS"
    camera_step: int = 1   # IsaacSimulationGate step — fire every N render frames
    enable_zed: bool = True
