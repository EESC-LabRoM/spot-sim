"""Configuration dataclass for the ROS 2 OmniGraph bridge."""
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class BridgeConfig:
    """Configuration for ROS bridge topics."""

    enable_joint_state: bool = True
    enable_tf: bool = True
    camera_specs: Optional[List[Dict[str, str]]] = None
    publishing_cameras: List[str] = field(default_factory=lambda: ["hand"])
    camera_hz: float = 10.0
    joint_state_topic: str = "/joint_states"
    tf_topic: str = "/tf"
    graph_path: str = "/ActionGraph_ROS"
    camera_step: int = 1   # IsaacSimulationGate step — computed from camera_hz by build()
    enable_zed: bool = True
