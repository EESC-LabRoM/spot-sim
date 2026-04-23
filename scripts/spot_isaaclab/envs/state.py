"""
State management for Spot Isaac environments using dataclasses.
Replaces 15+ flat attributes with structured state.
"""
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class VLMState:
    """VLM inference state"""
    tracking_mode: bool = False
    current_pos_command: str = "hold_position"
    current_att_command: str = "look_at_center"
    last_inference_step: int = 0
    consecutive_failures: int = 0

    def reset(self):
        """Reset VLM state"""
        self.tracking_mode = False
        self.current_pos_command = "hold_position"
        self.current_att_command = "look_at_center"
        self.consecutive_failures = 0


@dataclass
class ControlState:
    """Control system state"""
    pending_arm_target: Optional[torch.Tensor] = None
    last_control_step: int = 0
    control_interval: int = 2

    def reset(self):
        """Reset control state"""
        #self.pending_arm_target = None
        self.last_control_step = 0


@dataclass
class FallbackState:
    """Recovery/fallback state"""
    active: bool = False
    init_step: int = 0
    steps_elapsed: int = 0
    target_config: Optional[torch.Tensor] = None
    max_steps: int = 100

    def reset(self):
        """Reset fallback state"""
        self.active = False
        self.init_step = 0
        self.steps_elapsed = 0
        self.target_config = None


@dataclass
class EnvironmentState:
    """Complete environment state - replaces 15+ flat attributes"""
    step_count: int = 0
    first_reset_done: bool = False
    vlm: VLMState = field(default_factory=VLMState)
    control: ControlState = field(default_factory=ControlState)
    fallback: FallbackState = field(default_factory=FallbackState)

    def reset(self):
        """Reset all state"""
        self.step_count = 0
        self.vlm.reset()
        self.control.reset()
        self.fallback.reset()

    def __repr__(self) -> str:
        return (
            f"EnvironmentState(step={self.step_count}, "
            f"tracking={self.vlm.tracking_mode}, "
            f"fallback={self.fallback.active})"
        )
