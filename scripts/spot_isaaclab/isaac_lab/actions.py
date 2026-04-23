"""
Isaac Lab ActionTerm implementations for Spot robot.
Provides proper integration with Isaac Lab's ActionManager.
"""
import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.assets import Articulation
from isaaclab.utils import configclass


class SpotArmPositionAction(ActionTerm):
    """
    Action term for 6-DOF arm position control.
    Replaces custom _apply_control_once() method with proper Isaac Lab integration.
    """

    cfg: ActionTermCfg

    def __init__(self, cfg: ActionTermCfg, env):
        """
        Initialize arm position action.

        Args:
            cfg: Action term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Get arm articulation
        self._asset: Articulation = env.scene[cfg.asset_name]

        # Get arm joint indices (6 DOF, excluding gripper)
        arm_names = [n for n in self._asset.data.joint_names if n.startswith("arm")]
        if len(arm_names) >= 6:
            self._arm_indices = [self._asset.data.joint_names.index(n) for n in arm_names[:6]]
        else:
            raise ValueError(f"Expected at least 6 arm joints, found {len(arm_names)}")

        # Get leg indices for base locking
        leg_names = [n for n in self._asset.data.joint_names if any(x in n for x in ["_hx", "_hy", "_kn"])]
        self._leg_indices = [self._asset.data.joint_names.index(n) for n in leg_names]

        # Save standing configuration
        self._standing_leg_config = self._asset.data.joint_pos[0, self._leg_indices].clone()

        # Action buffers
        self._raw_actions = torch.zeros(env.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        print(f"[ACTION] SpotArmPositionAction initialized: 6 DOF arm")

    @property
    def action_dim(self) -> int:
        """Dimension of the action term"""
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the term"""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The actions computed after processing"""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """
        Process actions (called once per environment step).

        Args:
            actions: Raw actions [num_envs, 6]
        """
        # Store raw actions
        self._raw_actions[:] = actions

        # Process: scale, clip, etc. (currently identity)
        self._processed_actions[:] = actions

    def apply_actions(self):
        """
        Apply processed actions to asset (called every simulation step).
        Sets joint position targets for arm and locks base.
        """
        # Clone current joint positions
        joint_targets = self._asset.data.joint_pos.clone()

        # Set arm targets
        joint_targets[:, self._arm_indices] = self._processed_actions

        # Lock base (legs)
        joint_targets[:, self._leg_indices] = self._standing_leg_config

        # Apply targets
        self._asset.set_joint_position_target(joint_targets)


@configclass
class SpotArmPositionActionCfg(ActionTermCfg):
    """Configuration for Spot arm position action."""

    class_type: type = SpotArmPositionAction
    asset_name: str = "robot"


class SpotArmVelocityAction(ActionTerm):
    """
    Action term for 6-DOF arm velocity control.
    Alternative to position control for smoother movements.
    """

    cfg: ActionTermCfg

    def __init__(self, cfg: ActionTermCfg, env):
        super().__init__(cfg, env)

        self._asset: Articulation = env.scene[cfg.asset_name]

        # Get arm joint indices
        arm_names = [n for n in self._asset.data.joint_names if n.startswith("arm")][:6]
        self._arm_indices = [self._asset.data.joint_names.index(n) for n in arm_names]

        # Get leg indices
        leg_names = [n for n in self._asset.data.joint_names if any(x in n for x in ["_hx", "_hy", "_kn"])]
        self._leg_indices = [self._asset.data.joint_names.index(n) for n in leg_names]

        # Action buffers
        self._raw_actions = torch.zeros(env.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        print(f"[ACTION] SpotArmVelocityAction initialized: 6 DOF arm")

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Process velocity actions"""
        self._raw_actions[:] = actions
        # Could add scaling/clipping here
        self._processed_actions[:] = actions * 0.5  # Scale velocities

    def apply_actions(self):
        """Apply velocity targets"""
        # Clone current velocities
        joint_vel_targets = torch.zeros_like(self._asset.data.joint_vel)

        # Set arm velocity targets
        joint_vel_targets[:, self._arm_indices] = self._processed_actions

        # Lock base (zero velocity)
        joint_vel_targets[:, self._leg_indices] = 0.0

        # Apply targets
        self._asset.set_joint_velocity_target(joint_vel_targets)


@configclass
class SpotArmVelocityActionCfg(ActionTermCfg):
    """Configuration for Spot arm velocity action."""

    class_type: type = SpotArmVelocityAction
    asset_name: str = "robot"
