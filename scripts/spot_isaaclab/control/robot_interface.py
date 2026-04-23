"""
Hardware abstraction layer for Spot robot.
Handles joint control, interpolation, limits, and gripper.
"""
import torch
from typing import Dict, Tuple, Optional


class SpotRobotInterface:
    """Low-level robot control interface"""

    def __init__(
        self,
        robot,
        arm_indices,
        leg_indices,
        gripper_index: Optional[int] = None,
        device="cuda:0",
    ):
        """
        Initialize robot interface.

        Args:
            robot: Isaac Lab Articulation object
            arm_indices: List of 6 arm joint indices
            leg_indices: List of leg joint indices (for base locking)
            gripper_index: Optional gripper joint index
            device: Torch device
        """
        self.robot = robot
        self.arm_indices = arm_indices
        self.leg_indices = leg_indices
        self.gripper_index = gripper_index
        self.device = device

        # Save initial configurations
        self.initial_arm_config = robot.data.joint_pos[0, arm_indices].clone().to(device)
        self.standing_leg_config = robot.data.joint_pos[0, leg_indices].clone().to(device)

        # Gripper
        self.gripper_open_position = -1.0

        # Joint limits (6 DOF arm)
        self.arm_limits: Dict[int, Tuple[float, float]] = {
            0: (-2.618, 3.142),  # sh0
            1: (-3.142, 0.524),  # sh1
            2: (0.0, 3.142),  # el0
            3: (-2.793, 2.793),  # el1
            4: (-1.833, 1.833),  # wr0
            5: (-2.880, 2.880),  # wr1
        }

        # Workspace limits (base frame)
        self.workspace_limits = {
            "x": (0.25, 0.75),
            "y": (-0.35, 0.35),
            "z": (0.15, 0.95),
        }

        print(f"[ROBOT_IF] Initialized: 6 DOF arm + {len(leg_indices)} leg joints")
        if gripper_index is not None:
            print(f"[ROBOT_IF] Gripper at index {gripper_index}")

    def apply_arm_target(
        self,
        arm_config: torch.Tensor,
        lock_base: bool = True,
        gripper_open: float = -1.0,
    ):
        """
        Apply joint position targets to robot.

        Args:
            arm_config: [6] desired arm joint positions
            lock_base: If True, lock legs to standing configuration
            gripper_open: Gripper position (-1.0 = open, 1.0 = closed)
        """
        # Clone current joint positions
        full_target = self.robot.data.joint_pos.clone().to(self.device)

        # Clamp arm config to joint limits
        clamped = arm_config.clone()
        for i, (min_val, max_val) in self.arm_limits.items():
            clamped[i] = torch.clamp(clamped[i], min_val, max_val)

        # Set arm joints
        full_target[0, self.arm_indices] = clamped

        # Lock base if requested
        if lock_base:
            full_target[0, self.leg_indices] = self.standing_leg_config

        # Set gripper if available
        if self.gripper_index is not None:
            full_target[0, self.gripper_index] = gripper_open

        # Apply targets
        self.robot.set_joint_position_target(full_target)

    def interpolate_config(
        self,
        current: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Smooth interpolation between configurations.

        Args:
            current: Current arm configuration [6]
            target: Target arm configuration [6]
            alpha: Interpolation factor (0 = current, 1 = target)

        Returns:
            Interpolated configuration [6]
        """
        return current + alpha * (target - current)

    def check_workspace_limits(self, position: torch.Tensor) -> bool:
        """
        Check if position is within workspace limits.

        Args:
            position: [3] position in world frame

        Returns:
            True if within limits, False otherwise
        """
        x, y, z = position[0].item(), position[1].item(), position[2].item()

        x_min, x_max = self.workspace_limits["x"]
        y_min, y_max = self.workspace_limits["y"]
        z_min, z_max = self.workspace_limits["z"]

        within_limits = (
            x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max
        )

        if not within_limits:
            print(
                f"[WARN] Position out of workspace: x={x:.2f}, y={y:.2f}, z={z:.2f}"
            )

        return within_limits

    def get_current_arm_config(self) -> torch.Tensor:
        """
        Get current arm joint positions.

        Returns:
            [6] arm joint positions
        """
        return self.robot.data.joint_pos[0, self.arm_indices].clone()

    def reset_to_initial(self):
        """Reset arm to initial configuration"""
        self.apply_arm_target(self.initial_arm_config, lock_base=True)

    def get_current_ee_pose(self):
        """Obtém pose atual do end-effector NO MUNDO."""
        kin_state = self.robot.root_physx_view.get_link_transforms()
        body_names = self.robot.body_names

        if "arm_link_wr1" in body_names:
            ee_idx = body_names.index("arm_link_wr1")
        else:
            print("[WARN] arm_link_wr1 não encontrado!")
            return torch.tensor([0.5, 0., 0.3], device=self.device), \
                   torch.tensor([1., 0., 0., 0.], device=self.device)

        ee_pos_w = kin_state[0, ee_idx, :3].clone()
        ee_quat_w = kin_state[0, ee_idx, 3:].clone()

        return ee_pos_w, ee_quat_w
