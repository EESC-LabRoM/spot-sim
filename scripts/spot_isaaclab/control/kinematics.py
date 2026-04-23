"""
Kinematic utilities for Spot arm.
Provides forward kinematics (FK) wrappers for end-effector and base poses.
"""
import torch
from typing import Tuple


class SpotKinematics:
    """Forward kinematics wrapper for Spot robot"""

    def __init__(self, robot, arm_indices, device="cuda:0"):
        """
        Initialize kinematics interface.

        Args:
            robot: Isaac Lab Articulation object
            arm_indices: List of arm joint indices
            device: Torch device
        """
        self.robot = robot
        self.arm_indices = arm_indices
        self.device = device

    def get_ee_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get end-effector pose in world frame.

        Returns:
            (ee_pos, ee_quat): Position [3] and quaternion [4] in world frame
        """
        kin_state = self.robot.root_physx_view.get_link_transforms()
        body_names = self.robot.body_names

        if "arm_link_wr1" in body_names:
            ee_idx = body_names.index("arm_link_wr1")
        else:
            print("[WARN] arm_link_wr1 not found in body_names!")
            # Return default pose
            return (
                torch.tensor([0.5, 0.0, 0.3], device=self.device),
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device),
            )

        ee_pos = kin_state[0, ee_idx, :3].clone()
        ee_quat = kin_state[0, ee_idx, 3:].clone()

        return ee_pos, ee_quat

    def get_base_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get base pose in world frame.

        Returns:
            (base_pos, base_quat): Position [3] and quaternion [4] in world frame
        """
        kin_state = self.robot.root_physx_view.get_link_transforms()
        body_names = self.robot.body_names

        base_idx = body_names.index("body") if "body" in body_names else 0
        base_pos = kin_state[0, base_idx, :3].clone()
        base_quat = kin_state[0, base_idx, 3:].clone()

        return base_pos, base_quat

    def get_ee_to_base_distance(self) -> float:
        """
        Calculate Euclidean distance from end-effector to base.

        Returns:
            Distance in meters
        """
        ee_pos, _ = self.get_ee_pose()
        base_pos, _ = self.get_base_pose()
        distance = torch.norm(ee_pos - base_pos).item()
        return distance
