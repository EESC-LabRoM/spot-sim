"""
Converts VLM commands to target poses for the robot.
Uses frame transformations to handle camera-relative movements.
"""
import torch
from typing import Tuple
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, quat_error_magnitude
from .frame_transforms import FrameTransformer


class CommandToPoseGenerator:
    """Generates target poses from VLM commands"""

    def __init__(self, device="cuda:0"):
        """
        Initialize command generator.

        Args:
            device: Torch device
        """
        self.device = device
        self.frame_transformer = FrameTransformer(device)

    def parse_init_pose(self, ee_pos: torch.Tensor, ee_quat: torch.Tensor):
        """Parse initial pose command (not implemented)"""
        self.ee_cur_pos = ee_pos.clone()
        self.ee_cur_quat = ee_quat.clone()

    def compute_relative_target(
        self,
        current_pos: torch.Tensor,
        current_quat: torch.Tensor,
        pos_command: str,
        att_command: str,
        pos_step: float = 0.04,
        rot_step_deg: float = 5.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute target pose from VLM commands (camera-relative).

        Position commands (camera frame):
        - "move_closer": +X camera (forward)
        - "move_away": -X camera (backward)
        - "adjust_right": +Y camera (right)
        - "adjust_left": -Y camera (left)
        - "adjust_up": +Z camera (up)
        - "adjust_down": -Z camera (down)
        - "hold_position": No movement

        Attitude commands (incremental):
        - "look_up": Pitch negative
        - "look_down": Pitch positive
        - "look_left": Yaw positive
        - "look_right": Yaw negative
        - "look_at_center": No rotation
        - "reset_attitude": Return to neutral orientation

        Args:
            current_pos: Current EE position (world frame) [3]
            current_quat: Current EE orientation (world frame) [4]
            pos_command: Position command string
            att_command: Attitude command string
            pos_step: Position step size (meters)
            rot_step_deg: Rotation step size (degrees)

        Returns:
            (target_pos, target_quat): Target pose in world frame
        """
        # Ensure correct device
        current_pos = current_pos.to(self.device)
        current_quat = current_quat.to(self.device)

        # 1. POSITION COMMAND → Camera displacement
        camera_disp = torch.zeros(3, device=self.device)

        if pos_command == "move_closer":
            camera_disp[0] = pos_step
        elif pos_command == "move_away":
            camera_disp[0] = -pos_step
        elif pos_command == "adjust_right":
            camera_disp[1] = pos_step
        elif pos_command == "adjust_left":
            camera_disp[1] = -pos_step
        elif pos_command == "adjust_up":
            camera_disp[2] = pos_step
        elif pos_command == "adjust_down":
            camera_disp[2] = -pos_step
        # "hold_position" → zero displacement

        # Transform to world frame
        world_disp = self.frame_transformer.camera_displacement_to_world(camera_disp, current_quat)
        target_pos = current_pos + world_disp

        # 2. ATTITUDE COMMAND → Orientation delta
        rot_step = rot_step_deg * (torch.pi / 180.0)
        roll, pitch, yaw = 0.0, 0.0, 0.0

        if att_command == "look_up":
            pitch = -rot_step  # Negative pitch = look up
        elif att_command == "look_down":
            pitch = rot_step  # Positive pitch = look down
        elif att_command == "look_left":
            yaw = rot_step  # Positive yaw = turn left
        elif att_command == "look_right":
            yaw = -rot_step  # Negative yaw = turn right
        elif att_command == "reset_attitude":
            # Return to neutral orientation (identity quaternion)
            target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            return target_pos, target_quat

        # Apply incremental rotation
        if att_command != "look_at_center":
            rot_quat = quat_from_euler_xyz(
                torch.tensor([roll], device=self.device),
                torch.tensor([pitch], device=self.device),
                torch.tensor([yaw], device=self.device),
            )

            # Ensure compatible shapes
            if current_quat.dim() == 1:
                current_quat = current_quat.unsqueeze(0)
            if rot_quat.dim() == 1:
                rot_quat = rot_quat.unsqueeze(0)

            # Apply rotation: target = current * incremental
            target_quat = quat_mul(current_quat, rot_quat).squeeze(0)

            # VALIDATION: Check if total rotation is not too extreme
            neutral_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

            if target_quat.dim() == 1:
                error = quat_error_magnitude(target_quat.unsqueeze(0), neutral_quat.unsqueeze(0))
            else:
                error = quat_error_magnitude(target_quat, neutral_quat.unsqueeze(0))

            max_rotation = 1.57  # ~90 degrees
            if error.item() > max_rotation:
                print(f"[WARN] Rotation too large ({error.item():.2f} rad), clamping")
                target_quat = current_quat.clone().squeeze(0)
        else:
            target_quat = current_quat.clone()

        return target_pos, target_quat

    def compute_absolute_target(
        self,
        current_pos: torch.Tensor,
        current_quat: torch.Tensor,
        command: str,
        step: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Legacy: Compute target pose from command (absolute world frame).
        Kept for compatibility with old code.

        Args:
            current_pos: Current position [3]
            current_quat: Current orientation [4]
            command: Movement command
            step: Step size

        Returns:
            (target_pos, target_quat): Target pose
        """
        #self.ee_cur_pos = current_pos.clone()
        target_pos = current_pos.clone()
        target_quat = current_quat.clone()

        if command == "move_closer":
            target_pos[0] += step
        elif command == "move_away":
            target_pos[0] -= step
        elif command == "adjust_right":
            target_pos[1] += step
        elif command == "adjust_left":
            target_pos[1] -= step
        elif command == "adjust_up":
            target_pos[2] += step
        elif command == "adjust_down":
            target_pos[2] -= step

        if command not in ["adjust_up", "adjust_down"]:
            target_pos[2] = self.ee_cur_pos[2]
        else:
            self.ee_cur_pos[2] = target_pos[2]

        if command not in ["adjust_left", "adjust_right"]:
            target_pos[1] = self.ee_cur_pos[1]
        else:
            self.ee_cur_pos[1] = target_pos[1]

        if command not in ["move_closer", "move_away"]:
            target_pos[0] = self.ee_cur_pos[0]
        else:
            self.ee_cur_pos[0] = target_pos[0]

        return target_pos, target_quat
