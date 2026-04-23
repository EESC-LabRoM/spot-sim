"""
Pose (6DOF) tracking environment - Position + Orientation control.
Refactored from attitude_tracking_env.py with proper state management and EE-relative movement.
"""
import numpy as np
import torch
from PIL import Image

from .base_env import BaseSpotEnv
from ..configs import global_config as config
from ..utils.depth_utils import calculate_camera_adjustments


class PoseTrackingEnv(BaseSpotEnv):
    """
    Environment for 6DOF pose tracking (position + orientation).
    Uses VLM to generate both position and attitude commands, CuRobo for IK.
    Movement is relative to end-effector frame for intuitive control.
    """

    def __init__(self, cfg, **kwargs):
        """
        Initialize pose tracking environment.

        Args:
            cfg: Environment configuration
            **kwargs: Additional arguments
        """
        super().__init__(cfg, **kwargs)

        # Set initial state
        self.state.control.control_interval = config.CONTROL_EVERY_N_STEPS
        self.state.fallback.max_steps = 100

        # Attitude-specific state
        self.state.vlm.current_att_command = "look_at_center"

        print("[ENV] Pose Tracking Environment initialized (6 DOF) - MODE: SEARCH")
        print("[ENV] Movement relative to EE frame")

    def _pre_physics_step(self, actions):
        """
        Pre-physics step logic (VLM processing and target calculation).
        Does NOT apply commands here - just calculates next target.

        Args:
            actions: Action tensor (not used in VLM control)
        """
        # 1. VLM processing - submit new request if interval elapsed
        if self.vlm and self.vlm.available:
            if (
                self.state.step_count - self.state.vlm.last_inference_step
            ) >= config.VLM_INFERENCE_INTERVAL:
                result = self._submit_to_vlm()
                if result:
                    self._process_vlm_result(result)
                    self.state.vlm.last_inference_step = self.state.step_count

        # 2. Calculate next target (rate-limited)
        if (
            self.state.step_count - self.state.control.last_control_step
        ) >= self.state.control.control_interval:
            if self.state.fallback.active:
                self._prepare_fallback_target()
            elif self.state.vlm.tracking_mode:
                self._prepare_arm_target()

            self.state.control.last_control_step = self.state.step_count

    def _submit_to_vlm(self):
        """Submit RGB and depth images to VLM for inference using Qwen skills with IK status"""
        try:
            # Get camera data
            rgb_data = self.scene["robot/hand_camera"].data.output["rgb"][0]
            depth_data = self.scene["robot/hand_camera"].data.output["distance_to_image_plane"][0]

            if rgb_data is None or depth_data is None:
                return None

            # Process RGB (ensure correct format)
            rgb_np = rgb_data[:, :, :3].cpu().numpy()
            # Convert from [0, 1] float to [0, 255] uint8 and ensure RGB order
            rgb_uint8 = (np.clip(rgb_np, 0, 1) * 255).astype("uint8")
            rgb_pil = Image.fromarray(rgb_uint8, mode="RGB").resize(
                (336, 336), Image.Resampling.LANCZOS
            )

            # Process depth (grayscale, not colored)
            depth_np = depth_data.cpu().numpy()
            if len(depth_np.shape) == 3:
                depth_np = depth_np[:, :, 0]

            # Normalize depth to 0-255 grayscale (clip to 5m max)
            depth_normalized = np.clip(depth_np, 0, 5.0) / 5.0
            depth_uint8 = (depth_normalized * 255).astype("uint8")
            depth_pil = Image.fromarray(depth_uint8, mode="L").resize((336, 336), Image.Resampling.LANCZOS)

            # Save images if configured
            if config.SAVE_IMAGES and (
                self.state.step_count % config.SAVE_IMAGES_EVERY_N_STEPS == 0
            ):
                rgb_pil.save("pose_tracking_rgb.jpg")
                depth_pil.save("pose_tracking_depth.jpg")

            # Calculate camera adjustments for distance
            adjustments = calculate_camera_adjustments(depth_np, target_distance=config.TARGET_DISTANCE)
            distance = adjustments["distance"]

            # Call appropriate Qwen skill based on tracking state
            if not self.state.vlm.tracking_mode:
                # SEARCH MODE
                result = self.vlm.skills.search_object(
                    target=config.TARGET_OBJECT,
                    rgb_image=rgb_pil,
                    confidence_threshold=0.7
                )
            else:
                # ATTITUDE TRACKING MODE
                # Calculate EE distance from base for workspace awareness
                curr_ee_pos, _ = self.kinematics.get_ee_pose()
                ee_distance = torch.norm(curr_ee_pos[0]).item()

                result = self.vlm.skills.control_attitude(
                    current_distance=distance,
                    ik_status="OK",  # Could add actual IK status check
                    ee_distance=ee_distance,
                    target_distance=config.TARGET_DISTANCE,
                    distance_tolerance=config.DISTANCE_TOLERANCE,
                    target_object=config.TARGET_OBJECT,
                    rgb_image=rgb_pil,
                    depth_image=depth_pil
                )

            return result

        except Exception as e:
            print(f"[ENV SUBMIT ERROR] {e}")
            return None

    def _process_vlm_result(self, response):
        """
        Process VLM response for 6DOF commands.

        Args:
            response: VLM response dictionary with pos_command and att_command
        """
        if not response:
            return

        # SEARCH MODE
        if not self.state.vlm.tracking_mode:
            if "status" in response:
                if response["status"] == "target_found":
                    print(f"[VLM] ✓ Object found! Starting tracking...")
                    self.state.vlm.tracking_mode = True
                    self.state.vlm.current_pos_command = "hold_position"
                    self.state.vlm.current_att_command = "look_at_center"
                else:
                    print(f"[VLM] Searching for object...")

        # TRACKING MODE
        else:
            # Extract position command
            if "pos_command" in response:
                new_pos_cmd = response["pos_command"].lower().strip()
            elif "command" in response:
                new_pos_cmd = response["command"].lower().strip()
            else:
                new_pos_cmd = self.state.vlm.current_pos_command

            # Extract attitude command
            if "att_command" in response:
                new_att_cmd = response["att_command"].lower().strip()
            else:
                new_att_cmd = "look_at_center"

            # Validate commands
            valid_pos_commands = [
                "move_closer",
                "move_away",
                "adjust_left",
                "adjust_right",
                "adjust_up",
                "adjust_down",
                "hold_position",
            ]

            valid_att_commands = [
                "look_at_center",
                "tilt_up",
                "tilt_down",
                "roll_left",
                "roll_right",
                "pan_left",
                "pan_right",
            ]

            if new_pos_cmd in valid_pos_commands and new_att_cmd in valid_att_commands:
                if (
                    self.state.vlm.current_pos_command != new_pos_cmd
                    or self.state.vlm.current_att_command != new_att_cmd
                ):
                    print("\n" + "=" * 20 + " VLM INFERENCE " + "=" * 20)
                    print(
                        f"[VLM] POS: {self.state.vlm.current_pos_command} -> {new_pos_cmd.upper()}"
                    )
                    print(
                        f"[VLM] ATT: {self.state.vlm.current_att_command} -> {new_att_cmd.upper()}"
                    )
                    if "reason" in response:
                        print(f"[VLM REASON] {response['reason']}")
                    print("=" * 54)

                self.state.vlm.current_pos_command = new_pos_cmd
                self.state.vlm.current_att_command = new_att_cmd

                # Reset failures on valid command
                if new_pos_cmd != "hold_position":
                    self.state.vlm.consecutive_failures = 0
            else:
                self.state.vlm.consecutive_failures += 1
                print(f"[VLM] Invalid command: pos={new_pos_cmd}, att={new_att_cmd}")

    def _prepare_arm_target(self):
        """
        Calculate next arm target (6 DOF) using IK with EE-relative movement.
        Stores result in self.state.control.pending_arm_target.
        """
        try:
            # FIX: Clear target for hold commands (fixes stale target bug)
            if (
                self.state.vlm.current_pos_command == "hold_position"
                and self.state.vlm.current_att_command == "look_at_center"
            ):
                self.state.control.pending_arm_target = None
                return

            # Get current state
            current_arm_config = self.robot_interface.get_current_arm_config()
            curr_ee_pos, curr_ee_quat = self.kinematics.get_ee_pose()

            # Generate target pose from VLM commands (EE-relative)
            tgt_pos, tgt_quat = self.command_generator.compute_relative_target(
                curr_ee_pos,
                curr_ee_quat,
                self.state.vlm.current_pos_command,
                self.state.vlm.current_att_command,
                pos_step=0.04,  # 4cm per command
                rot_step_deg=5.0,  # 5 degrees per command
            )

            # IK solve
            if self.motion_planner is not None:
                target_config = self.motion_planner.plan_to_pose(tgt_pos, tgt_quat, current_arm_config)

                if target_config is not None:
                    # Validate 6 DOF
                    if target_config.numel() != 6:
                        print(
                            f"[CONTROL] ✗ Planner returned {target_config.numel()} values (expected 6)"
                        )
                        self.state.control.pending_arm_target = None
                        return

                    # Smooth interpolation
                    self.state.control.pending_arm_target = self.robot_interface.interpolate_config(
                        current_arm_config, target_config, alpha=0.3
                    )
                    self.state.vlm.consecutive_failures = 0
                else:
                    self.state.control.pending_arm_target = None
                    self.state.vlm.consecutive_failures += 1
                    print(
                        f"[CONTROL] Failure {self.state.vlm.consecutive_failures}/"
                        f"{config.MAX_CONSECUTIVE_FAILURES} in planning"
                    )

                    # Trigger fallback if too many failures
                    if self.state.vlm.consecutive_failures >= config.MAX_CONSECUTIVE_FAILURES:
                        print("[FALLBACK] Starting return to initial position")
                        self.state.fallback.active = True
                        self.state.fallback.init_step = self.state.step_count
                        self.state.vlm.tracking_mode = False
                        self.state.vlm.current_att_command = "look_at_center"
            else:
                # No motion planner available
                self.state.control.pending_arm_target = None

        except Exception as e:
            print(f"[CONTROL ERROR] {e}")
            self.state.control.pending_arm_target = None

    def _prepare_fallback_target(self):
        """
        Calculate fallback target (6 DOF) - return to initial configuration.
        """
        current_arm_config = self.robot_interface.get_current_arm_config()
        initial_config = self.robot_interface.initial_arm_config

        # Smooth interpolation toward initial
        self.state.fallback.target_config = self.robot_interface.interpolate_config(
            current_arm_config, initial_config, alpha=0.4
        )

        self.state.fallback.steps_elapsed = self.state.step_count - self.state.fallback.init_step

        # Check completion
        distance = torch.norm(current_arm_config - initial_config).item()

        if distance < 0.05 or self.state.fallback.steps_elapsed >= self.state.fallback.max_steps:
            print("[FALLBACK] ✓ Return complete - reactivating tracking")
            self.state.fallback.active = False
            self.state.fallback.steps_elapsed = 0
            self.state.fallback.target_config = None
            self.state.vlm.consecutive_failures = 0
            self.state.vlm.tracking_mode = True  # Reactivate tracking
            self.state.vlm.current_att_command = "look_at_center"
