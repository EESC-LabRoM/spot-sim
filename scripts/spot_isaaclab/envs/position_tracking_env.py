"""
Position-only tracking environment (3D control).
Refactored from arm_tracking_env.py with fixes for thread safety, stale targets, and proper state management.
"""
import numpy as np
import torch
from PIL import Image

from .base_env import BaseSpotEnv
from ..configs import global_config as config
from ..utils.depth_utils import calculate_camera_adjustments


class PositionTrackingEnv(BaseSpotEnv):
    """
    Environment for 3D position tracking (no attitude control).
    Uses VLM to generate position commands, CuRobo for IK, and smooth interpolation.
    """

    def __init__(self, cfg, **kwargs):
        """
        Initialize position tracking environment.

        Args:
            cfg: Environment configuration
            **kwargs: Additional arguments
        """
        super().__init__(cfg, **kwargs)

        # Set initial state
        self.state.control.control_interval = config.CONTROL_EVERY_N_STEPS
        self.state.fallback.max_steps = 100

        init_pos,init_quat = self.kinematics.get_ee_pose()
        self.command_generator.parse_init_pose(init_pos, init_quat)

        # Deferred reactivation flag for fallback completion
        self._reactivate_tracking_next_frame = False

        print("[ENV] Position Tracking Environment initialized (6 DOF) - MODE: SEARCH")

    def _pre_physics_step(self, actions):
        """
        Pre-physics step logic (VLM processing and target calculation).
        Does NOT apply commands here - just calculates next target.

        Args:
            actions: Action tensor (not used in VLM control)
        """
        # 1. Handle deferred reactivation from fallback
        if self._reactivate_tracking_next_frame:
            self.state.vlm.tracking_mode = True
            self._reactivate_tracking_next_frame = False
            # Refresh EE pose to prevent stale cache
            curr_ee_pos, curr_ee_quat = self.robot_interface.get_current_ee_pose()
            print(f"[FALLBACK] ✓ Tracking reactivated (EE pose refreshed)")

        # 2. VLM processing - submit new request if interval elapsed
        if self.vlm and self.vlm.available:
            if (
                self.state.step_count - self.state.vlm.last_inference_step
            ) >= config.VLM_INFERENCE_INTERVAL:
                result = self._submit_to_vlm()
                if result:
                    self._process_vlm_result(result)
                    self.state.vlm.last_inference_step = self.state.step_count

        # 3. Calculate next target (rate-limited)
        if (
            self.state.step_count - self.state.control.last_control_step
        ) >= self.state.control.control_interval:
            if self.state.fallback.active:
                self._prepare_fallback_target()
            elif self.state.vlm.tracking_mode:
                self._prepare_arm_target()

            self.state.control.last_control_step = self.state.step_count

    def _submit_to_vlm(self):
        """Submit RGB and depth images to VLM for inference using Qwen skills"""
        try:
            # Get camera data
            rgb_data = self.scene["robot_camera_hand"].data.output["rgb"][0]
            depth_data = self.scene["robot_camera_hand"].data.output["distance_to_image_plane"][0]

            if rgb_data is None or depth_data is None:
                return None

            # Process RGB (ensure correct format)
            rgb_np = rgb_data[:, :, :3].cpu().numpy()

            if rgb_np.max() <= 1.0: rgb_np = (rgb_np * 255).astype('uint8')
            else: rgb_np = rgb_np.astype('uint8')
            if rgb_np.shape[-1] > 3: rgb_np = rgb_np[:, :, :3]

            rgb_pil = Image.fromarray(rgb_np, mode="RGB").resize((336, 336), Image.Resampling.LANCZOS)

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
                rgb_pil.save("position_tracking_rgb.jpg")
                depth_pil.save("position_tracking_depth.jpg")

            # Calculate camera adjustments for distance
            adjustments = calculate_camera_adjustments(depth_np, target_distance=config.TARGET_DISTANCE)
            distance = adjustments["distance"]

            # Call appropriate Qwen skill based on tracking state
            if not self.state.vlm.tracking_mode:
                # SEARCH MODE
                result = self.vlm.skills.search_object(
                    target=config.TARGET_OBJECT,
                    rgb_image=rgb_pil,
                    confidence_threshold=0.7,
                    use_async=True  # Enable async mode
                )
            else:
                # TRACKING MODE
                result = self.vlm.skills.track_position(
                    current_distance=distance,
                    target_distance=config.TARGET_DISTANCE,
                    distance_tolerance=config.DISTANCE_TOLERANCE,
                    target_object=config.TARGET_OBJECT,
                    rgb_image=rgb_pil,
                    depth_image=depth_pil,
                    use_async=True  # Enable async mode
                )

            return result

        except Exception as e:
            print(f"[ENV SUBMIT ERROR] {e}")
            return None

    def _process_vlm_result(self, response):
        """
        Process VLM response and update state.

        Args:
            response: VLM response dictionary
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
                else:
                    print(f"[VLM] Searching for object...")
                    #print(f"[VLM REASON] {response}")

        # TRACKING MODE
        else:
            if "command" in response:
                new_command = response["command"].lower().strip()
                valid_commands = [
                    "move_closer",
                    "move_away",
                    "adjust_left",
                    "adjust_right",
                    "adjust_up",
                    "adjust_down",
                    "hold_position",
                ]

                if new_command in valid_commands:
                    if self.state.vlm.current_pos_command != new_command:
                        print("\n" + "=" * 20 + " VLM INFERENCE " + "=" * 20)
                        print(f"[VLM] {self.state.vlm.current_pos_command} -> {new_command.upper()}")
                        if "reason" in response:
                            print(f"[VLM REASON] {response['reason']}")
                        print("=" * 54)

                    self.state.vlm.current_pos_command = new_command

                    # NOTE: Don't reset failures here - only reset on successful IK (line 233)
                    # VLM can give valid commands even when IK is failing
                else:
                    self.state.vlm.consecutive_failures += 1
                    print(f"[VLM] Invalid command: {new_command}")

    def _prepare_arm_target(self):
        """
        Calculate next arm target (6 DOF) using IK.
        Stores result in self.state.control.pending_arm_target.
        """
        try:
            # FIX: Clear target for hold commands (fixes stale target bug)
            if self.state.vlm.current_pos_command in ["hold_position"]:
                self.state.control.pending_arm_target = None
                return

            # Get current state
            current_arm_config = self.robot_interface.get_current_arm_config()
            curr_ee_pos, curr_ee_quat = self.kinematics.get_ee_pose()

            # Generate target pose from VLM command
            # Use legacy absolute command for backward compatibility
            tgt_pos, tgt_quat = self.command_generator.compute_absolute_target(
                curr_ee_pos, curr_ee_quat, self.state.vlm.current_pos_command, step=0.04
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
        # Ensure tensors are on correct device
        current_arm_config = self.robot_interface.get_current_arm_config().to(self.device)
        initial_config = self.robot_interface.initial_arm_config.to(self.device)

        # Smooth interpolation toward initial
        self.state.fallback.target_config = self.robot_interface.interpolate_config(
            current_arm_config, initial_config, alpha=0.4
        ).to(self.device)

        self.state.fallback.steps_elapsed = self.state.step_count - self.state.fallback.init_step

        # Check completion
        distance = torch.norm(current_arm_config - initial_config).item()

        if distance < 0.05 or self.state.fallback.steps_elapsed >= self.state.fallback.max_steps:
            print("[FALLBACK] ✓ Return complete - scheduling reactivation")
            self.state.fallback.active = False
            self.state.fallback.steps_elapsed = 0
            self.state.fallback.target_config = None
            self.state.vlm.consecutive_failures = 0
            # Defer reactivation to next frame (allows state to settle)
            self._reactivate_tracking_next_frame = True
