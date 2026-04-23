"""
Base environment for all Spot tracking tasks.
Provides shared functionality: robot setup, VLM integration, ROS bridge, control application.
"""
from abc import abstractmethod
import gymnasium as gym
import numpy as np
import torch
import carb
import omni
import weakref

from isaaclab.envs import ManagerBasedEnv

# Import our new modules
from ..control.kinematics import SpotKinematics
from ..control.frame_transforms import FrameTransformer
from ..control.robot_interface import SpotRobotInterface
from ..control.command_generator import CommandToPoseGenerator
from modules.models_interface.qwen import Qwen
from ..bridge.omnigraph_builder import ROSBridgeBuilder
from .state import EnvironmentState

# Import from spot_isaac configs and utils
from ..configs import global_config as config
from ..utils.robot_initialization import initialize_robot_sequence, get_arm_joint_info

# Try to import Curobo
try:
    from ..control.motion_planner import CuroboMotionPlanner

    CUROBO_AVAILABLE = True
except (ImportError, Exception, RuntimeError) as e:
    print(f"[WARN] Curobo unavailable: {e}")
    CUROBO_AVAILABLE = False


class BaseSpotEnv(ManagerBasedEnv, gym.Env):
    """
    Base class for Spot tracking environments.
    Provides shared functionality for all tracking tasks.
    """

    def __init__(self, cfg, **kwargs):
        """
        Initialize base environment.

        Args:
            cfg: Environment configuration
            **kwargs: Additional arguments
        """
        super().__init__(cfg=cfg)

        # Gym spaces (6 DOF arm)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Initialize state management
        self.state = EnvironmentState()

        # Setup components
        self._setup_robot()
        self._setup_vlm()
        self._setup_ros_bridge()
        self._setup_keyboard()

        print(f"[ENV] {self.__class__.__name__} initialized")

    def _setup_robot(self):
        """Initialize robot interfaces and control modules"""
        self.robot = self.scene["robot"]

        # Get all arm joint indices (including gripper)
        arm_joint_indices_all, _, _ = get_arm_joint_info(self.robot)

        # Apply visual colors if configured
        if config.APPLY_SPOT_COLORS:
            try:
                from scripts.spot_isaac.utils.visualization import apply_spot_colors

                apply_spot_colors(self.sim)
            except Exception:
                pass

        # Initialize robot sequence (stabilization, initial pose)
        initialize_robot_sequence(self.robot, self.scene, self.sim, arm_joint_indices_all)

        # Separate 6 DOF arm from gripper
        if len(arm_joint_indices_all) == 7:
            self.arm_indices = arm_joint_indices_all[:6]
            self.gripper_index = arm_joint_indices_all[-1]
            print(f"[ENV] Separated: 6 DOF arm + 1 gripper")
        else:
            self.arm_indices = arm_joint_indices_all
            self.gripper_index = None
            print(f"[ENV] Using {len(arm_joint_indices_all)} DOF")

        # Get leg indices for base locking
        leg_names = [n for n in self.robot.data.joint_names if any(x in n for x in ["_hx", "_hy", "_kn"])]
        self.leg_indices = [self.robot.data.joint_names.index(n) for n in leg_names]

        # Initialize control modules
        self.kinematics = SpotKinematics(self.robot, self.arm_indices, self.device)
        self.frame_transformer = FrameTransformer(self.device)
        self.robot_interface = SpotRobotInterface(
            self.robot, self.arm_indices, self.leg_indices, self.gripper_index, self.device
        )
        self.command_generator = CommandToPoseGenerator(self.device)

        # Motion planner (optional)
        self.motion_planner = None
        if CUROBO_AVAILABLE and config.USE_CUROBO:
            try:
                # Pass environment device so motion planner can convert outputs back
                self.motion_planner = CuroboMotionPlanner(
                    self.robot,
                    device="cuda:0",  # CuRobo must run on GPU
                    caller_device=str(self.device)  # Environment device (cpu, cuda:0, etc.)
                )
                print("[ENV] ✓ Curobo initialized (6 DOF)")
            except Exception as e:
                print(f"[ENV] Curobo unavailable: {e}")

        # Set initial arm position
        self.robot_interface.reset_to_initial()

        # Set pending target to prevent early falling
        self.state.control.pending_arm_target = self.robot_interface.initial_arm_config.clone()

        print("[ENV] ✓ Robot initialized")

    def _setup_vlm(self):
        """Initialize VLM using Qwen skills directly"""
        if config.VLM_ENABLED:
            try:
                self.vlm = Qwen(model_path=config.VLM_MODEL_PATH, device="cuda:0", auto_async=True)
                print(f"[ENV] ✓ VLM started (Qwen with async skills)")
            except Exception as e:
                print(f"[ENV] VLM setup failed: {e}")
                self.vlm = None
        else:
            self.vlm = None

    def _setup_ros_bridge(self):
        """Setup ROS 2 bridge for NVBlox using simplified interface"""
        # Only setup for first environment to avoid duplicate topics
        if self.scene.num_envs > 1:
            print("[WARN] ROS Bridge only for env 0")
            return

        from pathlib import Path
        urdf_path = Path("assets/spot/spot_with_arm.urdf").resolve()
        print(f"[ENV] Setting up URDF published based on {urdf_path}")

        # Define all cameras to publish (hand + 5 fisheye body cameras)
        cameras = [
            {
                "name": "hand",
                "scene_key": "robot_camera_hand",
                "frame_id": "hand_camera",
                "rgb_topic": "/spot/camera/hand/image",
                "depth_topic": "/spot/depth_registered/hand/image"
            },
            {
                "name": "frontleft",
                "scene_key": "robot_camera_frontleft",
                "frame_id": "frontleft_fisheye",
                "rgb_topic": "/spot/camera/frontleft/image",
                "depth_topic": "/spot/depth_registered/frontleft/image"
            },
            {
                "name": "frontright",
                "scene_key": "robot_camera_frontright",
                "frame_id": "frontright_fisheye",
                "rgb_topic": "/spot/camera/frontright/image",
                "depth_topic": "/spot/depth_registered/frontright/image"
            },
            {
                "name": "left",
                "scene_key": "robot_camera_left",
                "frame_id": "left_fisheye",
                "rgb_topic": "/spot/camera/left/image",
                "depth_topic": "/spot/depth_registered/left/image"
            },
            {
                "name": "right",
                "scene_key": "robot_camera_right",
                "frame_id": "right_fisheye",
                "rgb_topic": "/spot/camera/right/image",
                "depth_topic": "/spot/depth_registered/right/image"
            },
            {
                "name": "back",
                "scene_key": "robot_camera_back",
                "frame_id": "back_fisheye",
                "rgb_topic": "/spot/camera/back/image",
                "depth_topic": "/spot/depth_registered/back/image"
            },
        ]

        # Instance-based ROS bridge builder with multi-camera support
        builder = ROSBridgeBuilder(
            robot=self.robot,
            scene=self.scene,
            cameras=cameras,
            enable_robot_description=False,
            urdf_path=urdf_path
        )

        if builder.success:
            print("[ENV] ✓ ROS Bridge created with 6 cameras")
        else:
            print(f"[ERROR] ROS Bridge failed: {builder.error}")

    def _setup_keyboard(self):
        """Setup keyboard input for manual fallback trigger."""
        try:
            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()

            # Subscribe with weakref to allow garbage collection
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                self._keyboard,
                lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
            )
            print("[ENV] ✓ Keyboard input enabled (Press 'O' to trigger fallback)")
        except Exception as e:
            print(f"[WARN] Keyboard setup failed: {e}")
            self._keyboard_sub = None

    def _on_keyboard_event(self, event):
        """Handle keyboard events for manual control."""
        try:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if event.input.name == "O":
                    # Trigger fallback if not already active
                    if not self.state.fallback.active:
                        print("[KEYBOARD] ✓ Fallback triggered via 'O' key - returning to initial position")
                        self.state.fallback.active = True
                        self.state.fallback.init_step = self.state.step_count
                        self.state.vlm.tracking_mode = False
                        self.state.vlm.consecutive_failures = 0
                    else:
                        print("[KEYBOARD] Fallback already active")

                elif event.input.name == "ESCAPE":
                    # Optional: provide feedback for ESC key
                    print("[KEYBOARD] ESC pressed")

        except Exception as e:
            print(f"[KEYBOARD ERROR] {e}")

        return True  # Consume event

    def step(self, action):
        """
        Main step loop.

        Args:
            action: Action tensor (not used in VLM control)

        Returns:
            (observations, reward, done, truncated, info)
        """
        self.state.step_count += 1

        # Pre-physics logic (VLM processing, target calculation)
        self._pre_physics_step(action)

        # Apply control (single application point)
        self._apply_control()

        # Physics step
        self.scene.write_data_to_sim()
        should_render = self.state.step_count % self.cfg.sim.render_interval == 0
        self.sim.step(render=should_render)
        self.scene.update(dt=self.sim.get_physics_dt())

        return self._get_observations(), 0.0, False, False, {}

    def reset(self, seed=None, options=None):
        """
        Reset environment.

        Args:
            seed: Random seed
            options: Reset options

        Returns:
            (observations, info)
        """
        if seed is not None:
            np.random.seed(seed)

        self._reset_idx(None)
        self.state.reset()

        # Clear async caches for all VLM skills
        if self.vlm and self.vlm.available:
            for skill in [self.vlm.skills.search, self.vlm.skills.position, self.vlm.skills.attitude]:
                skill.clear_async_state()

        return self._get_observations(), {}

    def _apply_control(self):
        """
        Single point for applying joint targets.
        Replaces custom _apply_control_once() from original code.
        """
        if self.state.fallback.active and self.state.fallback.target_config is not None:
            # Fallback mode: return to safe configuration
            self.robot_interface.apply_arm_target(self.state.fallback.target_config, lock_base=True)
        elif self.state.control.pending_arm_target is not None:
            # Normal mode: apply computed target
            self.robot_interface.apply_arm_target(self.state.control.pending_arm_target, lock_base=True)
        else:
            # Hold position: maintain current configuration
            current_config = self.robot_interface.get_current_arm_config()
            self.robot_interface.apply_arm_target(current_config, lock_base=True)

    def _get_observations(self) -> dict:
        """
        Get observations (dummy for now, will be replaced with proper ObservationManager).

        Returns:
            Observation dictionary
        """
        return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

    def _reset_idx(self, env_ids):
        """
        Reset specific environments.

        Args:
            env_ids: Environment indices to reset (None = all)
        """
        super()._reset_idx(env_ids)

        # Reset robot to initial configuration
        self.robot_interface.reset_to_initial()

        # Reset state
        self.state.reset()
        self.state.first_reset_done = True

        # Set pending target after reset
        self.state.control.pending_arm_target = self.robot_interface.initial_arm_config.clone()

    @abstractmethod
    def _pre_physics_step(self, actions):
        """
        Pre-physics step logic (VLM processing, target calculation).
        Must be implemented by subclasses.

        Args:
            actions: Action tensor
        """
        raise NotImplementedError

    @abstractmethod
    def _process_vlm_result(self, response):
        """
        Process VLM response.
        Must be implemented by subclasses.

        Args:
            response: VLM response dictionary
        """
        raise NotImplementedError

    def close(self):
        """
        Graceful shutdown with proper resource cleanup.
        Fixes memory leak issues from original implementation.
        """
        print("[ENV] Shutting down...")

        # 0. Cleanup keyboard subscription
        if hasattr(self, '_keyboard_sub') and self._keyboard_sub:
            try:
                self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
                self._keyboard_sub = None
                print("[ENV] ✓ Keyboard subscription cleaned up")
            except Exception as e:
                print(f"[WARN] Keyboard cleanup failed: {e}")

        # 1. Stop VLM first (with timeout)
        if hasattr(self, "vlm") and self.vlm:
            try:
                self.vlm.unload_model()
                print("[ENV] ✓ VLM cleaned up")
            except Exception as e:
                print(f"[WARN] VLM cleanup failed: {e}")

        # 2. Stop motion planner
        if hasattr(self, "motion_planner") and self.motion_planner:
            try:
                # CuRobo cleanup if needed
                del self.motion_planner
            except Exception as e:
                print(f"[WARN] Motion planner cleanup failed: {e}")

        # 3. Close parent (Isaac Lab cleanup)
        try:
            super().close()
            print("[ENV] ✓ Isaac Lab environment closed")
        except Exception as e:
            print(f"[ERROR] Environment close failed: {e}")

        print("[ENV] Shutdown complete")
