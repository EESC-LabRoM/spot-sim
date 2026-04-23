"""
Main script for running Spot Isaac Lab tracking tasks.
Supports:
1. position-tracking: Position only (XYZ)
2. pose-tracking: Position + Attitude (6D Pose)

Refactored from spot_vlm with proper architecture and bug fixes.
"""

import argparse
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6 8.7 9.0+PTX"  # Ajuste conforme necessário para sua GPU

# --------------------------------PATCH INSPECT--------------------------------#
import inspect

# Save original function
_old_getfile = inspect.getfile


def _safe_getfile(object):
    """Patch to avoid crash with IsaacLab/Torch Package"""
    try:
        return _old_getfile(object)
    except TypeError:
        # If it fails saying it's built-in, return a safe fake name
        return "<isaaclab_module>"


# Apply patch
inspect.getfile = _safe_getfile
# -----------------------------END PATCH INSPECT------------------------------#

# Isaac Lab imports MUST come BEFORE
from isaaclab.app import AppLauncher

# 1. Configure Arguments (Before launching the App)
parser = argparse.ArgumentParser(description="Spot Isaac Lab Task Runner")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument(
    "--task",
    type=str,
    default="position-tracking",
    choices=["position-tracking", "pose-tracking"],
    help="Task name: 'position-tracking' (Position only) or 'pose-tracking' (Pos + Orientation).",
)
parser.add_argument("--video", action="store_true", default=False, help="Record video of the episode.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")

# Add standard Isaac Lab arguments (headless, livestreams, etc)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force enable_cameras to True, as the environment needs cameras
args_cli.enable_cameras = True

# 2. Start Simulation (Must be done before importing torch/isaaclab)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- FORCE LOAD EXTENSIONS (FIX) ---
# Use the Extension Manager directly to avoid ModuleNotFoundError
import omni.kit.app

manager = omni.kit.app.get_app().get_extension_manager()

# 1. Enable Core Nodes (Fixes: unrecognized type 'IsaacReadSimulationTime')
manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

# 2. Enable ROS 2 Bridge (Fixes: ROS topics not appearing)
manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
# -----------------------------------

# 3. System Imports (Post-Launch)
import torch
import traceback
import gymnasium as gym
from datetime import datetime
import os
import sys
from pathlib import Path

# --- PATH CORRECTION ---
# Add project root to sys.path to allow imports from 'scripts.spot_isaac'
# Assumes play.py is in .../scripts/spot_isaac/play.py
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]  # Go up 2 levels: spot_isaac -> scripts -> root
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
# ------------------------

from scripts.spot_isaaclab.configs import global_config as config

# Environment imports
try:
    from scripts.spot_isaaclab.envs.position_tracking_env import PositionTrackingEnv
    from scripts.spot_isaaclab.envs.pose_tracking_env import PoseTrackingEnv
    from scripts.spot_isaaclab.configs.env_cfg import PositionTrackingEnvCfg, PoseTrackingEnvCfg
except ImportError as e:
    print(f"\n[IMPORT ERROR] Could not import environments.")
    print(f"Make sure 'scripts.spot_isaaclab' is in PYTHONPATH.")
    print(f"Original error: {e}\n")
    # Try relative import as fallback
    try:
        print("[INFO] Trying relative import...")
        from envs.position_tracking_env import PositionTrackingEnv
        from envs.pose_tracking_env import PoseTrackingEnv
        from configs.env_cfg import PositionTrackingEnvCfg, PoseTrackingEnvCfg
        print("[INFO] Relative import worked!")
    except ImportError as e2:
        print(f"[FATAL ERROR] Relative import also failed: {e2}")
        raise e


def main():
    print(f"\n[INFO] Initializing task: {args_cli.task.upper()}")
    print(f"[INFO] VLM Enabled: {config.VLM_ENABLED}")

    # Environment selection
    if args_cli.task == "pose-tracking":
        # 6DOF pose tracking environment
        env_cfg = PoseTrackingEnvCfg()
        env_cfg = config_env(env_cfg)
        env = PoseTrackingEnv(cfg=env_cfg)
        print("[INFO] Environment loaded: PoseTrackingEnv (6D Control)")

    elif args_cli.task == "position-tracking":
        # 3D position tracking environment
        env_cfg = PositionTrackingEnvCfg()
        env_cfg = config_env(env_cfg)
        env = PositionTrackingEnv(cfg=env_cfg)
        print("[INFO] Environment loaded: PositionTrackingEnv (Position Only)")

    else:
        raise ValueError(f"Unknown task: {args_cli.task}")

    # --- Video Recording Configuration ---
    if args_cli.video:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_folder = os.path.join("videos", f"{args_cli.task}_{timestamp}")

        # Standard Gymnasium wrapper for recording
        # Note: IsaacLab environments usually need specific wrapper or custom render()
        # Here we use standard RecordVideo, assuming env.render() returns RGB array
        print(f"[INFO] Recording video to: {video_folder}")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,  # Record from start
            video_length=args_cli.video_length,
            name_prefix=f"spot_{args_cli.task}",
        )

    # Simulation Loop
    try:
        obs, _ = env.reset()

        while simulation_app.is_running():
            # The action sent to step here is 'dummy' because real control
            # is calculated internally by env based on VLM.
            # However, we maintain the correct format (num_envs, action_dim).

            # Action space is 6 (DOF) for both envs now
            actions = torch.zeros((env.num_envs, 6), device=env.device)

            # Step
            obs, rew, terminated, truncated, info = env.step(actions)

            # Reset if necessary (usually handled internally by ManagerBasedEnv, but good to have here)
            if terminated or truncated:
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\n[INFO] User interrupt.")
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        print("[INFO] Closing environment...")
        env.close()
        simulation_app.close()


def config_env(env_cfg_):
    """Configure environment parameters"""
    env_cfg_.scene.num_envs = args_cli.num_envs
    env_cfg_.sim.device = args_cli.device
    return env_cfg_


if __name__ == "__main__":
    main()
