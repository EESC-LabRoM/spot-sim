import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Setting up a Spot Gripper environment for Grasping Dataset Extraction."
)
parser.add_argument(
    "--num_envs", type=int, default=16, help="Number of environments to spawn."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""
import os
import sys
import torch

# This allows for absolute imports from 'spot_mgrasping'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from isaaclab.envs import ManagerBasedEnv
from dataset_generation.spot_grasping.grasp_env import SpotEndEffectorEnvCfg


def main():
    """Main function."""
    # scene setup
    env_cfg = SpotEndEffectorEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # create the environment
    env = ManagerBasedEnv(cfg=env_cfg)
    print("[INFO]: Starting simulation loop with Integrated Kinematic Action...")
    # # debugging: print min/max of the depth buffer
    # depth_buffer = env.scene["camera"].data.output["distance_to_image_plane"]
    # print(
    #     f"Min Depth: {torch.min(depth_buffer):.2f}, Max Depth: {torch.max(depth_buffer):.2f}"
    # )
    action_close = torch.ones(env.num_envs, 1, device=env.device)
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            # step env to simulate physics engine
            obs, _ = env.step(action_close)

            count += 1

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
