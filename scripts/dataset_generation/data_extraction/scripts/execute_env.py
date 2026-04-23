import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Spherical multi-view point-cloud dataset extraction."
)
parser.add_argument(
    "--n_azimuth", type=int, default=8, help="Azimuth steps per elevation ring."
)
parser.add_argument(
    "--n_elevation", type=int, default=3, help="Number of elevation rings."
)
parser.add_argument("--radius", type=float, default=0.6, help="Orbit radius in metres.")
parser.add_argument(
    "--min_elevation", type=float, default=10.0, help="Min elevation angle (deg)."
)
parser.add_argument(
    "--max_elevation", type=float, default=70.0, help="Max elevation angle (deg)."
)
parser.add_argument(
    "--output_dir", type=str, default="raw_dataset", help="Dataset output directory."
)
parser.add_argument(
    "--normal_radius", type=float, default=0.05, help="Normal estimation search radius."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import os
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from dataset_generation.data_extraction.data_env import RawDataExtractionEnvCfg
from dataset_generation.utils.camera_utils import (
    set_camera_pose,
    build_hemisphere_views,
)
from dataset_generation.utils.point_cloud_utils import object_point_cloud
from dataset_generation.utils.data_save_utils import (
    RawDatasetSaver,
    compute_all_point_normals_batch,
)


def _build_views(center: list, args) -> list:
    """Build the hemisphere view schedule centred on *center* (env-local)."""
    return build_hemisphere_views(
        center=center,
        radius=args.radius,
        n_azimuth=args.n_azimuth,
        n_elevation=args.n_elevation,
        min_elevation_deg=args.min_elevation,
        max_elevation_deg=args.max_elevation,
    )


def _place_cameras(env: ManagerBasedEnv, views: list) -> None:
    """Move every env's camera to its assigned viewpoint in one batched call."""
    positions = torch.tensor([p for p, _ in views], dtype=torch.float32)
    lookats = torch.tensor([l for _, l in views], dtype=torch.float32)

    set_camera_pose(
        camera=env.scene["camera"],
        pos=positions,
        lookat=lookats,
        env_origins=env.scene.env_origins,
        sim=env.sim,
    )


def _collect_and_save(
    env: ManagerBasedEnv,
    views: list,
    saver: RawDatasetSaver,
    normal_radius: float,
) -> None:
    """Extract point clouds from all envs and save one view file per env."""
    pc_batch = object_point_cloud(
        env=env,
        sensor_cfg=SceneEntityCfg("camera"),
        debug=False,
        visualize=False,
        visualization_counter={
            "count": 1,
            "last_sim_step": -1,
            "visualized_steps": set(),
        },
    )

    asset = env.scene["target_object"]
    com_batch = asset.data.root_com_pos_w - env.scene.env_origins  # (N, 3) env-local

    normals_batch = compute_all_point_normals_batch(
        pc_batch, search_radius=normal_radius, max_nn=30
    )

    metadata_batch = [
        {
            "view_id": i,
            "camera_pos_local": list(views[i][0]),
            "camera_lookat_local": list(views[i][1]),
            "radius": args_cli.radius,
            "normal_search_radius": normal_radius,
        }
        for i in range(env.num_envs)
    ]

    saved_paths = saver.save_batch_view_data(
        pc_coordinates_batch=pc_batch,
        pc_normal_map_batch=normals_batch,
        asset_com_batch=com_batch,
        start_view_id=0,
        metadata_batch=metadata_batch,
    )

    print(f"[INFO] Saved {len(saved_paths)} views → {saver.base_dir}")


# TODO: debug why some view.png hasn't point clouds -> why the hell they're not appearing in the plot?
def main():
    # 1. Compute view schedule → num_envs = num_views
    # Placeholder centre at origin; recomputed after the scene settles.
    views = _build_views(center=[1.0, 0.0, 0.5], args=args_cli)
    num_views = len(views)
    print(
        f"[INFO] Spherical grid: {num_views} views  "
        f"({args_cli.n_azimuth} az × {args_cli.n_elevation} el, "
        f"r={args_cli.radius} m)"
    )

    # 2. Build env — one environment per view
    env_cfg = RawDataExtractionEnvCfg()
    env_cfg.scene.num_envs = num_views
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedEnv(cfg=env_cfg)

    # 3. Reset + settle step so the object CoM is stable
    zero_action = torch.zeros(
        env.num_envs, env.action_manager.total_action_dim, device=env.device
    )
    env.reset()
    with torch.inference_mode():
        env.step(zero_action)

    # Re-build views centred on the actual object CoM (env-local frame,
    # identical across all envs since the object spawns at the same local pose)
    com_local = (
        env.scene["target_object"].data.root_com_pos_w[0] - env.scene.env_origins[0]
    ).tolist()
    views = _build_views(center=com_local, args=args_cli)
    print(f"[INFO] Object CoM (env-local): {[round(v, 4) for v in com_local]}")

    # 4. Place all N cameras in one batched call
    _place_cameras(env, views)

    # 5. Single step → all N cameras render simultaneously
    with torch.inference_mode():
        env.step(zero_action)

    # 6. Save all N views
    saver = RawDatasetSaver(base_dir=args_cli.output_dir)
    with torch.inference_mode():
        _collect_and_save(env, views, saver, normal_radius=args_cli.normal_radius)

    summary = saver.get_dataset_summary()
    print(
        f"[INFO] Done — {summary['total_views']} views, "
        f"{summary['total_points']} total points"
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
