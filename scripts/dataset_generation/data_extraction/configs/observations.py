"""
Observation configurations for Isaac Lab environments.

"""

import os
import sys
import torch

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg

# allows for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import utilities
from dataset_generation import Colors

from dataset_generation.utils.point_cloud_utils import (
    object_point_cloud,
    set_pca_backend,
)
from dataset_generation.utils.data_save_utils import (
    RawDatasetSaver,
    compute_all_point_normals_batch,
)

# from .scene import RawDataExtractionSceneCfg
# Set PCA backend
set_pca_backend("gpu")

# Visualization counter - tracks actual step calls
_visualization_counter = {"count": 0, "last_sim_step": -1, "visualized_steps": set()}

# Dataset saver
_dataset_saver = None


def get_CoM(env, asset_cfg: RigidObjectCfg, debug: bool = False):
    """Get center of mass (CoM) in env-local frame (world - env_origins).

    Env-local coordinates are independent of num_envs and env_spacing,
    matching the frame used when saving the point cloud dataset.
    """
    asset = env.scene[asset_cfg.name]
    asset_CoM_world = asset.data.root_com_pos_w  # (num_envs, 3)
    asset_CoM_local = asset_CoM_world - env.scene.env_origins  # (num_envs, 3)
    if debug:
        print(
            f"\n{Colors.YELLOW}[DEBUG CoM world]     {asset_CoM_world[0]}{Colors.END}"
        )
        print(f"{Colors.YELLOW}[DEBUG CoM env-local] {asset_CoM_local[0]}{Colors.END}")
    return asset_CoM_local


def get_pos(env, asset_cfg: RigidObjectCfg, debug: bool = False):
    """Get asset position in env-local frame (world - env_origins).

    Env-local coordinates are independent of num_envs and env_spacing,
    matching the frame used when saving the point cloud dataset.
    """
    asset = env.scene[asset_cfg.name]
    asset_pos_world = asset.data.root_pose_w[:, 0:3]  # (num_envs, 3)
    asset_pos_local = asset_pos_world - env.scene.env_origins  # (num_envs, 3)
    if debug:
        print(
            f"\n{Colors.YELLOW}[DEBUG POS world]     {asset_pos_world[0]}{Colors.END}"
        )
        print(f"{Colors.YELLOW}[DEBUG POS env-local] {asset_pos_local[0]}{Colors.END}")
    return asset_pos_local


def object_point_cloud_wrapper(
    env, sensor_cfg: SceneEntityCfg, debug: bool = False, visualize: bool = False
):
    """
    Wrapper function for object_point_cloud that uses the global visualization counter.

    This wrapper maintains backward compatibility with the original API while using
    the refactored utility function.

    Args:
        env: Environment object containing the scene with camera sensors.
        sensor_cfg (SceneEntityCfg): Scene entity configuration for the camera sensor.
        debug (bool, optional): Enable debug print statements. Defaults to False.
        visualize (bool, optional): Enable visualization at step 50. Defaults to False.

    Returns:
        torch.Tensor: Point cloud in env-local frame with shape (num_envs, H*W, 3).
    """
    global _visualization_counter

    current_sim_step = env._sim_step_counter
    if current_sim_step != _visualization_counter["last_sim_step"]:
        _visualization_counter["count"] += 1
        _visualization_counter["last_sim_step"] = current_sim_step

    return object_point_cloud(
        env=env,
        sensor_cfg=sensor_cfg,
        debug=debug,
        visualize=visualize,
        visualization_counter=_visualization_counter,
    )


def save_raw_data_wrapper(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    save_dir: str = "raw_dataset",
    save_interval: int = 10,
    compute_normals: bool = True,
    normal_search_radius: float = 0.05,
    debug: bool = False,
):
    """
    Wrapper to save raw point cloud data at specified intervals.

    All coordinates are saved in env-local frame (world - env.scene.env_origins),
    making the dataset independent of num_envs and env_spacing.

    This observation term automatically saves:
    - Point cloud coordinates (env-local frame)
    - Surface normals (env-local frame) - computed using point_cloud_utils
    - Asset center of mass (env-local frame)
    - Point cloud visualization (4-view PNG)

    IMPORTANT: Points and normals are saved in the SAME ORDER.
    pc_coordinates[i] corresponds to pc_normal_map[i] for all i.

    Args:
        env: Environment object
        sensor_cfg: Camera sensor configuration
        asset_cfg: Target object configuration
        save_dir: Directory to save dataset
        save_interval: Save every N environment steps
        compute_normals: Whether to compute surface normals
        normal_search_radius: Search radius for normal estimation (meters)
        debug: Print debug information

    Returns:
        torch.Tensor: Dummy tensor (this is a side-effect observation)
    """
    global _dataset_saver, _visualization_counter

    # Initialize saver on first call
    if _dataset_saver is None:
        _dataset_saver = RawDatasetSaver(base_dir=save_dir, auto_create=True)
        print(f"{Colors.CYAN}[DataSaver] Initialized at: {save_dir}{Colors.END}")

    # Update counter - this ensures we track EVERY observation call
    current_sim_step = env._sim_step_counter
    if current_sim_step != _visualization_counter["last_sim_step"]:
        _visualization_counter["count"] += 1
        _visualization_counter["last_sim_step"] = current_sim_step

    current_step = _visualization_counter["count"]

    # Check if we should save at this interval (start saving from step 1)
    should_save = (current_step > 0) and (current_step % save_interval == 0)

    if not should_save:
        if debug and current_step <= 20:
            print(
                f"{Colors.YELLOW}[DataSaver] Step {current_step}: Skipping save (interval={save_interval}){Colors.END}"
            )
        return torch.zeros(env.num_envs, 1, device=env.device)

    if debug:
        print(
            f"{Colors.GREEN}[DataSaver] Step {current_step}: Saving data...{Colors.END}"
        )

    # Get point cloud data
    pc_coordinates = object_point_cloud_wrapper(
        env, sensor_cfg, debug=False, visualize=False
    )

    # Get center of mass in env-local frame
    asset = env.scene[asset_cfg.name]
    asset_com = asset.data.root_com_pos_w - env.scene.env_origins  # (num_envs, 3)

    # Compute normals if requested using existing utilities
    if compute_normals:
        # Use the compute_all_point_normals_batch function which uses Open3D
        # This ensures normals are in the SAME ORDER as points
        pc_normals = compute_all_point_normals_batch(
            pc_coordinates, search_radius=normal_search_radius, max_nn=30
        )
    else:
        pc_normals = torch.zeros_like(pc_coordinates)

    # Prepare metadata for batch
    metadata_batch = [
        {
            "step": current_step,
            "env_id": i,
            "normal_search_radius": normal_search_radius if compute_normals else None,
        }
        for i in range(env.num_envs)
    ]

    # Save batch data
    # NOTE: The saver will create point cloud visualizations automatically
    saved_paths = _dataset_saver.save_batch_view_data(
        pc_coordinates_batch=pc_coordinates,
        pc_normal_map_batch=pc_normals,
        asset_com_batch=asset_com,
        metadata_batch=metadata_batch,
    )

    if debug:
        print(
            f"{Colors.GREEN}[DataSaver] Saved {len(saved_paths)} views at step {current_step}{Colors.END}"
        )
        summary = _dataset_saver.get_dataset_summary()
        print(
            f"{Colors.GREEN}[DataSaver] Total: {summary['total_views']} views, "
            f"{summary['total_points']} points{Colors.END}"
        )

    return torch.zeros(env.num_envs, 1, device=env.device)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        object_pc = ObsTerm(
            func=object_point_cloud_wrapper,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "debug": False,
                "visualize": False,
            },
        )

        object_com = ObsTerm(
            func=get_CoM,
            params={
                "asset_cfg": SceneEntityCfg("target_object"),
                "debug": False,
            },
        )
        object_pos = ObsTerm(
            func=get_pos,
            params={
                "asset_cfg": SceneEntityCfg("target_object"),
                "debug": False,
            },
        )

        save_raw_data = ObsTerm(
            func=save_raw_data_wrapper,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "asset_cfg": SceneEntityCfg("target_object"),
                "save_dir": "raw_dataset",
                "save_interval": 10,
                "compute_normals": True,
                "normal_search_radius": 0.05,
                "debug": True,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class ObservationsCfgMinimal:
    """Minimal configuration for speed."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Minimal observations."""

        object_com = ObsTerm(
            func=get_CoM,
            params={
                "asset_cfg": SceneEntityCfg("target_object"),
                "debug": False,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
