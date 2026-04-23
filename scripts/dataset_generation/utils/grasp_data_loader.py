"""
Grasp Data Loader Utility

This module provides functionality to load point cloud data from the raw_dataset
and provide grasp points and normals for the grasping environment.

The loader reads the collected point cloud data (coordinates + normals) and
distributes them across multiple environments for parallel grasping.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import isaaclab.utils.math as math_utils


class GraspDataLoader:
    """
    Loads point cloud data from raw_dataset and provides grasp points/normals.

    This class manages loading view data and distributing grasp points across
    multiple parallel environments for grasping experiments.
    """

    def __init__(
        self, dataset_dir: str = "raw_dataset", device: str = "cuda:0", view_id: int = 0
    ):
        """
        Initialize the grasp data loader.

        For testing phase, loads only a single view (default: view_000).

        Args:
            dataset_dir (str): Path to the raw dataset directory
            device (str): Device to load tensors on ('cuda:0' or 'cpu')
            view_id (int): Specific view ID to load (default: 0 for view_000)
        """
        self.dataset_dir = Path(dataset_dir)
        self.device = device

        # Storage for current view data
        self.current_view_id = None
        self.current_data = None
        self.pc_coordinates = None
        self.pc_normals = None
        self.asset_com = None

        # For testing phase: Load specific view only
        print(f"[GraspDataLoader] Testing mode: Loading view_{view_id:03d} only")
        self.load_view(view_id)

    def load_view(self, view_id: int) -> Dict:
        """
        Load a specific view from the dataset.

        Args:
            view_id (int): View ID to load

        Returns:
            dict: Loaded view data
        """
        view_dir = self.dataset_dir / f"view_{view_id:03d}"
        json_path = view_dir / "data.json"

        if not json_path.exists():
            raise FileNotFoundError(f"View {view_id} not found at {json_path}")

        # Load JSON data
        with open(json_path, "r") as f:
            data = json.load(f)

        # Convert to tensors
        self.pc_coordinates = torch.tensor(
            data["pc_coordinates"], dtype=torch.float32, device=self.device
        )
        self.pc_normals = torch.tensor(
            data["pc_normal_map"], dtype=torch.float32, device=self.device
        )
        self.asset_com = torch.tensor(
            data["asset_CoM"], dtype=torch.float32, device=self.device
        )

        self.current_view_id = view_id
        self.current_data = data

        print(f"[GraspDataLoader] Loaded view {view_id}: {data['n_points']} points")

        return data

    def get_grasp_poses_for_envs(
        self, num_envs: int, selection_mode: str = "random", grasp_offset: float = 0.05
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get grasp poses (point + normal) for multiple parallel environments.

        Each environment gets one grasp point from the loaded point cloud.

        Args:
            num_envs (int): Number of environments (grippers)
            selection_mode (str): How to select points:
                - "random": Random selection with replacement
                - "sequential": First N points in order
                - "uniform": Uniformly distributed across point cloud
            grasp_offset (float): Offset distance along normal for approach pose

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - grasp_points: (num_envs, 3) - grasp point coordinates
                - grasp_normals: (num_envs, 3) - surface normals at grasp points
                - approach_points: (num_envs, 3) - approach points offset along normals
        """
        if self.pc_coordinates is None:
            raise RuntimeError("No view loaded. Call load_view() first.")

        n_points = len(self.pc_coordinates)

        if n_points == 0:
            raise ValueError("Loaded view has no valid points")

        # Select indices based on mode
        if selection_mode == "random":
            indices = torch.randint(0, n_points, (num_envs,), device=self.device)

        elif selection_mode == "sequential":
            # Take first num_envs points, wrap if needed
            indices = torch.arange(num_envs, device=self.device) % n_points

        elif selection_mode == "uniform":
            # Uniformly distribute across point cloud
            step = max(1, n_points // num_envs)
            indices = (
                torch.arange(0, num_envs * step, step, device=self.device)[:num_envs]
                % n_points
            )

        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode}")

        # Get grasp points and normals
        grasp_points = self.pc_coordinates[indices]
        grasp_normals = self.pc_normals[indices]

        # Normalize normals to ensure unit vectors
        grasp_normals = math_utils.normalize(grasp_normals)

        # Compute approach points (offset along normal)
        approach_points = grasp_points + grasp_offset * grasp_normals

        return grasp_points, grasp_normals, approach_points

    def get_single_grasp_pose(
        self, point_index: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single grasp pose (point + normal).

        Args:
            point_index (int, optional): Specific point index. If None, random point.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - grasp_point: (3,) - grasp point coordinates
                - grasp_normal: (3,) - surface normal at grasp point
        """
        if self.pc_coordinates is None:
            raise RuntimeError("No view loaded. Call load_view() first.")

        n_points = len(self.pc_coordinates)

        if point_index is None:
            point_index = random.randint(0, n_points - 1)
        else:
            point_index = point_index % n_points  # Wrap if out of bounds

        grasp_point = self.pc_coordinates[point_index]
        grasp_normal = self.pc_normals[point_index]

        # Normalize normal
        grasp_normal = math_utils.normalize(grasp_normal.unsqueeze(0)).squeeze(0)

        return grasp_point, grasp_normal

    def get_asset_com(self) -> torch.Tensor:
        """
        Get the asset's center of mass from loaded view.

        Returns:
            torch.Tensor: (3,) - Center of mass coordinates
        """
        if self.asset_com is None:
            raise RuntimeError("No view loaded. Call load_view() first.")

        return self.asset_com

    def get_dataset_info(self) -> Dict:
        """
        Get information about the current loaded view.

        Returns:
            dict: View information including n_points, view_id, etc.
        """
        if self.current_data is None:
            return {"loaded": False, "view_id": None}

        return {
            "loaded": True,
            "view_id": self.current_view_id,
            "n_points": self.current_data["n_points"],
            "asset_com": self.current_data["asset_CoM"],
            "timestamp": self.current_data.get("timestamp", "unknown"),
        }

    def sample_grasp_batch(
        self, batch_size: int, selection_mode: str = "random"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of grasp points and normals.

        Useful for training or batch processing.

        Args:
            batch_size (int): Number of samples
            selection_mode (str): Selection mode (random, sequential, uniform)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - points: (batch_size, 3)
                - normals: (batch_size, 3)
        """
        points, normals, _ = self.get_grasp_poses_for_envs(
            num_envs=batch_size, selection_mode=selection_mode
        )
        return points, normals


def compute_grasp_orientation_from_normal(
    normal: torch.Tensor, up_axis: str = "z"
) -> torch.Tensor:
    """
    Compute gripper orientation quaternion from surface normal.

    The gripper's approach direction is aligned with the negative normal
    (approaching from outside the object).

    Args:
        normal (torch.Tensor): Surface normal vector, shape (3,) or (N, 3)
        up_axis (str): Global up direction ("z" or "y")

    Returns:
        torch.Tensor: Orientation quaternion [w, x, y, z], shape (4,) or (N, 4)
    """
    # Ensure batch dimension
    is_single = normal.ndim == 1
    if is_single:
        normal = normal.unsqueeze(0)

    device = normal.device
    batch_size = normal.shape[0]

    # Normalize normal
    normal = math_utils.normalize(normal)

    # Approach direction is opposite to normal (gripper approaches from outside)
    z_axis = -normal

    # Define global up vector
    if up_axis == "z":
        global_up = torch.tensor([0.0, 0.0, 1.0], device=device).repeat(batch_size, 1)
    else:
        global_up = torch.tensor([0.0, 1.0, 0.0], device=device).repeat(batch_size, 1)

    # When z_axis ≈ ±global_up, cross(global_up, z_axis) ≈ zero — use fallback axis
    if up_axis == "z":
        fallback = torch.tensor([0.0, 1.0, 0.0], device=device).repeat(batch_size, 1)
    else:
        fallback = torch.tensor([1.0, 0.0, 0.0], device=device).repeat(batch_size, 1)
    dot = (z_axis * global_up).sum(dim=-1, keepdim=True).abs()  # (N, 1)
    up_vec = torch.where(dot > 0.99, fallback, global_up)  # (N, 3)

    # Compute x-axis (right vector): x = up × z, then normalise
    x_axis = math_utils.normalize(torch.cross(up_vec, z_axis, dim=1))

    # Compute y-axis (actual up): y = z × x, then normalise
    y_axis = math_utils.normalize(torch.cross(z_axis, x_axis, dim=1))

    # Build rotation matrix: [x, y, z] as columns — shape (N, 3, 3)
    rot_matrix = torch.stack([x_axis, y_axis, z_axis], dim=-1)

    # Convert to quaternion (w, x, y, z) using Isaac Lab's numerically stable implementation
    quat = math_utils.quat_from_matrix(rot_matrix)

    if is_single:
        quat = quat.squeeze(0)

    return quat


if __name__ == "__main__":
    # Example usage - Testing Mode (single view)
    print("Grasp Data Loader - Testing Mode (view_000 only)\n")

    # Initialize loader - loads view_000 by default
    loader = GraspDataLoader(dataset_dir="raw_dataset", view_id=0)

    # Get dataset info
    info = loader.get_dataset_info()
    print(f"Dataset Info:")
    print(f"  View ID: {info['view_id']}")
    print(f"  Points: {info['n_points']}")
    print(f"  Asset CoM: {info['asset_com']}\n")

    # Get grasp poses for 16 environments
    num_envs = 16
    grasp_points, grasp_normals, approach_points = loader.get_grasp_poses_for_envs(
        num_envs=num_envs, selection_mode="random", grasp_offset=0.05
    )

    print(f"Generated grasp poses for {num_envs} environments:")
    print(f"  Grasp points shape: {grasp_points.shape}")
    print(f"  Grasp normals shape: {grasp_normals.shape}")
    print(f"  Approach points shape: {approach_points.shape}")
    print(f"\nFirst grasp pose:")
    print(f"  Point: {grasp_points[0]}")
    print(f"  Normal: {grasp_normals[0]}")
    print(f"  Approach: {approach_points[0]}")
