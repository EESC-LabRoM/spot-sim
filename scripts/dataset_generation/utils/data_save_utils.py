"""
Data saving utilities for raw point cloud dataset extraction.

This module provides functionality to save point cloud observations including:
- Point coordinates (world frame)
- Normal maps (world frame) - computed using existing point_cloud_utils
- Center of Mass (world frame)
- Point cloud visualizations (PNG)

Data is saved in JSON format (metadata) and PNG format (visualizations).
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt
from datetime import datetime

# Import existing utilities
from .visualization_utils import Colors

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print(
        f"{Colors.YELLOW}[IMPORT WARNING] Open3D not available. Normal computation will be disabled.{Colors.END}"
    )


def compute_all_point_normals(
    points: torch.Tensor, search_radius: float = 0.05, max_nn: int = 30
) -> torch.Tensor:
    """
    Compute surface normals for all points in a point cloud using Open3D.

    This function wraps the Open3D normal estimation to compute normals for
    ALL points at once, maintaining the same order as the input points.

    Args:
        points (torch.Tensor): Point cloud of shape (N, 3)
        search_radius (float): Radius for neighbor search. Default: 0.05
        max_nn (int): Maximum number of neighbors. Default: 30

    Returns:
        torch.Tensor: Normal vectors of shape (N, 3) in the SAME ORDER as input points.
                     Returns zeros for invalid points (0, 0, 0).
    """
    if not OPEN3D_AVAILABLE:
        print("[WARNING] Open3D not available. Returning zero normals.")
        return torch.zeros_like(points)

    # Convert to numpy
    if isinstance(points, torch.Tensor):
        points_np = points.detach().cpu().numpy()
        device = points.device
    else:
        points_np = np.asarray(points)
        device = torch.device("cpu")

    # Filter valid points
    valid_mask = np.any(points_np != 0, axis=1)
    valid_points = points_np[valid_mask]

    if len(valid_points) == 0:
        return torch.zeros_like(points)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)

    # Estimate normals - this computes normals for all points at once
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=search_radius, max_nn=max_nn
        )
    )

    # Extract normals - these are in the SAME ORDER as valid_points
    normals_valid = np.asarray(pcd.normals)

    # Create full normal array (zeros for invalid points)
    # This preserves the original point order
    normals_full = np.zeros_like(points_np)
    normals_full[valid_mask] = normals_valid

    # Convert back to tensor
    return torch.from_numpy(normals_full).to(device)


def compute_all_point_normals_batch(
    points_batch: torch.Tensor, search_radius: float = 0.05, max_nn: int = 30
) -> torch.Tensor:
    """
    Compute surface normals for a batch of point clouds.

    Each point cloud's normals are computed independently, maintaining point order.

    Args:
        points_batch (torch.Tensor): Batch of point clouds, shape (B, N, 3)
        search_radius (float): Radius for neighbor search. Default: 0.05
        max_nn (int): Maximum number of neighbors. Default: 30

    Returns:
        torch.Tensor: Batch of normal vectors, shape (B, N, 3).
                     Normals are in the SAME ORDER as input points for each batch.
    """
    batch_size = points_batch.shape[0]
    normals_batch = []

    for i in range(batch_size):
        normals = compute_all_point_normals(
            points_batch[i], search_radius=search_radius, max_nn=max_nn
        )
        normals_batch.append(normals)

    return torch.stack(normals_batch, dim=0)


def save_point_cloud_visualization(
    points: np.ndarray,
    save_path: Path,
    title: str = "Point Cloud View",
    camera_pos: Optional[np.ndarray] = None,
    camera_lookat: Optional[np.ndarray] = None,
):
    """
    Save a point cloud visualization projected onto the camera image plane.

    Projects 3D points into 2D using the camera view frame, coloured by
    depth (distance from camera). When camera pose is not provided falls back
    to a default front-facing viewpoint.

    Args:
        points (np.ndarray): Point cloud array of shape (N, 3), env-local frame.
        save_path (Path): Path to save the PNG file.
        title (str): Title for the visualization.
        camera_pos (np.ndarray, optional): Camera eye position, shape (3,).
        camera_lookat (np.ndarray, optional): Camera look-at point, shape (3,).
    """
    valid_mask = np.any(points != 0, axis=1)
    valid_points = points[valid_mask]

    if len(valid_points) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, "No valid points", ha="center", va="center", fontsize=20)
        ax.axis("off")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    # Build camera view axes
    if camera_pos is not None and camera_lookat is not None:
        cam_pos = np.asarray(camera_pos, dtype=np.float64)
        cam_target = np.asarray(camera_lookat, dtype=np.float64)
    else:
        cam_target = valid_points.mean(axis=0)
        cam_pos = cam_target + np.array([1.0, 0.0, 0.0])

    forward = cam_target - cam_pos
    forward /= np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # Project points into camera frame
    pts_rel = valid_points - cam_pos
    depth = pts_rel @ forward
    proj_right = pts_rel @ right
    proj_up = pts_rel @ up

    in_front = depth > 0
    depth = depth[in_front]
    proj_right = proj_right[in_front]
    proj_up = proj_up[in_front]

    if len(depth) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(
            0.5, 0.5, "All points behind camera", ha="center", va="center", fontsize=16
        )
        ax.axis("off")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        proj_right,
        proj_up,
        c=depth,
        cmap="plasma_r",
        s=1,
        alpha=0.7,
    )
    plt.colorbar(scatter, ax=ax, label="Depth (m)")
    ax.set_xlabel("Right →")
    ax.set_ylabel("Up ↑")
    ax.set_aspect("equal")
    ax.set_title(f"{title}\n{len(depth)} points", fontsize=11, fontweight="bold")
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


class RawDatasetSaver:
    """
    Handles saving of raw point cloud data for grasping dataset generation.

    Directory structure:
        raw_dataset/
            |___view_000/
                    |___data.json
                    |___pc_view.png
            |___view_001/
                    |___data.json
                    |___pc_view.png
            ...

    [!]: Points and normals are saved in the same order.
    - pc_coordinates[i] corresponds to pc_normal_map[i]
    """

    def __init__(self, base_dir: str = "raw_dataset", auto_create: bool = True):
        """
        Initialize the dataset saver.

        Args:
            base_dir (str): Base directory for saving the dataset
            auto_create (bool): Automatically create base directory if it doesn't exist
        """
        self.base_dir = Path(base_dir)
        self.view_counter = 0

        if auto_create:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            print(f"[DatasetSaver] Initialized at: {self.base_dir.absolute()}")

    def save_view_data(
        self,
        pc_coordinates: torch.Tensor,
        pc_normal_map: torch.Tensor,
        asset_com: torch.Tensor,
        view_id: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Save a complete view of point cloud data.

        Args:
            pc_coordinates (torch.Tensor): Point cloud coordinates in world frame, shape (N, 3)
            pc_normal_map (torch.Tensor): Normal vectors in world frame, shape (N, 3)
                                         MUST be in same order as pc_coordinates
            asset_com (torch.Tensor): Center of mass in world frame, shape (3,)
            view_id (int, optional): Custom view ID, otherwise auto-incremented
            metadata (dict, optional): Additional metadata to include

        Returns:
            Path: Path to the created view directory
        """
        # Use provided view_id or auto-increment
        if view_id is None:
            view_id = self.view_counter
            self.view_counter += 1

        # Create view directory
        view_dir = self.base_dir / f"view_{view_id:03d}"
        view_dir.mkdir(parents=True, exist_ok=True)

        # Convert tensors to numpy for serialization
        pc_coords_np = self._to_numpy(pc_coordinates)
        pc_normals_np = self._to_numpy(pc_normal_map)
        asset_com_np = self._to_numpy(asset_com)

        # Filter out zero/invalid points (preserving order)
        valid_mask = np.any(pc_coords_np != 0, axis=1)
        pc_coords_valid = pc_coords_np[valid_mask]
        pc_normals_valid = pc_normals_np[valid_mask]

        n_points = len(pc_coords_valid)

        # Prepare JSON data
        json_data = {
            "view": view_id,
            "n_points": int(n_points),
            "asset_CoM": asset_com_np.tolist(),
            "pc_coordinates": pc_coords_valid.tolist(),
            "pc_normal_map": pc_normals_valid.tolist(),
            "timestamp": datetime.now().isoformat(),
        }

        # Add optional metadata
        if metadata is not None:
            json_data["metadata"] = metadata

        # Save JSON
        json_path = view_dir / "data.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        # Save point cloud visualization
        png_path = view_dir / "pc_view.png"
        cam_pos = (
            np.array(metadata["camera_pos_local"])
            if metadata and "camera_pos_local" in metadata
            else None
        )
        cam_lookat = (
            np.array(metadata["camera_lookat_local"])
            if metadata and "camera_lookat_local" in metadata
            else None
        )
        save_point_cloud_visualization(
            pc_coords_np,
            png_path,
            title=f"Point Cloud View {view_id:03d}",
            camera_pos=cam_pos,
            camera_lookat=cam_lookat,
        )

        print(
            f"[DatasetSaver] Saved view {view_id:03d} ({n_points} points) to {view_dir}"
        )

        return view_dir

    def save_batch_view_data(
        self,
        pc_coordinates_batch: torch.Tensor,
        pc_normal_map_batch: torch.Tensor,
        asset_com_batch: torch.Tensor,
        start_view_id: Optional[int] = None,
        metadata_batch: Optional[List[Dict]] = None,
    ) -> List[Path]:
        """
        Save multiple views from batched data.

        Args:
            pc_coordinates_batch (torch.Tensor): Batch of point clouds, shape (B, N, 3)
            pc_normal_map_batch (torch.Tensor): Batch of normals, shape (B, N, 3)
                                                MUST be in same order as coordinates
            asset_com_batch (torch.Tensor): Batch of CoMs, shape (B, 3)
            start_view_id (int, optional): Starting view ID
            metadata_batch (list, optional): List of metadata dicts for each view

        Returns:
            list: List of paths to created view directories
        """
        batch_size = pc_coordinates_batch.shape[0]
        saved_paths = []

        if start_view_id is None:
            start_view_id = self.view_counter

        for i in range(batch_size):
            pc_coords = pc_coordinates_batch[i]
            pc_normals = pc_normal_map_batch[i]
            asset_com = asset_com_batch[i]

            metadata = None
            if metadata_batch is not None and i < len(metadata_batch):
                metadata = metadata_batch[i]

            view_path = self.save_view_data(
                pc_coordinates=pc_coords,
                pc_normal_map=pc_normals,
                asset_com=asset_com,
                view_id=start_view_id + i,
                metadata=metadata,
            )
            saved_paths.append(view_path)

        self.view_counter = start_view_id + batch_size

        return saved_paths

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    def load_view_data(self, view_id: int) -> Dict:
        """
        Load data from a specific view.

        Args:
            view_id (int): View ID to load

        Returns:
            dict: Loaded data including coordinates, normals (in same order), CoM
        """
        view_dir = self.base_dir / f"view_{view_id:03d}"
        json_path = view_dir / "data.json"

        if not json_path.exists():
            raise FileNotFoundError(f"View {view_id:03d} not found at {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        return data

    def get_dataset_summary(self) -> Dict:
        """
        Get summary statistics of the saved dataset.

        Returns:
            dict: Summary including total views, total points, etc.
        """
        view_dirs = sorted(self.base_dir.glob("view_*"))
        n_views = len(view_dirs)

        if n_views == 0:
            return {
                "total_views": 0,
                "total_points": 0,
                "dataset_path": str(self.base_dir.absolute()),
            }

        total_points = 0
        for view_dir in view_dirs:
            json_path = view_dir / "data.json"
            if json_path.exists():
                with open(json_path, "r") as f:
                    data = json.load(f)
                    total_points += data.get("n_points", 0)

        return {
            "total_views": n_views,
            "total_points": total_points,
            "avg_points_per_view": total_points / n_views if n_views > 0 else 0,
            "dataset_path": str(self.base_dir.absolute()),
        }

    def reset_counter(self):
        """Reset the view counter to 0."""
        self.view_counter = 0
        print("[DatasetSaver] View counter reset to 0")


# Convenience function
def save_point_cloud_with_normals(
    pc_coordinates: torch.Tensor,
    pc_normals: torch.Tensor,
    asset_com: torch.Tensor,
    save_dir: str = "raw_dataset",
    view_id: Optional[int] = None,
) -> Path:
    """
    Convenience function to save point cloud data.

    Args:
        pc_coordinates (torch.Tensor): Point cloud coordinates (N, 3)
        pc_normals (torch.Tensor): Normal vectors (N, 3) - SAME ORDER as coordinates
        asset_com (torch.Tensor): Center of mass (3,)
        save_dir (str): Directory to save data
        view_id (int, optional): View identifier

    Returns:
        Path: Path to saved view directory
    """
    saver = RawDatasetSaver(base_dir=save_dir)
    return saver.save_view_data(
        pc_coordinates=pc_coordinates,
        pc_normal_map=pc_normals,
        asset_com=asset_com,
        view_id=view_id,
    )


if __name__ == "__main__":
    print("Data save utilities module loaded successfully.")
    print("Key functions:")
    print("  - compute_all_point_normals(): Compute normals for entire point cloud")
    print("  - compute_all_point_normals_batch(): Batch version")
    print("  - RawDatasetSaver: Main class for saving datasets")
    print("\nNOTE: Points and normals are always saved in the SAME ORDER!")
