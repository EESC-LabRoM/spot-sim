"""
Point cloud processing utilities for grasping.

This module provides geometric analysis tools for 3D point clouds, specifically
for estimating object pose and alignment using Principal Component Analysis (PCA).
Includes CPU (Open3D) and GPU-accelerated (PyTorch) implementations with flexible backends.

ENHANCED VERSION with configurable PCA backend:
- "gpu": Fast GPU implementation using torch.pca_lowrank (recommended)
- "torch": Custom PyTorch eigendecomposition (fallback)
- "open3d": CPU-based Open3D (single point cloud only)
"""

import numpy as np
import os
import sys
import torch
from typing import Tuple, Optional, Literal

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset_generation.utils import (
    visualize_point_cloud,
    visualize_camera_outputs,
    Colors,
)

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("[WARNING] Open3D not available. Using PyTorch implementations only.")


# Global backend configuration
_DEFAULT_BACKEND: Literal["gpu", "torch", "open3d"] = "gpu"


def object_point_cloud(
    env,
    sensor_cfg,
    debug=False,
    visualize=False,
    visualization_counter=None,
    debug_step_interval=50,
):
    """
    Convert depth image to world-frame point cloud with object segmentation.

    Args:
        env: Environment object containing the scene with camera sensors.
        sensor_cfg: Scene entity configuration for the camera sensor.
        debug (bool): Enable debug print statements.
        visualize (bool): Enable visualization.
        visualization_counter (dict): Counter for visualization state.
        debug_step_interval (int): Show debug info every N steps. Default: 50.

    Returns:
        torch.Tensor: Point cloud in env-local frame, shape (num_envs, H*W, 3).
                      Env-local = world - env.scene.env_origins, making coordinates
                      independent of num_envs and env_spacing.
    """
    import torch
    import isaaclab.utils.math as math_utils

    # Initialize local counter if not provided
    if visualization_counter is None:
        visualization_counter = {"count": 0, "visualized_steps": set()}

    camera = env.scene[sensor_cfg.name]

    visualization_counter["count"] += 1
    current_step = visualization_counter["count"]

    # Decide when to show debug output
    show_debug = debug and (
        current_step == 1 or current_step % debug_step_interval == 0
    )

    # Check BEFORE marking as visualized
    should_visualize = (
        visualize
        and current_step == 50
        and 50 not in visualization_counter["visualized_steps"]
    )

    # Mark as visualized ONCE at the beginning
    if should_visualize:
        visualization_counter["visualized_steps"].add(50)

    # Visualize camera outputs
    if should_visualize:
        visualize_camera_outputs(camera, current_step)

    # Safeguard
    if camera.data.output is None or len(camera.data.info) == 0:
        if debug:
            print(
                f"{Colors.YELLOW}[DEBUG Step {current_step}]: Camera data not ready yet{Colors.END}"
            )
        return torch.zeros(
            (camera.num_envs, camera.cfg.height * camera.cfg.width, 3),
            device=camera.device,
        )

    depth = camera.data.output["distance_to_image_plane"]
    segmentation = camera.data.output["semantic_segmentation"]

    # Find target object ID
    obj_id = None
    try:
        info = camera.data.info

        if isinstance(info, list) and len(info) > 0:
            info_dict = info[0]
        elif isinstance(info, dict):
            info_dict = info
        else:
            info_dict = None

        if info_dict is not None:
            if show_debug:
                print(
                    f"\n[DEBUG] ===== SEMANTIC INFO ANALYSIS (Step {current_step}) ====="
                )
                print(f"[DEBUG] Info dict keys: {list(info_dict.keys())}")

            if "semantic_segmentation" in info_dict:
                seg_info = info_dict["semantic_segmentation"]

                if show_debug:
                    print(f"[DEBUG] semantic_segmentation type: {type(seg_info)}")
                    if isinstance(seg_info, dict):
                        print(
                            f"[DEBUG] semantic_segmentation keys: {list(seg_info.keys())}"
                        )
                        for key, value in seg_info.items():
                            print(f"[DEBUG]   '{key}': {value}")
                    else:
                        print(f"[DEBUG] semantic_segmentation content: {seg_info}")

                if isinstance(seg_info, dict) and "idToLabels" in seg_info:
                    id_to_labels = seg_info["idToLabels"]

                    if show_debug:
                        print(
                            f"{Colors.GREEN}[DEBUG] idToLabels found: {id_to_labels}{Colors.END}"
                        )

                    for id_str, label_dict in id_to_labels.items():
                        if "targetobject" in label_dict:
                            obj_id = int(id_str)
                            if show_debug:
                                print(
                                    f"{Colors.GREEN}[DEBUG] Found targetobject with ID: {obj_id} "
                                    f"-> (value: {label_dict['targetobject']}){Colors.END}"
                                )
                            break

                    if obj_id is None and show_debug:
                        print(
                            f"{Colors.RED}[DEBUG] targetobject not found in any labels{Colors.END}"
                        )

                elif show_debug:
                    print(
                        f"{Colors.RED}[DEBUG] idToLabels not found in semantic_segmentation{Colors.END}"
                    )

    except Exception as e:
        if show_debug:
            print(f"{Colors.RED}[DEBUG] Error accessing semantic info: {e}{Colors.END}")
            import traceback

            traceback.print_exc()

    # Apply masking
    if obj_id is not None:
        mask = segmentation != obj_id
        depth = depth.clone()
        depth[mask] = float("inf")

        if show_debug:
            object_pixels = (~mask).sum().item()
            total_pixels = mask.numel()
            print(
                f"[DEBUG] Masking applied: {object_pixels}/{total_pixels} pixels kept "
                f"({100 * object_pixels / total_pixels:.1f}%)"
            )
    elif show_debug:
        print(
            f"{Colors.YELLOW}[DEBUG] No object masking applied (obj_id is None){Colors.END}"
        )

    # Project to 3D  (camera frame -> world frame -> env-local frame)
    intrinsics = camera.data.intrinsic_matrices
    # Step 1: unproject depth into camera-local 3D coordinates
    points_camera_frame = math_utils.unproject_depth(depth, intrinsics)
    # Step 2: transform from camera frame to world frame using the camera pose
    points_world_frame = math_utils.transform_points(
        points_camera_frame, camera.data.pos_w, camera.data.quat_w_ros
    )
    # Step 3: capture the valid-point mask NOW, while points are still in world frame.
    # Background/masked pixels have depth=inf -> unproject -> inf -> transform -> inf.
    # We must record which points are finite BEFORE the env-origin subtraction,
    # because subtracting env_origins from inf gives nan/inf but subtracting from a
    # valid-but-background zero would silently produce a non-zero value that fools
    # the downstream zero-filter in data_save_utils.
    valid_mask = torch.isfinite(points_world_frame)  # (num_envs, H*W, 3)
    # Step 4: convert world frame → env-local frame
    # env_origins shape: (num_envs, 3) → unsqueeze to (num_envs, 1, 3) for broadcast
    env_origins = env.scene.env_origins.unsqueeze(1)  # (num_envs, 1, 3)
    points_env_local = points_world_frame - env_origins  # (num_envs, H*W, 3)
    # Step 5: zero out invalid points AFTER the subtraction using the pre-shift mask
    points_env_local[~valid_mask] = 0.0

    if show_debug:
        valid_mask = torch.any(points_env_local[0] != 0, dim=-1)
        valid_count = valid_mask.sum().item()
        total_points = points_env_local.shape[1]
        print(
            f"[DEBUG] Valid points: {valid_count}/{total_points} "
            f"({100 * valid_count / total_points:.1f}%)"
        )
        print(f"[DEBUG] {'=' * 60}\n")

    # Visualize point cloud (env-local frame)
    if should_visualize:
        pc_numpy = points_env_local[0].cpu().numpy()
        visualize_point_cloud(pc_numpy, current_step)

    return points_env_local


def set_pca_backend(backend: Literal["gpu", "torch", "open3d"]):
    """
    Set the default PCA backend globally.

    Args:
        backend: One of "gpu" (torch.pca_lowrank), "torch" (custom), or "open3d" (CPU)
    """
    global _DEFAULT_BACKEND
    if backend not in ["gpu", "torch", "open3d"]:
        raise ValueError(
            f"Invalid backend: {backend}. Must be 'gpu', 'torch', or 'open3d'"
        )

    if backend == "open3d" and not OPEN3D_AVAILABLE:
        print("[WARNING] Open3D not available. Falling back to 'gpu' backend.")
        _DEFAULT_BACKEND = "gpu"
    else:
        _DEFAULT_BACKEND = backend
        print(f"[INFO] PCA backend set to: {backend}")


def get_principal_components(points, backend: Optional[str] = None):
    """
    Calculates the centroid and principal directions (axes) of a point cloud using PCA.

    Flexible implementation that can use different backends.

    This function identifies the object's orientation by finding the directions
    of maximum variance. The first principal component typically aligns with
    the object's longest axis (e.g., the length of a bottle).

    Args:
        points (torch.Tensor | np.ndarray): Point cloud of shape (N, 3).
                                            Invalid points [0, 0, 0] are filtered out.
        backend (str, optional): PCA backend - "gpu", "torch", or "open3d".
                                If None, uses global default.

    Returns:
        tuple[np.ndarray, np.ndarray] | tuple[None, None]:
            - centroid: The mean position (center of mass) of the cloud. Shape (3,).
            - eigenvectors: The principal axes as columns. Shape (3, 3).
                           eigenvectors[:, 0] is the primary direction (PC1).
            Returns (None, None) if insufficient valid points.
    """
    backend = backend or _DEFAULT_BACKEND

    if backend == "open3d":
        return _get_principal_components_open3d(points)
    else:
        # Convert to torch if needed
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points)

        # Process single point cloud
        points_batch = points.unsqueeze(0)  # Add batch dimension
        centroids, eigenvectors = get_principal_components_batch(
            points_batch, backend=backend
        )

        # Return as numpy
        centroid = centroids[0].cpu().numpy()
        eigvecs = eigenvectors[0].cpu().numpy()

        # Check if we got valid results
        if np.allclose(centroid, 0) and np.allclose(eigvecs, 0):
            return None, None

        return centroid, eigvecs


def _get_principal_components_open3d(points):
    """Original Open3D CPU implementation."""
    if not OPEN3D_AVAILABLE:
        print("[WARNING] Open3D not available. Use GPU backend instead.")
        return None, None

    # Convert torch.Tensor to numpy if necessary
    if isinstance(points, torch.Tensor):
        points_np = points.detach().cpu().numpy()
    else:
        points_np = points

    # Filter out zero points
    valid_mask = np.any(points_np != 0, axis=1)
    valid_points = points_np[valid_mask]

    if len(valid_points) < 3:
        return None, None

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)

    # Compute mean and covariance matrix
    centroid, covariance = pcd.compute_mean_and_covariance()

    # Calculate Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort by eigenvalues in descending order (PC1, PC2, PC3)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    return centroid, eigenvectors


def get_principal_components_batch(
    points_batch: torch.Tensor,
    min_valid_points: int = 3,
    backend: Optional[Literal["gpu", "torch"]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched GPU-accelerated PCA for multiple point clouds.

    This function processes multiple point clouds in parallel on GPU,
    significantly faster than sequential CPU processing.

    Args:
        points_batch (torch.Tensor): Batch of point clouds, shape (B, N, 3)
                                     where B is batch size, N is number of points.
        min_valid_points (int): Minimum number of valid points required for PCA.
                               Defaults to 3.
        backend (str, optional): PCA backend - "gpu" (pca_lowrank) or "torch" (eigen).
                                If None, uses global default.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - centroids: Mean positions, shape (B, 3)
            - eigenvectors: Principal axes, shape (B, 3, 3)
                           For each batch: eigenvectors[i, :, 0] is PC1

            For point clouds with insufficient points, returns zeros.

    Example:
        >>> points = torch.rand(4, 1000, 3, device='cuda')  # 4 point clouds
        >>> centroids, axes = get_principal_components_batch(points, backend="gpu")
        >>> print(centroids.shape)  # (4, 3)
        >>> print(axes.shape)       # (4, 3, 3)
    """
    backend = backend or _DEFAULT_BACKEND

    # Use pca_lowrank backend if requested
    if backend == "gpu":
        return _get_principal_components_pca_lowrank(points_batch, min_valid_points)
    else:  # backend == "torch"
        return _get_principal_components_eigen(points_batch, min_valid_points)


def _get_principal_components_pca_lowrank(
    points_batch: torch.Tensor, min_valid_points: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated PCA using torch.pca_lowrank().

    This is the FASTEST implementation for GPU processing.

    Note: torch.pca_lowrank uses SVD (Singular Value Decomposition) internally,
    which is more numerically stable than eigendecomposition for PCA.
    """
    device = points_batch.device
    batch_size = points_batch.shape[0]

    # Initialize outputs
    centroids = torch.zeros(batch_size, 3, device=device, dtype=points_batch.dtype)
    eigenvectors = torch.zeros(
        batch_size, 3, 3, device=device, dtype=points_batch.dtype
    )

    # Process each point cloud in batch
    for i in range(batch_size):
        points = points_batch[i]

        # Filter out zero points (invalid/masked points)
        valid_mask = torch.any(points != 0, dim=1)
        valid_points = points[valid_mask]

        if valid_points.shape[0] < min_valid_points:
            continue

        # Compute centroid
        centroid = torch.mean(valid_points, dim=0)
        centroids[i] = centroid

        # Center the points
        centered = valid_points - centroid

        # Use torch.pca_lowrank for fast GPU PCA
        # Returns: U (left singular vectors), S (singular values), V (right singular vectors)
        # For PCA: principal components are the columns of V (or U depending on orientation)
        try:
            # pca_lowrank returns (U, S, V) where V contains the principal components
            # We want q=3 to get the top 3 components
            U, S, V = torch.pca_lowrank(centered, q=3, center=False, niter=2)

            # V has shape (3, 3) and columns are principal components
            # Already sorted by descending singular values (which correspond to eigenvalues)
            eigenvectors[i] = V

        except RuntimeError as e:
            # Fallback to eigendecomposition if pca_lowrank fails
            print(
                f"[WARNING] pca_lowrank failed for batch {i}, using eigen fallback: {e}"
            )
            covariance = torch.mm(centered.T, centered) / valid_points.shape[0]
            eigenvalues, eigvecs = torch.linalg.eigh(covariance)
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvectors[i] = eigvecs[:, idx]

    return centroids, eigenvectors


def _get_principal_components_eigen(
    points_batch: torch.Tensor, min_valid_points: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom PyTorch eigendecomposition implementation.

    This is the original implementation using covariance matrix eigendecomposition.
    Slightly slower than pca_lowrank but more explicit.
    """
    device = points_batch.device
    batch_size = points_batch.shape[0]

    # Initialize outputs
    centroids = torch.zeros(batch_size, 3, device=device, dtype=points_batch.dtype)
    eigenvectors = torch.zeros(
        batch_size, 3, 3, device=device, dtype=points_batch.dtype
    )

    # Process each point cloud in batch
    for i in range(batch_size):
        points = points_batch[i]

        # Filter out zero points (invalid/masked points)
        valid_mask = torch.any(points != 0, dim=1)
        valid_points = points[valid_mask]

        if valid_points.shape[0] < min_valid_points:
            continue

        # Compute centroid
        centroid = torch.mean(valid_points, dim=0)

        # Center the points
        centered = valid_points - centroid

        # Compute covariance matrix: C = (1/N) * X^T * X
        # Shape: (3, N) @ (N, 3) -> (3, 3)
        covariance = torch.mm(centered.T, centered) / valid_points.shape[0]

        # Compute eigenvalues and eigenvectors
        # torch.linalg.eigh returns eigenvalues in ascending order
        eigenvalues, eigvecs = torch.linalg.eigh(covariance)

        # Sort by eigenvalues in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigvecs = eigvecs[:, idx]

        # Store results
        centroids[i] = centroid
        eigenvectors[i] = eigvecs

    return centroids, eigenvectors


def get_pca_features_batch(
    points_batch: torch.Tensor,
    min_valid_points: int = 3,
    backend: Optional[Literal["gpu", "torch"]] = None,
) -> torch.Tensor:
    """
    Extract PCA features (centroid + flattened axes) for batched point clouds.

    Convenience function that returns features in a flat format suitable
    for neural network input or observation vectors.

    Args:
        points_batch (torch.Tensor): Batch of point clouds, shape (B, N, 3)
        min_valid_points (int): Minimum number of valid points required.
        backend (str, optional): PCA backend - "gpu" or "torch". If None, uses global default.

    Returns:
        torch.Tensor: Features of shape (B, 12) containing:
                     - [0:3]: centroid (x, y, z)
                     - [3:12]: flattened eigenvectors (3x3 matrix flattened)
                     Returns zeros for point clouds with insufficient points.

    Example:
        >>> points = torch.rand(4, 1000, 3, device='cuda')
        >>> features = get_pca_features_batch(points, backend="gpu")
        >>> print(features.shape)  # (4, 12)
        >>> centroids = features[:, :3]
        >>> axes = features[:, 3:].reshape(-1, 3, 3)
    """
    centroids, eigenvectors = get_principal_components_batch(
        points_batch, min_valid_points, backend=backend
    )

    # Flatten eigenvectors from (B, 3, 3) to (B, 9)
    eigenvectors_flat = eigenvectors.reshape(eigenvectors.shape[0], -1)

    # Concatenate centroid and axes: (B, 3) + (B, 9) = (B, 12)
    features = torch.cat([centroids, eigenvectors_flat], dim=1)

    return features


def compare_pca_backends(points_batch: torch.Tensor, verbose: bool = True) -> dict:
    """
    Compare different PCA backend implementations for accuracy and speed.

    Useful for debugging and validating that all backends give similar results.

    Args:
        points_batch (torch.Tensor): Batch of point clouds to test
        verbose (bool): Print detailed comparison

    Returns:
        dict: Comparison results with timing and accuracy metrics
    """
    import time

    results = {}

    # Test GPU backend (pca_lowrank)
    if torch.cuda.is_available():
        points_gpu = points_batch.cuda()

        start = time.time()
        cent_gpu, axes_gpu = get_principal_components_batch(points_gpu, backend="gpu")
        torch.cuda.synchronize()
        time_gpu = time.time() - start

        results["gpu"] = {"time": time_gpu, "centroids": cent_gpu, "axes": axes_gpu}

    # Test torch backend (eigendecomposition)
    start = time.time()
    cent_torch, axes_torch = get_principal_components_batch(
        points_batch, backend="torch"
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_torch = time.time() - start

    results["torch"] = {"time": time_torch, "centroids": cent_torch, "axes": axes_torch}

    # Test Open3D backend (if available, single point cloud only)
    if OPEN3D_AVAILABLE and points_batch.shape[0] > 0:
        start = time.time()
        cent_o3d, axes_o3d = get_principal_components(
            points_batch[0].cpu().numpy(), backend="open3d"
        )
        time_o3d = time.time() - start

        if cent_o3d is not None:
            results["open3d"] = {
                "time": time_o3d,
                "centroids": torch.from_numpy(cent_o3d).unsqueeze(0),
                "axes": torch.from_numpy(axes_o3d).unsqueeze(0),
            }

    if verbose:
        print("\n" + "=" * 70)
        print("PCA BACKEND COMPARISON")
        print("=" * 70)

        # Timing comparison
        print("\nTiming (ms):")
        for backend, data in results.items():
            print(f"  {backend:10s}: {data['time'] * 1000:.2f}ms")

        # Accuracy comparison (compare to torch backend as reference)
        if "gpu" in results and "torch" in results:
            cent_diff = torch.mean(
                torch.abs(results["gpu"]["centroids"] - results["torch"]["centroids"])
            ).item()
            axes_diff = torch.mean(
                torch.abs(results["gpu"]["axes"] - results["torch"]["axes"])
            ).item()

            print(f"\nAccuracy (GPU vs Torch):")
            print(f"  Centroid MAE: {cent_diff:.6f}")
            print(f"  Axes MAE: {axes_diff:.6f}")

            if cent_diff < 1e-4 and axes_diff < 1e-4:
                print("  ✅ Results are numerically identical")
            elif cent_diff < 1e-2 and axes_diff < 1e-2:
                print("  ⚠️  Minor differences (acceptable)")
            else:
                print("  ❌ Significant differences detected!")

        print("=" * 70 + "\n")

    return results


# Keep original functions for backward compatibility
def get_point_normal(points, point_index, search_radius=0.05):
    """
    Estimates the surface normal of a specific point within a cloud.

    CPU-based implementation using Open3D.

    Useful for fine-tuning the gripper approach angle relative to a
    specific contact point on the object's surface.

    Args:
        points (torch.Tensor | np.ndarray): Point cloud of shape (N, 3).
        point_index (int): The index of the point to retrieve the normal for.
        search_radius (float): Radius for neighbor search to estimate local plane.

    Returns:
        np.ndarray | None: The estimated normal vector [nx, ny, nz],
                          or None if Open3D unavailable or index invalid.
    """
    if not OPEN3D_AVAILABLE:
        print("[WARNING] Open3D not available. Cannot compute normals.")
        return None

    if isinstance(points, torch.Tensor):
        points_np = points.detach().cpu().numpy()
    else:
        points_np = points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # Estimate normals using a hybrid search (radius and max neighbors)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=search_radius, max_nn=30
        )
    )

    normals = np.asarray(pcd.normals)

    if point_index < len(normals):
        return normals[point_index]
    return None


# Set default backend on import
set_pca_backend("gpu")


def select_random_point_and_normal(points, search_radius=0.05):
    """
    Selects a random valid point from the cloud and calculates its surface normal.

    This is useful for local grasping strategies where the gripper needs to align
    with a specific surface patch rather than the entire object's principal axes.

    Args:
        points (torch.Tensor | np.ndarray): Point cloud of shape (N, 3).
        search_radius (float): Radius for normal estimation. Defaults to 0.05.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - selected_point: Coordinates [x, y, z] of the random point.
            - normal: The estimated normal vector [nx, ny, nz].
            Returns (None, None) if no valid points are found.
    """
    # Convert to numpy for selection logic
    if isinstance(points, torch.Tensor):
        points_np = points.detach().cpu().numpy()
    else:
        points_np = points

    # Filter out invalid [0, 0, 0] points (masked regions) [cite: 81]
    valid_mask = np.any(points_np != 0, axis=1)
    valid_points = points_np[valid_mask]

    if len(valid_points) == 0:
        return None, None

    # Select a random index within the valid point set
    random_idx = np.random.randint(0, len(valid_points))
    selected_point = valid_points[random_idx]

    # Calculate normal using the existing Open3D-based utility [cite: 124, 130]
    normal = get_point_normal(valid_points, random_idx, search_radius=search_radius)

    return selected_point, normal
