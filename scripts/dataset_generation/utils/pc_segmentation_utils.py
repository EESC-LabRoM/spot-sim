"""
Point cloud segmentation and clustering utilities for robotic grasping.

This module provides methods to decompose complex objects into functional
sub-parts using geometric segmentation (RANSAC) and clustering (DBSCAN).
"""

import open3d as o3d
import numpy as np
import torch
from typing import List, Tuple, Optional


def segment_by_clusters(
    points,
    eps: float = 0.005,  # 5 mm
    min_points: int = 50,  #
    adaptive: bool = True,  # automatically adjust eps
    debug: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Clusters the point cloud into separate components using DBSCAN.
    Identifies sub-bodies based on physical proximity.

    - eps more sensitive
    - adaptive mode to auto-calculate eps
    - validation and debugging

    Args:
        points: Point cloud (torch.Tensor or np.ndarray) of shape (N, 3)
        eps (float): Maximum distance between points in same cluster (meters) -> Default: 0.005 (5mm)
        min_points (int): Minimum points to form a cluster. Default: 50
        adaptive (bool): If True, automatically calculate eps based on point cloud size
        debug (bool): Print detailed clustering information

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]:
            - clusters: List of point arrays for each cluster
            - centroids: List of centroid positions for each cluster
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    valid_points = points[np.any(points != 0, axis=1)]

    if len(valid_points) < min_points:
        if debug:
            print(f"[DBSCAN] Insufficient points: {len(valid_points)} < {min_points}")
        return [], []

    pcd.points = o3d.utility.Vector3dVector(valid_points)

    # Adaptive eps calculation
    if adaptive:
        # Calculate point cloud extent
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        max_extent = np.max(extent)

        # Set eps to 1-2% of max extent (heuristic)
        eps_adaptive = max_extent * 0.015  # 1.5% of object size

        # Clamp to reasonable range
        eps_adaptive = np.clip(eps_adaptive, 0.003, 0.02)  # 3mm to 2cm

        if debug:
            print(f"[DBSCAN] Object extent: {extent}")
            print(
                f"[DBSCAN] Adaptive eps: {eps_adaptive:.4f}m ({eps_adaptive * 1000:.1f}mm)"
            )

        eps = eps_adaptive

    if debug:
        print(f"[DBSCAN] Input points: {len(valid_points)}")
        print(
            f"[DBSCAN] Parameters: eps={eps:.4f}m ({eps * 1000:.1f}mm), min_points={min_points}"
        )

    # DBSCAN clustering
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )

    # Count clusters
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
    n_noise = np.sum(labels == -1)

    if debug:
        print(f"[DBSCAN] Found {n_clusters} clusters")
        print(f"[DBSCAN] Noise points: {n_noise} ({100 * n_noise / len(labels):.1f}%)")

    clusters = []
    centroids = []

    # Iterate through all found clusters (label -1 is noise)
    max_label = labels.max()
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]

        if len(cluster_indices) == 0:
            continue

        cluster_pcd = pcd.select_by_index(cluster_indices)
        cluster_points = np.asarray(cluster_pcd.points)

        clusters.append(cluster_points)
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)

        if debug:
            print(
                f"[DBSCAN]   Cluster {i}: {len(cluster_points)} points, "
                f"centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]"
            )

    # Sort by size (largest first)
    if clusters:
        sizes = [len(c) for c in clusters]
        sorted_indices = np.argsort(sizes)[::-1]
        clusters = [clusters[i] for i in sorted_indices]
        centroids = [centroids[i] for i in sorted_indices]

        if debug:
            print(f"[DBSCAN] Clusters sorted by size: {[len(c) for c in clusters]}")

    return clusters, centroids


def analyze_point_cloud_structure(points, debug: bool = True):
    """
    Comprehensive analysis of point cloud structure.

    Args:
        points: Point cloud
        debug (bool): Print analysis
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    valid_points = points[np.any(points != 0, axis=1)]

    if len(valid_points) == 0:
        print("[ANALYSIS] No valid points!")
        return

    # Basic statistics
    centroid = np.mean(valid_points, axis=0)
    std = np.std(valid_points, axis=0)
    min_coords = np.min(valid_points, axis=0)
    max_coords = np.max(valid_points, axis=0)
    extent = max_coords - min_coords

    # Nearest neighbor distances
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)

    # Sample 1000 points for distance analysis
    sample_size = min(1000, len(valid_points))
    sample_indices = np.random.choice(len(valid_points), sample_size, replace=False)
    sample_points = valid_points[sample_indices]

    distances = []
    for point in sample_points:
        dists = np.linalg.norm(valid_points - point, axis=1)
        dists_sorted = np.sort(dists)
        if len(dists_sorted) > 1:
            distances.append(dists_sorted[1])  # Nearest neighbor (excluding self)

    if distances:
        mean_nn_dist = np.mean(distances)
        median_nn_dist = np.median(distances)
        std_nn_dist = np.std(distances)
    else:
        mean_nn_dist = median_nn_dist = std_nn_dist = 0

    if debug:
        print(f"\n{'=' * 70}")
        print(f"POINT CLOUD STRUCTURE ANALYSIS")
        print(f"{'=' * 70}")
        print(f"Total points: {len(valid_points)}")
        print(f"\nCentroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
        print(f"Std deviation: [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")
        print(f"\nExtent (size):")
        print(f"  X: {extent[0]:.3f}m ({extent[0] * 100:.1f}cm)")
        print(f"  Y: {extent[1]:.3f}m ({extent[1] * 100:.1f}cm)")
        print(f"  Z: {extent[2]:.3f}m ({extent[2] * 100:.1f}cm)")
        print(f"\nNearest neighbor distances (sampled):")
        print(f"  Mean: {mean_nn_dist:.4f}m ({mean_nn_dist * 1000:.1f}mm)")
        print(f"  Median: {median_nn_dist:.4f}m ({median_nn_dist * 1000:.1f}mm)")
        print(f"  Std: {std_nn_dist:.4f}m ({std_nn_dist * 1000:.1f}mm)")
        print(f"\nRecommended DBSCAN eps:")
        print(
            f"  Conservative: {median_nn_dist * 2:.4f}m ({median_nn_dist * 2000:.1f}mm)"
        )
        print(
            f"  Aggressive: {median_nn_dist * 5:.4f}m ({median_nn_dist * 5000:.1f}mm)"
        )
        print(
            f"  Object size based: {np.max(extent) * 0.015:.4f}m ({np.max(extent) * 15:.1f}mm)"
        )
        print(f"{'=' * 70}\n")

    return {
        "num_points": len(valid_points),
        "centroid": centroid,
        "extent": extent,
        "mean_nn_distance": mean_nn_dist,
        "median_nn_distance": median_nn_dist,
        "recommended_eps": median_nn_dist * 3,  # 3x median distance
    }


def segment_by_obb_tree(
    points, max_depth: int = 1, min_points: int = 100, debug: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray], List]:
    """
    Decomposes a point cloud into hierarchical sub-parts using Open3D OrientedBoundingBoxes.

    Args:
        points: Point cloud (torch.Tensor or np.ndarray).
        max_depth (int): Recursion depth. depth=1 produces 2 parts, depth=2 produces 4.
        min_points (int): Minimum points per segment. Default: 100
        debug (bool): Print segmentation details

    Returns:
        tuple: (list of cluster points, list of centroids, list of o3d OBB objects)
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    valid_mask = np.any(points != 0, axis=1)
    valid_points = points[valid_mask]

    if len(valid_points) < min_points:
        if debug:
            print(f"[OBB Tree] Insufficient points: {len(valid_points)} < {min_points}")
        return [], [], []

    if debug:
        print(f"\n{'=' * 70}")
        print(f"OBB TREE SEGMENTATION")
        print(f"{'=' * 70}")
        print(f"Input: {len(valid_points)} points")
        print(f"Max depth: {max_depth}, Min points: {min_points}\n")

    # Build OBB tree recursively
    clusters, centroids, obbs = _build_obb_tree_recursive(
        valid_points, max_depth, min_points, current_depth=0, debug=debug
    )

    # Sort by size (largest first)
    if clusters:
        sizes = [len(c) for c in clusters]
        sorted_indices = np.argsort(sizes)[::-1]
        clusters = [clusters[i] for i in sorted_indices]
        centroids = [centroids[i] for i in sorted_indices]
        obbs = [obbs[i] for i in sorted_indices]

        if debug:
            print(f"\n{'=' * 70}")
            print(f"SEGMENTATION RESULTS")
            print(f"{'=' * 70}")
            print(f"Found {len(clusters)} segments")
            for i, (cluster, cent) in enumerate(zip(clusters, centroids)):
                extent = np.asarray(obbs[i].extent)
                print(f"\nSegment {i}:")
                print(f"  Points: {len(cluster)}")
                print(f"  Centroid: [{cent[0]:.3f}, {cent[1]:.3f}, {cent[2]:.3f}]")
                print(f"  Extent: [{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}]")
            print(f"\nSegments sorted by size: {[len(c) for c in clusters]}")
            print(f"{'=' * 70}\n")

    return clusters, centroids, obbs


def _build_obb_tree_recursive(
    points: np.ndarray,
    max_depth: int,
    min_points: int,
    current_depth: int = 0,
    debug: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List]:
    """
    Internal recursive function for building OBB tree.

    Args:
        points: Valid points to process
        max_depth: Maximum recursion depth
        min_points: Minimum points to continue splitting
        current_depth: Current depth in tree
        debug: Print debug info

    Returns:
        tuple: (clusters, centroids, obbs)
    """
    if len(points) < min_points:
        return [], [], []

    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Get the Oriented Bounding Box
    obb = pcd.get_oriented_bounding_box()
    center = np.asarray(obb.center)
    extent = np.asarray(obb.extent)

    if debug:
        indent = "  " * current_depth
        print(f"{indent}[OBB Tree] Level {current_depth}: {len(points)} points")
        print(
            f"{indent}  Centroid: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]"
        )
        print(f"{indent}  Extent: [{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}]")

    # Base case: max depth reached or too few points
    if current_depth >= max_depth or len(points) < min_points * 2:
        if debug:
            indent = "  " * current_depth
            print(f"{indent}  → Leaf node (depth={current_depth})")
        return [points], [center], [obb]

    # Identify the longest axis for splitting
    split_axis_idx = np.argmax(extent)

    # Get the split vector (eigenvector corresponding to longest extent)
    # Open3D OBB.R is rotation matrix where columns are local axes
    split_vector = np.asarray(obb.R)[:, split_axis_idx]

    # Partition points based on projection onto the longest axis
    centered_points = points - center
    projection = centered_points @ split_vector

    mask_a = projection > 0
    points_a = points[mask_a]
    points_b = points[~mask_a]

    # Check if split is meaningful
    if len(points_a) < min_points or len(points_b) < min_points:
        if debug:
            indent = "  " * current_depth
            print(
                f"{indent}  → Leaf node (split too small: {len(points_a)}, {len(points_b)})"
            )
        return [points], [center], [obb]

    if debug:
        indent = "  " * current_depth
        print(f"{indent}  Split: {len(points_a)} | {len(points_b)} points")

    # Recursive splitting
    clusters_a, centroids_a, obbs_a = _build_obb_tree_recursive(
        points_a, max_depth, min_points, current_depth + 1, debug
    )
    clusters_b, centroids_b, obbs_b = _build_obb_tree_recursive(
        points_b, max_depth, min_points, current_depth + 1, debug
    )

    return clusters_a + clusters_b, centroids_a + centroids_b, obbs_a + obbs_b


def segment_multi_method(
    points, method: str = "auto", debug: bool = False, **kwargs
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Segment point cloud using multiple methods and choose the best result.

    Args:
        points: Point cloud
        method (str): Segmentation method:
            - "auto": Try DBSCAN and OBB, choose best
            - "dbscan": Use DBSCAN only
            - "obb": Use OBB Tree only
        debug (bool): Enable debugging
        **kwargs: Method-specific parameters

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: clusters, centroids
    """
    if method == "dbscan":
        return segment_by_clusters(points, debug=debug, **kwargs)

    elif method == "obb":
        clusters, centroids, _ = segment_by_obb_tree(points, debug=debug, **kwargs)
        return clusters, centroids

    elif method == "auto":
        # Try both methods
        if debug:
            print(f"\n[AUTO] Trying DBSCAN...")
        clusters_db, centroids_db = segment_by_clusters(points, debug=debug, **kwargs)

        if debug:
            print(f"\n[AUTO] Trying OBB Tree...")
        clusters_obb, centroids_obb, _ = segment_by_obb_tree(
            points,
            debug=debug,
            max_depth=kwargs.get("max_depth", 1),
            min_points=kwargs.get("min_points", 100),
        )

        # Choose based on which gives more meaningful segmentation
        n_db = len(clusters_db)
        n_obb = len(clusters_obb)

        if debug:
            print(f"\n[AUTO] DBSCAN found {n_db} clusters")
            print(f"[AUTO] OBB Tree found {n_obb} segments")

        # Prefer OBB if it finds 2 segments (typical for elongated objects)
        if n_obb == 2:
            if debug:
                print(f"[AUTO] Using OBB Tree result (optimal 2 segments)")
            return clusters_obb, centroids_obb
        elif n_db > 1:
            if debug:
                print(f"[AUTO] Using DBSCAN result (multiple clusters)")
            return clusters_db, centroids_db
        elif n_obb > 1:
            if debug:
                print(f"[AUTO] Using OBB Tree result")
            return clusters_obb, centroids_obb
        else:
            # Both found 1 cluster, use DBSCAN
            if debug:
                print(f"[AUTO] Both methods found 1 cluster, using DBSCAN")
            return clusters_db, centroids_db

    else:
        raise ValueError(f"Unknown method: {method}")
