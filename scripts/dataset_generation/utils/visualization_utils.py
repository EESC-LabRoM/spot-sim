"""
Visualization utilities for point clouds and camera outputs.

This module provides utilities for visualizing 3D point clouds and camera sensor data
including RGB images, depth maps, and semantic segmentation.

ENHANCED VERSION with configurable debug logging.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# Use non-interactive backend for saving plots
matplotlib.use("Agg")


class Colors:
    """ANSI color codes for terminal output."""

    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"


def visualize_point_cloud(points, step_num, output_dir=None):
    """
    Visualize 3D point cloud using matplotlib with multiple viewpoints.

    [... rest of docstring same as before ...]
    """
    if output_dir is None:
        output_dir = os.getcwd()

    os.makedirs(output_dir, exist_ok=True)

    # Filter out zero points (invalid/masked points)
    valid_mask = np.any(points != 0, axis=1)
    valid_points = points[valid_mask]

    if len(valid_points) == 0:
        print(f"{Colors.YELLOW}[WARNING] No valid points to visualize!{Colors.END}")
        return

    print(f"\n{'=' * 70}")
    print(f"POINT CLOUD VISUALIZATION - Step {step_num}")
    print(f"{'=' * 70}")
    print(f"{Colors.BLUE}Total points: {len(points)}{Colors.END}")
    print(
        f"{Colors.BLUE}Valid points: {len(valid_points)} "
        f"({100 * len(valid_points) / len(points):.1f}%){Colors.END}"
    )

    # Calculate statistics
    center = np.mean(valid_points, axis=0)
    min_coords = np.min(valid_points, axis=0)
    max_coords = np.max(valid_points, axis=0)

    print(f"{Colors.GREEN}Center: {center.tolist()}{Colors.END}")
    print(f"{Colors.GREEN}Bounds:{Colors.END}")
    print(f"  {Colors.GREEN}X: [{min_coords[0]:.3f}, {max_coords[0]:.3f}]{Colors.END}")
    print(f"  {Colors.GREEN}Y: [{min_coords[1]:.3f}, {max_coords[1]:.3f}]{Colors.END}")
    print(f"  {Colors.GREEN}Z: [{min_coords[2]:.3f}, {max_coords[2]:.3f}]{Colors.END}")

    # [... rest of visualization code same as before ...]
    # Create figure with multiple views
    fig = plt.figure(figsize=(20, 5))

    # 3D view
    ax1 = fig.add_subplot(141, projection="3d")
    scatter = ax1.scatter(
        valid_points[:, 0],
        valid_points[:, 1],
        valid_points[:, 2],
        c=valid_points[:, 2],
        cmap="viridis",
        s=1,
        alpha=0.6,
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("3D Point Cloud", fontsize=12, fontweight="bold")
    plt.colorbar(scatter, ax=ax1, label="Z coordinate", shrink=0.5)

    # Set equal aspect ratio
    max_range = (
        np.array(
            [
                max_coords[0] - min_coords[0],
                max_coords[1] - min_coords[1],
                max_coords[2] - min_coords[2],
            ]
        ).max()
        / 2.0
    )
    mid_x = (max_coords[0] + min_coords[0]) * 0.5
    mid_y = (max_coords[1] + min_coords[1]) * 0.5
    mid_z = (max_coords[2] + min_coords[2]) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # Top view (X-Y)
    ax2 = fig.add_subplot(142)
    ax2.scatter(
        valid_points[:, 0],
        valid_points[:, 1],
        c=valid_points[:, 2],
        cmap="viridis",
        s=2,
        alpha=0.6,
    )
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Top View (X-Y)", fontsize=12, fontweight="bold")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # Front view (X-Z)
    ax3 = fig.add_subplot(143)
    ax3.scatter(
        valid_points[:, 0],
        valid_points[:, 2],
        c=valid_points[:, 1],
        cmap="viridis",
        s=2,
        alpha=0.6,
    )
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.set_title("Front View (X-Z)", fontsize=12, fontweight="bold")
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.3)

    # Side view (Y-Z)
    ax4 = fig.add_subplot(144)
    ax4.scatter(
        valid_points[:, 1],
        valid_points[:, 2],
        c=valid_points[:, 0],
        cmap="viridis",
        s=2,
        alpha=0.6,
    )
    ax4.set_xlabel("Y")
    ax4.set_ylabel("Z")
    ax4.set_title("Side View (Y-Z)", fontsize=12, fontweight="bold")
    ax4.set_aspect("equal")
    ax4.grid(True, alpha=0.3)

    plt.suptitle(
        f"Bottle Point Cloud - Step {step_num}\n{len(valid_points)} points",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, f"pointcloud_step_{step_num}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[SAVED] Point cloud visualization saved to: {output_path}")
    print(f"{'=' * 70}\n")
    plt.close()


def visualize_camera_outputs(camera, step_num, output_dir=None):
    """Visualize RGB, Depth, and Segmentation outputs from a camera sensor."""
    if output_dir is None:
        output_dir = os.getcwd()

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'=' * 70}")
    print(f"CAMERA VISUALIZATION - Step {step_num}")
    print(f"{'=' * 70}")

    # Get data from first environment
    rgb_data = camera.data.output["rgb"][0].cpu().numpy()
    depth_data = camera.data.output["distance_to_image_plane"][0].cpu().numpy()
    seg_data = camera.data.output["semantic_segmentation"][0].cpu().numpy()

    # Print basic info
    print(f"RGB shape: {rgb_data.shape}, dtype: {rgb_data.dtype}")
    print(f"Depth shape: {depth_data.shape}, dtype: {depth_data.dtype}")
    print(f"Segmentation shape: {seg_data.shape}, dtype: {seg_data.dtype}")

    # [... rest of visualization code ...]
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Camera Outputs - Step {step_num}", fontsize=16, fontweight="bold")

    # RGB Image
    ax = axes[0, 0]
    if rgb_data.dtype != np.uint8:
        rgb_display = (
            (rgb_data * 255).astype(np.uint8)
            if rgb_data.max() <= 1.0
            else rgb_data.astype(np.uint8)
        )
    else:
        rgb_display = rgb_data

    if rgb_display.shape[-1] == 4:
        rgb_display = rgb_display[:, :, :3]

    ax.imshow(rgb_display)
    ax.set_title("RGB Image", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Depth Image
    ax = axes[0, 1]
    depth_display = depth_data.squeeze()
    depth_finite = depth_display.copy()
    depth_finite[~np.isfinite(depth_finite)] = 0
    max_depth = (
        depth_finite[depth_finite > 0].max() if (depth_finite > 0).any() else 1.0
    )
    depth_display[~np.isfinite(depth_display)] = max_depth

    im = ax.imshow(depth_display, cmap="viridis")
    ax.set_title(
        f"Depth Image\nMin: {depth_finite.min():.2f}, Max: {max_depth:.2f}",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    print(f"Depth stats: min={depth_finite.min():.3f}, max={max_depth:.3f}")

    # Segmentation Image
    ax = axes[1, 0]
    seg_display = seg_data.squeeze()
    unique_ids = np.unique(seg_display)

    im = ax.imshow(seg_display, cmap="tab20")
    ax.set_title(
        f"Segmentation Map\nUnique IDs: {unique_ids}", fontsize=14, fontweight="bold"
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    print(f"\nUnique segmentation IDs: {unique_ids}")
    for uid in unique_ids:
        count = (seg_display == uid).sum()
        percentage = 100 * count / seg_display.size
        print(f"  ID {uid}: {count:6d} pixels ({percentage:5.1f}%)")

    # Segmentation Histogram
    ax = axes[1, 1]
    counts = [np.sum(seg_display == uid) for uid in unique_ids]
    bars = ax.bar(unique_ids.astype(str), counts, color="steelblue", edgecolor="black")
    ax.set_title("Segmentation ID Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Segmentation ID", fontsize=12)
    ax.set_ylabel("Pixel Count", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}\n({100 * count / seg_display.size:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"camera_visualization_step_{step_num}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n[SAVED] Visualization saved to: {output_path}")
    plt.close()

    # Print camera info dict
    print(f"\n{'=' * 70}")
    print("CAMERA INFO DICT ANALYSIS")
    print(f"{'=' * 70}")
    info = camera.data.info
    print(f"Info type: {type(info)}")
    print(f"Info keys: {info.keys() if isinstance(info, dict) else 'N/A'}")

    if isinstance(info, dict) and "semantic_segmentation" in info:
        seg_info = info["semantic_segmentation"]
        print(f"\nSegmentation info type: {type(seg_info)}")
        if isinstance(seg_info, dict):
            print(f"Segmentation info keys: {list(seg_info.keys())}")
            for key, value in seg_info.items():
                print(f"  '{key}': {value}")
        else:
            print(f"Segmentation info content: {seg_info}")

    print(f"{'=' * 70}\n")


def visualize_pca_on_pc(points, centroid, eigenvectors, step_num, output_dir=None):
    """
    Visualize PCA principal components (axes) overlaid on a 3D point cloud.

    Args:
        points (np.ndarray): Point cloud of shape (N, 3).
        centroid (np.ndarray): Center of mass (3,).
        eigenvectors (np.ndarray): Principal axes (3, 3) as columns.
        step_num (int): Current simulation step for filename.
        output_dir (str, optional): Directory to save the plot.
    """

    if output_dir is None:
        output_dir = os.getcwd()

    os.makedirs(output_dir, exist_ok=True)

    # Filter out zero points (invalid/masked points)
    valid_mask = np.any(points != 0, axis=1)
    valid_points = points[valid_mask]

    if len(valid_points) == 0:
        return

    # Create figure with 4 views (1x4 grid)
    fig = plt.figure(figsize=(24, 6))

    # Scale axes for visibility (length based on object bounds)
    scale = np.max(np.linalg.norm(valid_points - centroid, axis=1)) * 0.5
    colors = ["r", "g", "b"]
    labels = ["PC1 (Major)", "PC2 (Minor)", "PC3 (Minor)"]

    # 1. 3D View
    ax1 = fig.add_subplot(141, projection="3d")
    ax1.scatter(
        valid_points[:, 0],
        valid_points[:, 1],
        valid_points[:, 2],
        c=valid_points[:, 2],
        cmap="viridis",
        s=1,
        alpha=0.2,
    )

    for i in range(3):
        axis = eigenvectors[:, i]
        ax1.quiver(
            centroid[0],
            centroid[1],
            centroid[2],
            axis[0],
            axis[1],
            axis[2],
            color=colors[i],
            length=scale,
            label=labels[i],
            linewidth=3,
        )

    ax1.set_title("3D PCA Alignment", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize="small")

    # 2. 2D Orthographic Views
    # Configuration for Top (X-Y), Front (X-Z), Side (Y-Z) views
    views = [
        (142, [0, 1], "Top View (X-Y)", "X", "Y"),
        (143, [0, 2], "Front View (X-Z)", "X", "Z"),
        (144, [1, 2], "Side View (Y-Z)", "Y", "Z"),
    ]

    for subplot_idx, axes_idx, title, xlabel, ylabel in views:
        ax = fig.add_subplot(subplot_idx)
        ax.scatter(
            valid_points[:, axes_idx[0]],
            valid_points[:, axes_idx[1]],
            c=valid_points[:, 2],
            cmap="viridis",
            s=2,
            alpha=0.2,
        )

        for i in range(3):
            axis = eigenvectors[:, i]
            # Draw 2D projection of the eigenvectors
            ax.quiver(
                centroid[axes_idx[0]],
                centroid[axes_idx[1]],
                axis[axes_idx[0]],
                axis[axes_idx[1]],
                color=colors[i],
                scale=3,
                width=0.01,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"PCA Alignment Analysis - Step {step_num}", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()

    # Save logic
    output_path = os.path.join(output_dir, f"pca_multiview_step_{step_num}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(
        f"{Colors.BLUE}[SAVED] Multi-view PCA visualization saved to: {output_path}{Colors.END}"
    )
    plt.close()


def visualize_point_and_vector(
    points, selected_point, vector, step_num, type_vector="vector", output_dir=None
):
    """
    Visualizes a point cloud with a highlighted point and its vector in 4 views.
    Based on visualize_point_cloud() to maintain consistent view angles.
    Args:
        points (np.ndarray): Full point cloud (N, 3).
        selected_point (np.ndarray): Coordinates of the point to highlight (3,).
        vector (np.ndarray): Normal or Centroid vector at the selected point (3,).
        step_num (int): Current simulation step for naming the output file.
        type_vector (str, optional): Type of vector being visualized (default: "vector").
        output_dir (str, optional): Directory to save the plot.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    # Filter invalid points for background cloud
    valid_mask = np.any(points != 0, axis=1)
    valid_points = points[valid_mask]
    if len(valid_points) == 0 or selected_point is None or vector is None:
        return
    fig = plt.figure(figsize=(24, 6))
    # Calculate scale for the vector visualization based on cloud bounds
    max_coords = np.max(valid_points, axis=0)
    min_coords = np.min(valid_points, axis=0)
    scale = np.max(max_coords - min_coords) * 0.2
    # 1. 3D View
    ax1 = fig.add_subplot(141, projection="3d")
    ax1.scatter(
        valid_points[:, 0],
        valid_points[:, 1],
        valid_points[:, 2],
        c="gray",
        s=1,
        alpha=0.1,
    )  # Background cloud
    ax1.scatter(
        selected_point[0],
        selected_point[1],
        selected_point[2],
        c="red",
        s=50,
        label="Grasp Point",
    )
    # Normalize vector for consistent visualization if it's very small
    vector_magnitude = np.linalg.norm(vector)
    if vector_magnitude > 0:
        display_vector = vector / vector_magnitude * scale
    else:
        display_vector = vector

    ax1.quiver(
        selected_point[0],
        selected_point[1],
        selected_point[2],
        display_vector[0],
        display_vector[1],
        display_vector[2],
        length=1.0,
        color="red",
        linewidth=3,
        label=type_vector,
    )
    ax1.set_title("3D View", fontsize=12, fontweight="bold")
    # 2. 2D Views (Top, Front, Side) following visualize_point_cloud logic
    views = [
        (142, [0, 1], "Top View (X-Y)"),
        (143, [0, 2], "Front View (X-Z)"),
        (144, [1, 2], "Side View (Y-Z)"),
    ]
    for subplot_idx, axes_idx, title in views:
        ax = fig.add_subplot(subplot_idx)
        ax.scatter(
            valid_points[:, axes_idx[0]],
            valid_points[:, axes_idx[1]],
            c="gray",
            s=2,
            alpha=0.1,
        )
        ax.scatter(
            selected_point[axes_idx[0]], selected_point[axes_idx[1]], c="red", s=60
        )
        # Normalize vector for 2D projection
        vector_2d = np.array([vector[axes_idx[0]], vector[axes_idx[1]]])
        vector_2d_magnitude = np.linalg.norm(vector_2d)
        if vector_2d_magnitude > 0:
            display_vector_2d = vector_2d / vector_2d_magnitude
        else:
            display_vector_2d = vector_2d

        ax.quiver(
            selected_point[axes_idx[0]],
            selected_point[axes_idx[1]],
            display_vector_2d[0],
            display_vector_2d[1],
            color="red",
            scale=1,
            scale_units="xy",
            width=0.015,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    plt.suptitle(
        f"Surface Analysis: Point and {type_vector} - Step {step_num}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"point_{type_vector}_step_{step_num}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(
        f"{Colors.BLUE}[SAVED] Point {type_vector} visualization saved to: {output_path}{Colors.END}"
    )
    plt.close()


def visualize_segmented_clouds(clusters, centroids, step_num, output_dir=None):
    """
    Visualizes multiple point cloud clusters with their respective centroids.
    Uses a 4-view layout (3D, Top, Front, Side) to validate segmentation.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if not clusters:
        print(f"{Colors.RED}[WARNING] No clusters to visualize!{Colors.END}")
        return

    fig = plt.figure(figsize=(24, 6))
    # Use a colormap to distinguish clusters
    cmap = plt.get_cmap("tab10")

    # View configurations following the project pattern
    views = [
        (141, None, "3D Segmented View"),
        (142, [0, 1], "Top View (X-Y)"),
        (143, [0, 2], "Front View (X-Z)"),
        (144, [1, 2], "Side View (Y-Z)"),
    ]

    for subplot_idx, axes_idx, title in views:
        if axes_idx is None:  # 3D View
            ax = fig.add_subplot(subplot_idx, projection="3d")
            for i, (cluster, centroid) in enumerate(zip(clusters, centroids)):
                color = cmap(i % 10)
                ax.scatter(
                    cluster[:, 0],
                    cluster[:, 1],
                    cluster[:, 2],
                    color=color,
                    s=1,
                    alpha=0.3,
                )
                ax.scatter(
                    centroid[0],
                    centroid[1],
                    centroid[2],
                    color="black",
                    s=100,
                    marker="X",
                    edgecolors="white",
                )
        else:  # 2D Orthographic Views
            ax = fig.add_subplot(subplot_idx)
            for i, (cluster, centroid) in enumerate(zip(clusters, centroids)):
                color = cmap(i % 10)
                ax.scatter(
                    cluster[:, axes_idx[0]],
                    cluster[:, axes_idx[1]],
                    color=color,
                    s=2,
                    alpha=0.4,
                )
                ax.scatter(
                    centroid[axes_idx[0]],
                    centroid[axes_idx[1]],
                    color="black",
                    s=80,
                    marker="X",
                )
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight="bold")

    plt.suptitle(
        f"Point Cloud Segmentation Analysis - Step {step_num}\n"
        f"Clusters Found: {len(clusters)}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"segmentation_analysis_step_{step_num}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(
        f"{Colors.BLUE}[SAVED] Segmentation visualization saved to: {output_path}{Colors.END}"
    )
    plt.close()


def visualize_obb_segmentation(clusters, obbs, step_num, output_dir=None):
    """
    Visualizes segmented clusters with their Oriented Bounding Boxes in 4 views.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(24, 6))
    cmap = plt.get_cmap("tab10")

    views = [
        (141, None, "3D OBB Tree"),
        (142, [0, 1], "Top (X-Y)"),
        (143, [0, 2], "Front (X-Z)"),
        (144, [1, 2], "Side (Y-Z)"),
    ]

    for subplot_idx, axes_idx, title in views:
        if axes_idx is None:
            ax = fig.add_subplot(subplot_idx, projection="3d")
            for i, (cluster, obb) in enumerate(zip(clusters, obbs)):
                color = cmap(i % 10)
                ax.scatter(
                    cluster[:, 0],
                    cluster[:, 1],
                    cluster[:, 2],
                    color=color,
                    s=1,
                    alpha=0.2,
                )
                # Plot OBB center
                center = obb.center
                ax.scatter(
                    center[0], center[1], center[2], color="black", s=100, marker="X"
                )
        else:
            ax = fig.add_subplot(subplot_idx)
            for i, (cluster, obb) in enumerate(zip(clusters, obbs)):
                color = cmap(i % 10)
                ax.scatter(
                    cluster[:, axes_idx[0]],
                    cluster[:, axes_idx[1]],
                    color=color,
                    s=2,
                    alpha=0.3,
                )
                ax.scatter(
                    obb.center[axes_idx[0]],
                    obb.center[axes_idx[1]],
                    color="black",
                    s=80,
                    marker="X",
                )
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight="bold")

    plt.suptitle(
        f"OBB Tree Decomposition - Step {step_num}", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"obb_decomposition_step_{step_num}.png"), dpi=150
    )
    plt.close()
