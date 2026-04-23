"""
Camera setup utilities for IsaacLab dataset extraction.

API overview
------------
IsaacLab exposes two pose-setting methods on every Camera / TiledCamera:

    camera.set_world_poses(positions, orientations, env_ids, convention)
        Set pose from (N,3) positions and (N,4) quaternions [w,x,y,z].
        Supported conventions: "opengl" | "ros" | "world".

    camera.set_world_poses_from_view(eyes, targets, env_ids)
        Set pose from (N,3) eye positions and (N,3) look-at targets.
        This is the direct IsaacLab equivalent of Genesis's
        `cam.set_pose(pos, lookat)`.

Both methods exist on Camera AND TiledCamera and update the prim transform
immediately.  After calling them you should call `sim.render()` (and optionally
`camera.update(dt=0, force_recompute=True)`) before reading new data.

Genesis → IsaacLab mapping
---------------------------
Genesis:
    cam.set_pose(pos=(x,y,z), lookat=(lx,ly,lz))

IsaacLab (this module):
    set_camera_pose(camera, pos, lookat, env_origins, env_ids, sim)

The function handles:
  • env-local → world frame translation via env_origins
  • batched (N environments) and single-env inputs
  • triggering sim.render() + camera.update() so data is fresh after the call
  • a pure-math fallback (get_rotation_to_lookat) when only a quaternion is needed
"""

from __future__ import annotations

from typing import Sequence

import torch

# Isaac Lab math utilities
from isaaclab.utils.math import quat_from_matrix

# Low-level math helper (retained and hardened from original implementation


def get_rotation_to_lookat(
    camera_pos: torch.Tensor | tuple | list,
    target_pos: torch.Tensor | tuple | list,
    up_axis: str = "z",
) -> torch.Tensor:
    """Compute the quaternion [w, x, y, z] that orients a camera toward a target.

    Follows the OpenGL / USD camera convention where the camera looks along its
    local **-Z** axis and **+Y** is up.  The returned quaternion can be passed
    directly to ``camera.set_world_poses(..., convention="opengl")``.

    Args:
        camera_pos: Eye position.  Shape ``(3,)`` or ``(N, 3)``.
        target_pos: Look-at point.  Shape ``(3,)`` or ``(N, 3)``.
        up_axis:    Global up direction – ``"z"`` (default) or ``"y"``.

    Returns:
        Quaternion tensor ``[w, x, y, z]``.
        Shape ``(4,)`` for single inputs, ``(N, 4)`` for batched inputs.

    Notes:
        A degenerate case occurs when the view direction is parallel to the
        global up vector (e.g. camera directly above or below the target).
        In this situation the right-vector cross product degenerates to zero,
        so we fall back to a secondary up vector ``+X`` automatically.
    """
    pos = torch.as_tensor(camera_pos, dtype=torch.float32)
    target = torch.as_tensor(target_pos, dtype=torch.float32)

    # local +Z points *from target to camera* (camera looks along -Z)
    z_axis = torch.nn.functional.normalize(pos - target, dim=-1)

    # primary global up
    if up_axis == "z":
        global_up = torch.tensor([0.0, 0.0, 1.0], device=pos.device)
    else:
        global_up = torch.tensor([0.0, 1.0, 0.0], device=pos.device)

    # detect degeneracy: |z_axis · global_up| ≈ 1
    # broadcast-safe dot product for both (3,) and (N,3) inputs
    dot = (z_axis * global_up).sum(dim=-1, keepdim=True).abs()  # (..., 1)
    is_degenerate = dot > 0.9999

    fallback_up = torch.tensor([1.0, 0.0, 0.0], device=pos.device)

    # right vector:  x = up × z
    x_axis = torch.nn.functional.normalize(
        torch.linalg.cross(global_up, z_axis), dim=-1
    )
    x_axis_fallback = torch.nn.functional.normalize(
        torch.linalg.cross(fallback_up, z_axis), dim=-1
    )

    if z_axis.ndim == 1:
        # single-vector case
        if is_degenerate.item():
            x_axis = x_axis_fallback
    else:
        # batched case
        x_axis = torch.where(is_degenerate, x_axis_fallback, x_axis)

    # actual up:  y = z × x
    y_axis = torch.nn.functional.normalize(torch.linalg.cross(z_axis, x_axis), dim=-1)

    # rotation matrix whose columns are [x, y, z]
    rot_matrix = torch.stack([x_axis, y_axis, z_axis], dim=-1)

    if rot_matrix.ndim == 2:  # (3,3) → add batch dim
        rot_matrix = rot_matrix.unsqueeze(0)

    quat = quat_from_matrix(rot_matrix)  # [w, x, y, z]
    return quat.squeeze(0)  # (4,) or (N,4)


# Genesis-style set_pose  →  IsaacLab adaptatio


def set_camera_pose(
    camera,
    pos: torch.Tensor | tuple | list,
    lookat: torch.Tensor | tuple | list,
    env_origins: torch.Tensor | None = None,
    env_ids: Sequence[int] | None = None,
    sim=None,
    force_recompute: bool = True,
) -> None:
    """Move an IsaacLab camera to *pos* and point it at *lookat*.

    This is the direct IsaacLab adaptation of Genesis's::

        cam.set_pose(pos=(x,y,z), lookat=(lx,ly,lz))

    It wraps ``camera.set_world_poses_from_view`` (the native IsaacLab API)
    and additionally handles:

    * **env-local → world** translation when ``env_origins`` is provided.
    * **Batched** inputs  ``(N, 3)`` for updating multiple environments at once.
    * **Single-env** scalar inputs  ``(3,)`` broadcast to all envs (or ``env_ids``).
    * **Renderer flush** via ``sim.render()`` + ``camera.update()`` so the very
      next ``camera.data.output[...]`` access returns data from the new pose.

    Args:
        camera:
            An ``isaaclab.sensors.Camera`` or ``isaaclab.sensors.TiledCamera``
            instance that has already been initialised (i.e. the scene has been
            built).
        pos:
            Camera eye position.  Either:

            * ``(3,)``   – same position applied to all ``env_ids``
            * ``(N, 3)`` – one position per environment in ``env_ids``

            Coordinates are interpreted as **env-local** when ``env_origins`` is
            given, and as **world-frame** otherwise.
        lookat:
            Look-at target point.  Same shape rules as ``pos``.
        env_origins:
            Grid origins of each environment in world frame, shape ``(num_envs, 3)``.
            Typically ``env.scene.env_origins``.  Pass ``None`` if you are already
            working in world frame.
        env_ids:
            Which environment indices to update.  ``None`` updates all environments.
        sim:
            The ``SimulationContext`` instance (``env.sim``).  When provided the
            renderer is flushed after moving the camera so that the updated image
            is immediately available.
        force_recompute:
            Passed to ``camera.update(dt=0, force_recompute=...)`` after the
            renderer flush.  Defaults to ``True``.

    Example – single view, env-local coordinates::

        set_camera_pose(
            camera     = env.scene["camera"],
            pos        = (1.5, 0.0, 1.2),
            lookat     = (0.0, 0.0, 0.5),
            env_origins= env.scene.env_origins,
            sim        = env.sim,
        )

    Example – multi-view loop (Genesis-style)::

        import numpy as np
        for i, (p, t) in enumerate(VIEW_POSES):
            set_camera_pose(camera, pos=p, lookat=t,
                            env_origins=env.scene.env_origins, sim=env.sim)
            data = camera.data.output["rgb"]
            dataset_saver.save_view_data(..., view_id=i)

    Notes:
        ``set_world_poses_from_view`` is available on both ``Camera`` and
        ``TiledCamera``.  If for some reason your IsaacLab build is older and
        lacks that method the function falls back to ``set_world_poses`` using
        the quaternion computed by :func:`get_rotation_to_lookat`.

        After calling this function the camera prim is immediately repositioned
        in the USD stage.  However, rendered pixel data is only refreshed once
        the renderer pipeline has been flushed (hence the ``sim.render()`` call).
    """
    # 1. Normalise inputs to torch float32 tensors                        #
    device = env_origins.device if env_origins is not None else torch.device("cpu")
    eyes = torch.as_tensor(pos, dtype=torch.float32, device=device)
    targets = torch.as_tensor(lookat, dtype=torch.float32, device=device)

    # 2. Resolve env_ids and determine number of cameras to update        #
    # Try to get total number of environments from the camera object
    try:
        num_envs_total = camera.num_instances
    except AttributeError:
        # Fallback: infer from env_origins if available
        num_envs_total = env_origins.shape[0] if env_origins is not None else 1

    if env_ids is None:
        env_ids_tensor = torch.arange(
            num_envs_total, dtype=torch.long, device=eyes.device
        )
    else:
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=eyes.device)

    n_update = len(env_ids_tensor)

    # 3. Broadcast single (3,) input to all target environments           #
    if eyes.ndim == 1:  # (3,) → (N, 3)
        eyes = eyes.unsqueeze(0).expand(n_update, -1).clone()
        targets = targets.unsqueeze(0).expand(n_update, -1).clone()

    # 4. Convert env-local → world frame (matching GraspAction convention) #
    if env_origins is not None:
        origins = env_origins[env_ids_tensor]  # (N, 3) – selected rows only
        eyes = eyes + origins
        targets = targets + origins

    # 5. Move the camera                                                  #
    # Prefer the native look-at API (Camera + TiledCamera both have it)
    if hasattr(camera, "set_world_poses_from_view"):
        camera.set_world_poses_from_view(
            eyes=eyes,
            targets=targets,
            env_ids=env_ids,  # pass original env_ids (None = all)
        )
    else:
        # ── Fallback: compute quaternion manually and use set_world_poses ──
        # This path is a safety net for very old IsaacLab builds.
        quat = get_rotation_to_lookat(eyes, targets, up_axis="z")  # (N, 4)
        if quat.ndim == 1:
            quat = quat.unsqueeze(0)
        camera.set_world_poses(
            positions=eyes,
            orientations=quat,
            env_ids=env_ids,
            convention="opengl",
        )

    # 6. Flush renderer so data is ready on the next read                 #
    if sim is not None:
        sim.render()
        camera.update(dt=0.0, force_recompute=force_recompute)


# Multi-view schedule helper


def build_orbit_views(
    center: tuple | list,
    radius: float,
    n_views: int,
    height: float | None = None,
    elevation_deg: float = 30.0,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Generate a horizontal orbit of camera poses around *center*.

    Each element of the returned list is ``(pos, lookat)`` in the same
    env-local coordinate frame as *center* – ready to pass to
    :func:`set_camera_pose`.

    Args:
        center:        Look-at target, env-local frame, e.g. object CoM.
        radius:        Orbit radius in metres.
        n_views:       Number of equally-spaced views around the circle.
        height:        Camera height above *center*.  When ``None``, height is
                       derived from *elevation_deg*.
        elevation_deg: Elevation angle in degrees (ignored if *height* given).
                       Default 30°.

    Returns:
        List of ``(pos, lookat)`` tuples, length == *n_views*.

    Example::

        views = build_orbit_views(center=(0, 0, 0.3), radius=1.2,
                                  n_views=8, elevation_deg=30)
        for view_id, (pos, lookat) in enumerate(views):
            set_camera_pose(camera, pos=pos, lookat=lookat,
                            env_origins=env.scene.env_origins, sim=env.sim)
            ...  # collect data
    """
    import math

    cx, cy, cz = center
    if height is None:
        height = radius * math.tan(math.radians(elevation_deg))

    views = []
    for i in range(n_views):
        angle = 2.0 * math.pi * i / n_views
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        z = cz + height
        views.append(((x, y, z), (cx, cy, cz)))
    return views


def build_hemisphere_views(
    center: tuple | list,
    radius: float,
    n_azimuth: int = 8,
    n_elevation: int = 3,
    min_elevation_deg: float = 10.0,
    max_elevation_deg: float = 70.0,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Generate a hemisphere grid of camera poses around *center*.

    Produces a grid of ``n_azimuth × n_elevation`` viewpoints evenly spread
    over the upper hemisphere, giving good all-around coverage of an object
    for dataset generation.

    Args:
        center:            Look-at target (env-local).
        radius:            Distance from *center* to camera.
        n_azimuth:         Number of azimuth steps (longitude).  Default 8.
        n_elevation:       Number of elevation steps (latitude).  Default 3.
        min_elevation_deg: Minimum elevation angle in degrees.  Default 10°.
        max_elevation_deg: Maximum elevation angle in degrees.  Default 70°.

    Returns:
        List of ``(pos, lookat)`` tuples, length == ``n_azimuth × n_elevation``.
    """
    import math

    cx, cy, cz = center
    views = []

    elevations = [
        min_elevation_deg
        + (max_elevation_deg - min_elevation_deg) * i / max(n_elevation - 1, 1)
        for i in range(n_elevation)
    ]

    for elev_deg in elevations:
        elev_rad = math.radians(elev_deg)
        for j in range(n_azimuth):
            azim_rad = 2.0 * math.pi * j / n_azimuth
            x = cx + radius * math.cos(elev_rad) * math.cos(azim_rad)
            y = cy + radius * math.cos(elev_rad) * math.sin(azim_rad)
            z = cz + radius * math.sin(elev_rad)
            views.append(((x, y, z), (cx, cy, cz)))

    return views
