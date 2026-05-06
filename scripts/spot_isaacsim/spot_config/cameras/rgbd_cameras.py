"""
RGB camera configuration for Spot robot in Isaac Sim.

Defines RGB camera sensors attached to the robot using dataclass configurations.
Compatible with native Isaac Sim Camera API (isaacsim.sensors.camera.Camera).

Body cameras are parented directly to the `body` link with transforms composed
from the URDF joint chain baked into `translation` and `orientation_rpy`.
This works with both the relic URDF (no camera links) and the extended URDF.
"""

import functools
import numpy as np
from scripts.spot_isaacsim.scene import CameraConfig

# =============================================================================
# Camera mount joint chain (from simulation/assets/spot/spot_with_arm.urdf)
# Used to compose body-relative transforms for each body camera.
# parent=None means the transform is already expressed in the body frame.
# =============================================================================

_CAMERA_MOUNT_JOINTS = {
    "head": {
        "parent": None,
        "xyz": [0.0, 0.0, 0.0],
        "rpy": [0.0, 0.0, 0.0],
    },
    "frontleft": {
        "parent": "head",
        "xyz": [0.41275, 0.03719, 0.02395],
        "rpy": [-2.589351, 1.137527, -3.136510],
    },
    "frontleft_fisheye": {
        "parent": "frontleft",
        "xyz": [0.07825, 0.00035, 0.00200],
        "rpy": [-0.005921, 0.000265, 0.012604],
    },
    "frontright": {
        "parent": "head",
        "xyz": [0.41262, -0.03788, 0.02454],
        "rpy": [2.633190, 1.143761, -3.106119],
    },
    "frontright_fisheye": {
        "parent": "frontright",
        "xyz": [0.07805, 0.00055, 0.00224],
        "rpy": [0.001813, 0.000403, 0.009670],
    },
    "left_fisheye": {
        "parent": None,
        "xyz": [0.0, 0.1, 0.0],
        "rpy": [0.0, 0.0, 1.57],
    },
    "right_fisheye": {
        "parent": None,
        "xyz": [0.0, -0.1, 0.0],
        "rpy": [0.0, 0.0, -1.57],
    },
    "back_fisheye": {
        "parent": None,
        "xyz": [-0.45, 0.0, 0.0],
        "rpy": [0.0, 0.0, 3.14],
    },
}


def _rpy_to_mat3(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _make_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = _rpy_to_mat3(rpy)
    T[:3, 3] = xyz
    return T


@functools.lru_cache(maxsize=None)
def _body_T(name):
    """Recursively compose transforms from body frame to mount link `name`."""
    entry = _CAMERA_MOUNT_JOINTS[name]
    T_local = _make_T(entry["xyz"], entry["rpy"])
    if entry["parent"] is None:
        return T_local
    return _body_T(entry["parent"]) @ T_local


def _mat3_to_rpy(R):
    """Extract ZYX Euler angles (roll, pitch, yaw) from rotation matrix."""
    r = np.arctan2(R[2, 1], R[2, 2])
    p = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    y = np.arctan2(R[1, 0], R[0, 0])
    return (r, p, y)


def _compute_body_cam_transform(mount_name, local_xyz, local_rpy):
    """
    Compute the full body-relative transform for a camera.

    Composes: T_body_mount (from URDF joint chain) @ T_local_offset
    Returns: (translation_tuple, orientation_rpy_tuple)
    """
    T_body_mount = _body_T(mount_name)
    T_local = _make_T(local_xyz, local_rpy)
    T_final = T_body_mount @ T_local
    translation = tuple(T_final[:3, 3].tolist())
    orientation_rpy = _mat3_to_rpy(T_final[:3, :3])
    return translation, orientation_rpy


# =============================================================================
# RGB Camera Configurations
# =============================================================================

# Hand camera mounted on arm wrist link (unchanged — arm_link_wr1 is in relic URDF)
hand_camera_config = CameraConfig(
    name="hand",
    prim_path="arm_link_wr1/hand_cam",
    resolution=(336, 336),
    translation=(0.15, 0.0, 0.02),
    orientation_rpy=(0, 0, 0),
    focal_length=18.0,
)

# Body cameras — parented to `body` link with baked body-relative transforms
_fl_xyz, _fl_rpy = _compute_body_cam_transform("frontleft_fisheye", [0.0, 0.0, 0.01], [0.0, -np.pi/2, 0.0])
frontleft_camera_config = CameraConfig(
    name="frontleft",
    prim_path="body/frontleft_cam",
    resolution=(480, 640),
    translation=_fl_xyz,
    orientation_rpy=_fl_rpy,
    focal_length=18.0,
)

_fr_xyz, _fr_rpy = _compute_body_cam_transform("frontright_fisheye", [0.0, 0.0, 0.01], [0.0, -np.pi/2, 0.0])
frontright_camera_config = CameraConfig(
    name="frontright",
    prim_path="body/frontright_cam",
    resolution=(480, 640),
    translation=_fr_xyz,
    orientation_rpy=_fr_rpy,
    focal_length=18.0,
)

_l_xyz, _l_rpy = _compute_body_cam_transform("left_fisheye", [0.05, 0.0, 0.02], [0.0, 0.0, 0.0])
left_camera_config = CameraConfig(
    name="left",
    prim_path="body/left_cam",
    resolution=(640, 480),
    translation=_l_xyz,
    orientation_rpy=_l_rpy,
    focal_length=18.0,
)

_r_xyz, _r_rpy = _compute_body_cam_transform("right_fisheye", [0.05, 0.0, 0.02], [0.0, 0.0, 0.0])
right_camera_config = CameraConfig(
    name="right",
    prim_path="body/right_cam",
    resolution=(640, 480),
    translation=_r_xyz,
    orientation_rpy=_r_rpy,
    focal_length=18.0,
)

_b_xyz, _b_rpy = _compute_body_cam_transform("back_fisheye", [0.06, 0.0, 0.0], [0.0, 0.0, 0.0])
back_camera_config = CameraConfig(
    name="rear",
    prim_path="body/rear_cam",
    resolution=(640, 480),
    translation=_b_xyz,
    orientation_rpy=_b_rpy,
    focal_length=18.0,
)

# Convenience list of all body cameras
body_camera_configs = [
    frontleft_camera_config,
    frontright_camera_config,
    left_camera_config,
    right_camera_config,
    back_camera_config,
]

# All RGB cameras (hand + body)
all_rgb_camera_configs = [hand_camera_config] + body_camera_configs
