import math
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def advance_along_local_x(pos: np.ndarray, quat_wxyz: np.ndarray, distance: float) -> np.ndarray:
    """Translate pos by distance along the pose's local X-axis (wxyz quaternion)."""
    R = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return pos + distance * R.as_matrix()[:, 0]


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw (rotation about Z) from a quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(a: float) -> float:
    """Wrap angle to (-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def rpy_deg_to_quat(rpy_deg: Tuple[float, float, float]) -> np.ndarray:
    """Convert RPY (roll, pitch, yaw) in degrees to quaternion (w, x, y, z). ZYX extrinsic."""
    r, p, y = np.radians(rpy_deg)
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def euclidean_distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 3D points."""
    return float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)))


def euclidean_distance_2d(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 2D (or 3D, XY only) points."""
    return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))


def bearing_from_positions(from_pos: np.ndarray, to_pos: np.ndarray) -> float:
    """Bearing angle (radians) from from_pos to to_pos in the XY plane."""
    return math.atan2(float(to_pos[1]) - float(from_pos[1]), float(to_pos[0]) - float(from_pos[0]))
