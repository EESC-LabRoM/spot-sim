from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def duration_to_s(duration) -> float:
    """Convert builtin_interfaces/Duration to seconds."""
    return duration.sec + duration.nanosec * 1e-9


def parse_pose_stamped(
    msg,
    ee_offset: np.ndarray,
    approach_dist: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a geometry_msgs/PoseStamped into (pre_grasp_pos, grasp_pos, quat_wxyz).

    Applies URDF→world quaternion correction (90° around X), applies ee_offset
    in the grasp frame, then backs off by approach_dist along local X for pre_grasp.

    Returns:
        (pre_grasp_pos, grasp_pos, quat_wxyz) — all (3,) float64; quat_wxyz is (w,x,y,z).
    """
    from scripts.spot_isaacsim.utils.math import advance_along_local_x

    p, q = msg.pose.position, msg.pose.orientation
    q_urdf    = (Rotation.from_quat([q.x, q.y, q.z, q.w]) * Rotation.from_euler("x", np.pi / 2)).as_quat()
    quat_wxyz = np.array([q_urdf[3], q_urdf[0], q_urdf[1], q_urdf[2]])

    grasp_pos = np.array([p.x, p.y, p.z])
    R         = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
    offset_w  = R @ ee_offset
    grasp_pos += offset_w

    pre_grasp_pos = advance_along_local_x(grasp_pos, quat_wxyz, -approach_dist) + offset_w
    return pre_grasp_pos, grasp_pos, quat_wxyz
