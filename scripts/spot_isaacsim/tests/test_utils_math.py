import math
import numpy as np
from scripts.spot_isaacsim.utils.math import (
    advance_along_local_x,
    yaw_from_quaternion,
    normalize_angle,
    rpy_deg_to_quat,
    euclidean_distance_3d,
    euclidean_distance_2d,
    bearing_from_positions,
)


def test_advance_along_local_x_identity():
    pos = np.array([0.0, 0.0, 0.0])
    quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # identity
    result = advance_along_local_x(pos, quat_wxyz, 1.0)
    np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-6)


def test_advance_along_local_x_negative():
    pos = np.array([5.0, 0.0, 0.0])
    quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    result = advance_along_local_x(pos, quat_wxyz, -0.15)
    np.testing.assert_allclose(result, [4.85, 0.0, 0.0], atol=1e-6)


def test_yaw_from_quaternion_identity():
    assert abs(yaw_from_quaternion(0.0, 0.0, 0.0, 1.0)) < 1e-6


def test_yaw_from_quaternion_90deg():
    # 90° rotation around Z: qz = sin(π/4), qw = cos(π/4)
    s = math.sin(math.pi / 4)
    result = yaw_from_quaternion(0.0, 0.0, s, s)
    assert abs(result - math.pi / 2) < 1e-6


def test_normalize_angle_positive():
    assert abs(normalize_angle(0.0)) < 1e-9
    assert abs(normalize_angle(math.pi - 0.001) - (math.pi - 0.001)) < 1e-9


def test_normalize_angle_wraps():
    assert abs(normalize_angle(math.pi + 0.1) - (-math.pi + 0.1)) < 1e-6
    assert abs(normalize_angle(-math.pi - 0.1) - (math.pi - 0.1)) < 1e-6


def test_rpy_deg_to_quat_identity():
    q = rpy_deg_to_quat((0.0, 0.0, 0.0))
    np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_rpy_deg_to_quat_90_yaw():
    q = rpy_deg_to_quat((0.0, 0.0, 90.0))
    s = math.sin(math.pi / 4)
    np.testing.assert_allclose(q, [s, 0.0, 0.0, s], atol=1e-6)


def test_euclidean_distance_3d():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([3.0, 4.0, 0.0])
    assert abs(euclidean_distance_3d(p1, p2) - 5.0) < 1e-6


def test_euclidean_distance_2d():
    p1 = np.array([0.0, 0.0])
    p2 = np.array([3.0, 4.0])
    assert abs(euclidean_distance_2d(p1, p2) - 5.0) < 1e-6


def test_bearing_from_positions():
    from_pos = np.array([0.0, 0.0])
    to_pos   = np.array([1.0, 0.0])
    assert abs(bearing_from_positions(from_pos, to_pos) - 0.0) < 1e-6

    to_pos_north = np.array([0.0, 1.0])
    assert abs(bearing_from_positions(from_pos, to_pos_north) - math.pi / 2) < 1e-6
