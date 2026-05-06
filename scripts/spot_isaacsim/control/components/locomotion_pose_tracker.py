"""
LocomotionPoseTracker — PID pose control, joint locking, and teleport for SpotLocomotionController.

Owns all navigation state: target pose, PID integrators, lock ramp, and nav phase.
SpotLocomotionController creates one of these and delegates all pose-tracking operations to it.
"""

import math
from math import atan2, cos, hypot, sin

import numpy as np

from scripts.spot_isaacsim.utils.math import normalize_angle

# ---------------------------------------------------------------------------
# Tolerances and PID gains
# ---------------------------------------------------------------------------

_SETPOINT_STABLE_THRESH = 13     # forward() calls at 50 Hz before locking joints (= 0.26 s)
_POSITION_TOL           = 0.15   # metres
_YAW_TOL                = 0.1    # radians

_LOCK_RAMP_DURATION = 0.20   # seconds
_LOCK_MULTIPLIER    = 15.0

_TELEPORT_POS_TOL = 0.03   # metres
_TELEPORT_YAW_TOL = 0.03   # radians (~1.7°)

_LONG_DIST_THRESH = 1.5    # m — steer-then-drive threshold
_STEER_YAW_TOL   = 0.05   # rad

_PID_KP      = np.array([1.5, 1.5, 2.0], dtype=np.float64)
_PID_KI      = np.array([0.0, 0.0, 0.0], dtype=np.float64)
_PID_KD      = np.array([0.2, 0.2, 0.3], dtype=np.float64)
_PID_MAX_VEL = np.array([1.0, 1.0, 4.0], dtype=np.float64)


class LocomotionPoseTracker:
    """PID pose tracking, joint locking, teleport, and navigation phase logic for Spot locomotion."""

    def __init__(self, robot) -> None:
        self._robot = robot

        self._target_pose: np.ndarray | None = None
        self._last_setpoint: np.ndarray | None = None
        self._setpoint_stable_steps: int = 0
        self._pid_integral = np.zeros(3, dtype=np.float64)
        self._pid_prev_error = np.zeros(3, dtype=np.float64)
        self._nav_phase: int = 0   # 0=FREE, 1=STEER, 2=DRIVE

        self._locked: bool = False
        self._ramp_elapsed: float = -1.0

        self._robot_parent_path: str = ""

    def initialize(self, robot_parent_path: str) -> None:
        """Call after world.reset(). Resets lock state and stores USD path for gain writes."""
        self._robot_parent_path = robot_parent_path
        self._locked = False
        self._ramp_elapsed = -1.0

    # ------------------------------------------------------------------
    # Target management
    # ------------------------------------------------------------------

    def set_target(self, x: float, y: float, yaw: float) -> None:
        """Set a new world-frame pose target. Resets PID integrator on new setpoint."""
        new_sp = np.array([x, y, yaw], dtype=np.float64)
        if self._last_setpoint is None or not np.allclose(new_sp, self._last_setpoint, atol=1e-3):
            self._setpoint_stable_steps = 0
            self._last_setpoint = new_sp.copy()
            self._nav_phase = 0
            if self._locked:
                self.set_locked(False)
                print("[LOCO] Setpoint changed — joints unlocked")
            self._pid_integral[:] = 0.0
            self._pid_prev_error[:] = 0.0
        self._target_pose = new_sp

    def clear_target(self) -> None:
        """Clear pose target (switch to velocity mode)."""
        self._target_pose = None

    @property
    def target(self) -> np.ndarray | None:
        return self._target_pose

    # ------------------------------------------------------------------
    # Lock state
    # ------------------------------------------------------------------

    @property
    def is_locked(self) -> bool:
        return self._locked

    def set_locked(self, state: bool) -> None:
        if self._locked == state:
            return
        self._locked = state
        if state:
            self._ramp_elapsed = 0.0
        else:
            self._ramp_elapsed = -1.0
            from spot_config.physics.drive_gains import apply_leg_drive_multiplier
            apply_leg_drive_multiplier(self._robot_parent_path, multiplier=1.0)

    def tick_lock_ramp(self, dt: float) -> None:
        if self._ramp_elapsed < 0.0:
            return
        self._ramp_elapsed = min(self._ramp_elapsed + dt, _LOCK_RAMP_DURATION)
        t = self._ramp_elapsed / _LOCK_RAMP_DURATION
        multiplier = 1.0 + (_LOCK_MULTIPLIER - 1.0) * t
        from spot_config.physics.drive_gains import apply_leg_drive_multiplier
        apply_leg_drive_multiplier(self._robot_parent_path, multiplier, silent=True)
        if self._ramp_elapsed >= _LOCK_RAMP_DURATION:
            self._ramp_elapsed = -1.0
            print(f"[LOCO] Lock ramp complete (×{_LOCK_MULTIPLIER:.1f})")

    def check_lock_trigger(self) -> bool:
        """Return True (and lock) if robot has been stable at goal long enough."""
        pos, _ = self._robot.get_world_pose()
        yaw = self.get_world_yaw()
        tx, ty, tyaw = self._target_pose[0], self._target_pose[1], self._target_pose[2]

        pos_err = np.hypot(tx - float(pos[0]), ty - float(pos[1]))
        yaw_err = abs(normalize_angle(tyaw - yaw))

        if pos_err < _POSITION_TOL and yaw_err < _YAW_TOL:
            self._setpoint_stable_steps += 1
        else:
            self._setpoint_stable_steps = 0

        if self._setpoint_stable_steps >= _SETPOINT_STABLE_THRESH:
            self.set_locked(True)
            print(f"[LOCO] Joints locked (pos_err={pos_err:.3f}m, yaw_err={yaw_err:.3f}rad)")
            return True
        return False

    def check_drift(self) -> bool:
        """Check if locked robot has drifted outside tolerance.

        Returns True = still within tolerance (stay locked).
        Returns False = drifted; also resets lock state so caller falls through to PID.
        """
        if self._target_pose is None:
            return True  # no target — stay locked
        pos, _ = self._robot.get_world_pose()
        yaw = self.get_world_yaw()
        pos_err = np.hypot(
            self._target_pose[0] - float(pos[0]),
            self._target_pose[1] - float(pos[1]),
        )
        yaw_err = abs(normalize_angle(self._target_pose[2] - yaw))
        if pos_err > _POSITION_TOL * 2.0 or yaw_err > _YAW_TOL * 2.0:
            self.set_locked(False)
            self._setpoint_stable_steps = 0
            print(f"[LOCO] Joints unlocked — robot moved out of tolerance "
                  f"(pos_err={pos_err:.3f}m, yaw_err={yaw_err:.3f}rad)")
            return False
        return True

    # ------------------------------------------------------------------
    # PID
    # ------------------------------------------------------------------

    def get_world_yaw(self) -> float:
        _, q = self._robot.get_world_pose()
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def update_pid_command(self, dt: float) -> np.ndarray:
        """Compute vx/vy/wz command using steer-then-drive for long distances, full PID otherwise."""
        pos, _ = self._robot.get_world_pose()
        yaw = self.get_world_yaw()

        tx, ty, tyaw = self._target_pose[0], self._target_pose[1], self._target_pose[2]
        dx = tx - float(pos[0])
        dy = ty - float(pos[1])
        dist = hypot(dx, dy)
        bearing = atan2(dy, dx)
        bearing_err = normalize_angle(bearing - yaw)

        if dist > _LONG_DIST_THRESH:
            if self._nav_phase == 0:
                self._nav_phase = 1
                print(f"[LOCO] Long distance ({dist:.1f}m) — STEER phase")
            elif self._nav_phase == 1 and abs(bearing_err) < _STEER_YAW_TOL:
                self._nav_phase = 2
                print("[LOCO] Bearing aligned — DRIVE phase")
        else:
            if self._nav_phase in (1, 2):
                self._nav_phase = 0
                print(f"[LOCO] Within range ({dist:.1f}m) — FREE phase")

        if self._nav_phase == 1:
            wz = float(np.clip(_PID_KP[2] * bearing_err, -_PID_MAX_VEL[2], _PID_MAX_VEL[2]))
            return np.array([0.0, 0.0, wz], dtype=np.float32)

        if self._nav_phase == 2:
            vx = float(np.clip(_PID_KP[0] * dist, 0.0, _PID_MAX_VEL[0]))
            wz = float(np.clip(_PID_KP[2] * bearing_err, -_PID_MAX_VEL[2], _PID_MAX_VEL[2]))
            return np.array([vx, 0.0, wz], dtype=np.float32)

        # FREE: full 3-DOF PID
        error_bx =  dx * cos(yaw) + dy * sin(yaw)
        error_by = -dx * sin(yaw) + dy * cos(yaw)
        error_yaw = normalize_angle(tyaw - yaw)

        error = np.array([error_bx, error_by, error_yaw], dtype=np.float64)
        self._pid_integral += error * dt
        derivative = (error - self._pid_prev_error) / dt if dt > 0 else np.zeros(3)
        self._pid_prev_error = error.copy()

        cmd = _PID_KP * error + _PID_KI * self._pid_integral + _PID_KD * derivative
        return np.clip(cmd, -_PID_MAX_VEL, _PID_MAX_VEL).astype(np.float32)

    # ------------------------------------------------------------------
    # Teleport
    # ------------------------------------------------------------------

    def teleport_to(self, x: float, y: float, z: float, yaw: float) -> None:
        half = yaw / 2.0
        self._robot.set_world_pose(
            position=np.array([x, y, z], dtype=np.float64),
            orientation=np.array([math.cos(half), 0.0, 0.0, math.sin(half)]),
        )
        self._robot.set_linear_velocity(np.zeros(3))
        self._robot.set_angular_velocity(np.zeros(3))
        self._robot.set_joint_velocities(np.zeros(self._robot.num_dof))

    # ------------------------------------------------------------------
    # Keyboard nudge
    # ------------------------------------------------------------------

    def nudge(self, kb_vel: np.ndarray) -> None:
        """Advance _target_pose by kb_vel (body frame) in world frame."""
        if self._target_pose is None:
            pos, _ = self._robot.get_world_pose()
            robot_yaw = self.get_world_yaw()
            self._target_pose = np.array(
                [float(pos[0]), float(pos[1]), robot_yaw], dtype=np.float64
            )
            self._last_setpoint = self._target_pose.copy()

        cur_x, cur_y, cur_yaw_t = self._target_pose
        robot_yaw = self.get_world_yaw()
        dx_b, dy_b, dyaw = kb_vel

        dx_w = dx_b * cos(robot_yaw) - dy_b * sin(robot_yaw)
        dy_w = dx_b * sin(robot_yaw) + dy_b * cos(robot_yaw)

        new_pose = np.array(
            [cur_x + dx_w, cur_y + dy_w, normalize_angle(cur_yaw_t + dyaw)],
            dtype=np.float64,
        )
        self._target_pose = new_pose
        self._last_setpoint = new_pose.copy()
        self._setpoint_stable_steps = 0
        self._nav_phase = 0
        if self._locked:
            self.set_locked(False)
