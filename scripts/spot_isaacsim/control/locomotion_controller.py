"""
Spot Locomotion Controller using the IsaacRobotics 12-DOF spot_policy.pt.

The policy observes only the 12 leg joints and outputs position targets for them.
Arm joints are controlled independently by the Lula IK controller.

Relic 19-DOF robot leg indices → policy 12-DOF indices (same joint order):
  Relic idx  name      Policy idx
         1   fl_hx         0
         2   fr_hx         1
         3   hl_hx         2
         4   hr_hx         3
         6   fl_hy         4
         7   fr_hy         5
         8   hl_hy         6
         9   hr_hy         7
        11   fl_kn         8
        12   fr_kn         9
        13   hl_kn        10
        14   hr_kn        11

Default positions (from env.yaml / SPOT_STANDING_JOINT_POSITIONS):
  fl_hx=+0.10, fr_hx=-0.10, hl_hx=+0.10, hr_hx=-0.10
  f*_hy=0.90, h*_hy=1.10, *_kn=-1.503

Keyboard controls — move the position setpoint in robot body frame:
    UP/DOWN    — advance/retreat target pose (robot-forward direction)
    LEFT/RIGHT — strafe target pose (robot-lateral direction)
    N/M        — rotate yaw setpoint left/right

PID position/yaw control:
    set_target_pose(x, y, yaw) — drive to a world-frame pose
    PID gains are module-level constants (_PID_KP, _PID_KI, _PID_KD).

Joint locking:
    When the setpoint is unchanged for _SETPOINT_STABLE_THRESH steps and the
    robot is within position/yaw tolerance, the policy is bypassed and joints
    are locked at _DEFAULT_LEG_POS.  Any setpoint change unlocks immediately.
"""

import sys
from math import atan2, cos, hypot, sin
from pathlib import Path

_SPOT_ISAACSIM = Path(__file__).parent.parent
if str(_SPOT_ISAACSIM) not in sys.path:
    sys.path.insert(0, str(_SPOT_ISAACSIM))

import numpy as np
import omni
import torch
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from spot_config.robot import SPOT_STANDING_JOINT_POSITIONS

_POLICY_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "external" / "IsaacRobotics" / "policies"
)
_POLICY_PATH  = _POLICY_DIR / "spot/models/spot_policy.pt"
_ENV_YAML_PATH = _POLICY_DIR / "spot/params/env.yaml"

# ---------------------------------------------------------------------------
# DOF mapping: Relic 19-DOF robot → 12-DOF policy (leg joints only)
# ---------------------------------------------------------------------------
_RELIC_LEG_INDICES = np.array(
    [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14], dtype=np.int32
)

_POLICY_LEG_NAMES = [
    "fl_hx", "fr_hx", "hl_hx", "hr_hx",
    "fl_hy", "fr_hy", "hl_hy", "hr_hy",
    "fl_kn", "fr_kn", "hl_kn", "hr_kn",
]

_DEFAULT_LEG_POS = np.array(
    [SPOT_STANDING_JOINT_POSITIONS[n] for n in _POLICY_LEG_NAMES],
    dtype=np.float32,
) * 1.1

# ---------------------------------------------------------------------------
# Joint-locking parameters
# ---------------------------------------------------------------------------
_SETPOINT_STABLE_THRESH = 13     # forward() calls at 50 Hz before locking joints (= 0.26 s)
_POSITION_TOL           = 0.15  # metres — position tolerance to trigger lock
_YAW_TOL                = 0.1  # radians — yaw tolerance to trigger lock

# ---------------------------------------------------------------------------
# Long-distance navigation: steer-then-drive
# ---------------------------------------------------------------------------
_LONG_DIST_THRESH = 1.0    # m — above this, steer to face target before driving
_STEER_YAW_TOL   = 0.05   # rad — bearing alignment tolerance to switch to DRIVE (~8°)

# ---------------------------------------------------------------------------
# PID gains: [x_body, y_body, yaw]
# ---------------------------------------------------------------------------
_PID_KP      = np.array([1.5, 1.5, 2.0], dtype=np.float64)
_PID_KI      = np.array([0.0, 0.0, 0.0], dtype=np.float64)
_PID_KD      = np.array([0.2, 0.2, 0.3], dtype=np.float64)
_PID_MAX_VEL = np.array([1.0, 1.0, 4.0], dtype=np.float64)  # [vx, vy, wz]

# Keyboard pose-setpoint rates (body-frame, per forward() call at 50 Hz)
# 0.04 m/call × 50 Hz = 2.0 m/s — matches _PID_MAX_VEL
from scripts.spot_isaacsim.control.keyboard_controller import KEY_POSE_MAP as _KEY_POSE_MAP


def _normalize_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    import math
    return (a + math.pi) % (2 * math.pi) - math.pi


class SpotLocomotionController:
    """Locomotion for Spot using the 12-DOF spot_policy.pt, based on the IsaacSim Policy/Quadruped example.

    The policy observes and controls only the 12 leg joints.
    Arm joints are left for the IK controller.

    Supports two control modes:
    - Velocity mode: set_command(vx, vy, wz) or keyboard
    - Pose mode: set_target_pose(x, y, yaw) — PID drives to world-frame pose
    """

    def __init__(
        self,
        robot,
        physics_dt: float = 1.0 / 500.0,
        collision_avoidance: bool = True,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            robot: SingleArticulation — the 19-DOF Relic Spot robot.
            physics_dt: Physics timestep (1/500). forward() is called at 50 Hz by the caller
                (every decimation=10 physics steps), so dt passed to forward() is 1/50 s.
            collision_avoidance: Enable depth-camera collision avoidance filter.
                Requires cameras to be passed via set_cameras() after construction.
        """
        self._robot = robot
        self._physics_dt = physics_dt
        self._device = device
        self._action_scale = 0.2       # matches SpotFlatTerrainPolicy training
        self._command = np.zeros(3)
        self._action = np.zeros(12)
        self._previous_action = np.zeros(12)

        # --- PID / pose control state ---
        self._target_pose: np.ndarray | None = None   # [x, y, yaw] world frame
        self._last_setpoint: np.ndarray | None = None
        self._setpoint_stable_steps: int = 0
        self._pid_integral = np.zeros(3, dtype=np.float64)
        self._pid_prev_error = np.zeros(3, dtype=np.float64)

        # --- Keyboard pose nudge state ---
        # Tracked as a set of currently-held keys so that timer races (release
        # firing twice or press+release out of order) cannot corrupt the velocity.
        self._kb_held: set[str] = set()
        self._kb_vel = np.zeros(3, dtype=np.float64)  # recomputed from _kb_held each call

        # --- Navigation phase (steer-then-drive for long distances) ---
        self._nav_phase: int = 0   # 0=FREE, 1=STEER, 2=DRIVE

        # --- Joint lock state ---
        self._locked: bool = False

        # --- Collision avoidance filter (optional) ---
        self._collision_avoidance_enabled: bool = collision_avoidance
        self._collision_avoidance = None

        if robot.num_dof != 19:
            raise RuntimeError(
                f"[LOCO] Expected 19-DOF robot, got {robot.num_dof}. "
                f"Verify spot_with_arm URDF is loaded."
            )

        if not _POLICY_PATH.exists():
            raise RuntimeError(f"[LOCO] Policy not found: {_POLICY_PATH}")
        self._policy = torch.jit.load(str(_POLICY_PATH)).to(device)
        self._policy.eval()
        print(f"[LOCO] Loaded policy: {_POLICY_PATH.name} (12-DOF legs only)")

        self._decimation = self._parse_decimation()
        print(f"[LOCO] Decimation: {self._decimation}  "
              f"(forward() called at {1.0 / (self._decimation * physics_dt):.0f} Hz, "
              f"policy dt = {self._decimation * physics_dt:.4f}s)")


    def _parse_decimation(self) -> int:
        """Extract decimation from env.yaml using regex (file uses !!python/tuple tags
        that are incompatible with yaml.safe_load)."""
        import re
        if not _ENV_YAML_PATH.exists():
            raise RuntimeError(f"[LOCO] env.yaml not found: {_ENV_YAML_PATH}")
        text = _ENV_YAML_PATH.read_text()
        match = re.search(r"^decimation:\s*(\d+)", text, re.MULTILINE)
        if not match:
            raise RuntimeError(
                f"[LOCO] 'decimation' key not found in {_ENV_YAML_PATH}"
            )
        return int(match.group(1))

    @property
    def decimation(self) -> int:
        """Physics steps per forward() call (read from env.yaml)."""
        return self._decimation

    def initialize(self) -> None:
        """Call after world.reset() and apply_all_physics(). Reads drive gains from USD."""
        import math
        import omni.usd
        from pxr import Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        articulation_root = stage.GetPrimAtPath(self._robot.prim_path)
        robot_prim = articulation_root.GetParent()
        self._robot_parent_path: str = str(robot_prim.GetPath())

        # Ensure lock state and gains are reset to defaults on (re-)initialization
        self._locked = False

        # Build name→prim map for all joints
        dof_names = list(self._robot.dof_names)
        joint_prim_map = {}
        for prim in Usd.PrimRange(robot_prim):
            if prim.GetName() in dof_names:
                joint_prim_map[prim.GetName()] = prim

        # Read actual drive gains from USD for effort computation.
        # USD stores angular drives in N·m/deg; convert to N·m/rad.
        deg_to_rad = 180.0 / math.pi  # multiply USD value by this to get N·m/rad
        self._stiffness_rad = np.zeros(19, dtype=np.float64)
        self._damping_rad = np.zeros(19, dtype=np.float64)
        for i, name in enumerate(dof_names):
            prim = joint_prim_map.get(name)
            if prim is None:
                continue
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive:
                continue
            stiff_deg = drive.GetStiffnessAttr().Get()
            damp_deg = drive.GetDampingAttr().Get()
            self._stiffness_rad[i] = stiff_deg * deg_to_rad
            self._damping_rad[i] = damp_deg * deg_to_rad

        # Track last applied drive targets (Relic DOF order) for effort computation
        # Initialize to standing pose (in Relic order)
        self._last_targets_relic = self._robot.get_joint_positions().copy()

        print("[LOCO] Initialization complete")

    # ------------------------------------------------------------------
    # Public control interface
    # ------------------------------------------------------------------

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        """Set velocity command directly (velocity mode — disables PID)."""
        self._target_pose = None
        self._command = np.array([vx, vy, wz])

    def set_target_pose(self, x: float, y: float, yaw: float) -> None:
        """Set world-frame position and yaw target (PID mode).

        The PID controller will compute vx/vy/wz commands each step.
        Once the setpoint is stable for _SETPOINT_STABLE_THRESH steps
        and the robot is within tolerance, joints are locked at DEFAULT_LEG_POS.
        """
        new_sp = np.array([x, y, yaw], dtype=np.float64)
        if self._last_setpoint is None or not np.allclose(new_sp, self._last_setpoint, atol=1e-3):
            self._setpoint_stable_steps = 0
            self._last_setpoint = new_sp.copy()
            self._nav_phase = 0
            if self._locked:
                self._set_locked(False)
                print("[LOCO] Setpoint changed — joints unlocked")
            # Reset PID integrator on new target
            self._pid_integral[:] = 0.0
            self._pid_prev_error[:] = 0.0
        self._target_pose = new_sp

    def set_cameras(self, cameras: dict) -> None:
        """Provide depth cameras for collision avoidance.

        Call after construction (cameras are created after the controller).
        If collision_avoidance=True was passed to __init__, this instantiates
        the filter automatically.
        """
        if self._collision_avoidance_enabled and cameras:
            from .locomotion_collision_avoidance import LocomotionCollisionAvoidance
            self._collision_avoidance = LocomotionCollisionAvoidance(
                cameras=cameras,
                obstacle_distance_m=0.30,
                device=self._device,
            )
        elif not self._collision_avoidance_enabled:
            print("[LOCO] Collision avoidance disabled")

    @property
    def command(self) -> np.ndarray:
        return self._command.copy()

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def forward(self, dt: float) -> None:
        """Run one locomotion update. Called at 50 Hz (every decimation physics steps).

        dt should be decimation * physics_dt (= 1/50 s) so the PID derivative is correct.
        """
        # Keyboard advances the pose setpoint in body-relative coordinates
        if np.any(self._kb_vel != 0.0):
            self._nudge_target_pose()

        # Joints locked — hold DEFAULT_LEG_POS, bypass policy
        if self._locked:
            # Drift check every call (50 Hz); unlock if robot has moved out of tolerance
            if self._target_pose is not None:
                pos, _ = self._robot.get_world_pose()
                yaw = self._get_world_yaw()
                pos_err = np.hypot(
                    self._target_pose[0] - float(pos[0]),
                    self._target_pose[1] - float(pos[1]),
                )
                yaw_err = abs(_normalize_angle(self._target_pose[2] - yaw))
                if pos_err > _POSITION_TOL*2.0 or yaw_err > _YAW_TOL*2.0:
                    self._set_locked(False)
                    self._setpoint_stable_steps = 0
                    print(f"[LOCO] Joints unlocked — robot moved out of tolerance "
                          f"(pos_err={pos_err:.3f}m, yaw_err={yaw_err:.3f}rad)")
                    # Fall through to PID / policy below
                else:
                    self._robot.apply_action(ArticulationAction(
                        joint_positions=_DEFAULT_LEG_POS,
                        joint_indices=_RELIC_LEG_INDICES,
                    ))
                    return
            else:
                self._robot.apply_action(ArticulationAction(
                    joint_positions=_DEFAULT_LEG_POS,
                    joint_indices=_RELIC_LEG_INDICES,
                ))
                return

        # PID update
        if self._target_pose is not None:
            self._update_pid_command(dt)
            self._check_lock_trigger()
            if self._locked:  # just transitioned to lock this step — skip policy
                self._robot.apply_action(ArticulationAction(
                    joint_positions=_DEFAULT_LEG_POS,
                    joint_indices=_RELIC_LEG_INDICES,
                ))
                return

        # Collision avoidance — filter _command before building observation
        if self._collision_avoidance is not None:
            self._command = self._collision_avoidance.apply(self._command)

        # RL policy
        obs = self._compute_observation()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).view(1, -1).float().to(self._device)
            self._action = self._policy(obs_t).detach().cpu().view(-1).numpy()
        self._previous_action = self._action.copy()

        leg_targets = _DEFAULT_LEG_POS + (self._action * self._action_scale)
        self._robot.apply_action(ArticulationAction(
            joint_positions=leg_targets,
            joint_indices=_RELIC_LEG_INDICES,
        ))
        self._last_targets_relic[_RELIC_LEG_INDICES] = leg_targets

    # ------------------------------------------------------------------
    # PID implementation
    # ------------------------------------------------------------------

    def _get_world_yaw(self) -> float:
        """Extract yaw from robot world quaternion (w, x, y, z)."""
        _, q = self._robot.get_world_pose()
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _update_pid_command(self, dt: float) -> None:
        """Compute vx/vy/wz command.

        For dist > _LONG_DIST_THRESH uses a steer-then-drive state machine:
          STEER (1): rotate to face the target bearing, no translation.
          DRIVE (2): drive forward while tracking bearing.
        For dist <= _LONG_DIST_THRESH falls back to the full 3-DOF PID (FREE).
        """
        pos, _ = self._robot.get_world_pose()
        yaw = self._get_world_yaw()

        tx, ty, tyaw = self._target_pose[0], self._target_pose[1], self._target_pose[2]
        dx = tx - float(pos[0])
        dy = ty - float(pos[1])
        dist = hypot(dx, dy)
        bearing = atan2(dy, dx)
        bearing_err = _normalize_angle(bearing - yaw)

        # --- Navigation phase transitions ---
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

        # --- Command per phase ---
        if self._nav_phase == 1:   # STEER: rotate only
            wz = float(np.clip(_PID_KP[2] * bearing_err, -_PID_MAX_VEL[2], _PID_MAX_VEL[2]))
            self._command = np.array([0.0, 0.0, wz], dtype=np.float32)

        elif self._nav_phase == 2:  # DRIVE: forward + bearing correction
            vx = float(np.clip(_PID_KP[0] * dist, 0.0, _PID_MAX_VEL[0]))
            wz = float(np.clip(_PID_KP[2] * bearing_err, -_PID_MAX_VEL[2], _PID_MAX_VEL[2]))
            self._command = np.array([vx, 0.0, wz], dtype=np.float32)

        else:   # FREE: full 3-DOF PID
            # Position error rotated into body frame
            error_bx =  dx * cos(yaw) + dy * sin(yaw)
            error_by = -dx * sin(yaw) + dy * cos(yaw)
            error_yaw = _normalize_angle(tyaw - yaw)

            error = np.array([error_bx, error_by, error_yaw], dtype=np.float64)
            self._pid_integral += error * dt
            derivative = (error - self._pid_prev_error) / dt if dt > 0 else np.zeros(3)
            self._pid_prev_error = error.copy()

            cmd = _PID_KP * error + _PID_KI * self._pid_integral + _PID_KD * derivative
            cmd = np.clip(cmd, -_PID_MAX_VEL, _PID_MAX_VEL)
            self._command = cmd.astype(np.float32)

    def _set_locked(self, state: bool) -> None:
        """Set lock state and update leg drive gains (10× when locked, 1× when unlocked)."""
        if self._locked == state:
            return  # no-op — avoid redundant USD writes
        self._locked = state
        from spot_config.physics.drive_gains import apply_leg_drive_multiplier
        apply_leg_drive_multiplier(
            self._robot_parent_path,
            multiplier=10.0 if state else 1.0,
        )

    def _check_lock_trigger(self) -> None:
        """Lock joints when robot has been at goal for _SETPOINT_STABLE_THRESH steps."""
        pos, _ = self._robot.get_world_pose()
        yaw = self._get_world_yaw()
        tx, ty, tyaw = self._target_pose[0], self._target_pose[1], self._target_pose[2]

        pos_err = np.hypot(tx - float(pos[0]), ty - float(pos[1]))
        yaw_err = abs(_normalize_angle(tyaw - yaw))

        if pos_err < _POSITION_TOL and yaw_err < _YAW_TOL:
            self._setpoint_stable_steps += 1
        else:
            self._setpoint_stable_steps = 0   # reset — robot is still moving

        if self._setpoint_stable_steps >= _SETPOINT_STABLE_THRESH:
            self._set_locked(True)
            self._command = np.zeros(3, dtype=np.float32)
            self._action[:] = 0.0
            self._previous_action[:] = 0.0
            print(f"[LOCO] Joints locked (pos_err={pos_err:.3f}m, yaw_err={yaw_err:.3f}rad)")

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _compute_observation(self) -> np.ndarray:
        """Build 48-dim observation: lin_vel(3) + ang_vel(3) + gravity(3) + cmd(3) + joint_pos(12) + joint_vel(12) + prev_action(12)."""
        lin_vel_I = self._robot.get_linear_velocity()
        ang_vel_I = self._robot.get_angular_velocity()
        _, q_IB = self._robot.get_world_pose()

        R_BI = quat_to_rot_matrix(q_IB).T
        lin_vel_b = R_BI @ lin_vel_I
        ang_vel_b = R_BI @ ang_vel_I
        gravity_b = R_BI @ np.array([0.0, 0.0, -1.0])

        # Extract only leg joint states (12 joints, already in policy order)
        leg_pos = self._robot.get_joint_positions()[_RELIC_LEG_INDICES]
        leg_vel = self._robot.get_joint_velocities()[_RELIC_LEG_INDICES]

        obs = np.zeros(48, dtype=np.float32)
        obs[0:3]   = lin_vel_b
        obs[3:6]   = ang_vel_b
        obs[6:9]   = gravity_b
        obs[9:12]  = self._command
        obs[12:24] = leg_pos - _DEFAULT_LEG_POS
        obs[24:36] = leg_vel
        obs[36:48] = self._previous_action

        return obs

    # ------------------------------------------------------------------
    # Keyboard pose nudge
    # ------------------------------------------------------------------

    def _nudge_target_pose(self) -> None:
        """Advance _target_pose by _kb_vel in body-relative coordinates.

        The position delta is expressed in the robot's current body frame and
        rotated to world frame before being added to the target.  This is called
        every physics step while any movement key is held.
        """
        if self._target_pose is None:
            # Bootstrap from current robot pose on first key press
            pos, _ = self._robot.get_world_pose()
            robot_yaw = self._get_world_yaw()
            self._target_pose = np.array(
                [float(pos[0]), float(pos[1]), robot_yaw], dtype=np.float64
            )
            self._last_setpoint = self._target_pose.copy()

        cur_x, cur_y, cur_yaw_t = self._target_pose
        robot_yaw = self._get_world_yaw()
        dx_b, dy_b, dyaw = self._kb_vel

        # Rotate body-frame delta to world frame using robot's current heading
        dx_w = dx_b * cos(robot_yaw) - dy_b * sin(robot_yaw)
        dy_w = dx_b * sin(robot_yaw) + dy_b * cos(robot_yaw)

        new_pose = np.array(
            [cur_x + dx_w, cur_y + dy_w, _normalize_angle(cur_yaw_t + dyaw)],
            dtype=np.float64,
        )
        # Update target without resetting PID integrator (motion is continuous)
        self._target_pose = new_pose
        self._last_setpoint = new_pose.copy()
        self._setpoint_stable_steps = 0   # prevent locking while moving
        self._nav_phase = 0               # re-evaluate distance on next step
        if self._locked:
            self._set_locked(False)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def on_key_event(self, key_name: str, pressed: bool) -> None:
        """Called by KeyboardController on each key press/release."""
        if key_name not in _KEY_POSE_MAP:
            return
        if pressed:
            self._kb_held.add(key_name)
        else:
            self._kb_held.discard(key_name)
        # Recompute velocity from the held-key set — immune to timer races
        vel = np.zeros(3, dtype=np.float64)
        for k in self._kb_held:
            vel += _KEY_POSE_MAP[k]
        self._kb_vel = vel
