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

Pose tracking (PID + lock) is delegated to LocomotionPoseTracker.
"""

import sys
from math import atan2, hypot
from pathlib import Path

_SPOT_ISAACSIM = Path(__file__).parent.parent.parent
if str(_SPOT_ISAACSIM) not in sys.path:
    sys.path.insert(0, str(_SPOT_ISAACSIM))

import math
import numpy as np
import torch
from spot_config.cfg.constants import SPOT_STANDING_JOINT_POSITIONS

from .locomotion_pose_tracker import (
    LocomotionPoseTracker,
    _TELEPORT_POS_TOL,
    _TELEPORT_YAW_TOL,
)
from scripts.spot_isaacsim.utils.math import normalize_angle

_POLICY_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "external" / "IsaacRobotics" / "policies"
)
_POLICY_PATH   = _POLICY_DIR / "spot/models/spot_policy.pt"
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
)

_LEG_ON_LOCK_POS = _DEFAULT_LEG_POS.copy() * 1.1

# Keyboard pose-setpoint rates (body-frame, per forward() call at 50 Hz)
from scripts.spot_isaacsim.control.interfaces.keyboard_controller import KEY_POSE_MAP as _KEY_POSE_MAP


class SpotLocomotionController:
    """Locomotion for Spot using the 12-DOF spot_policy.pt.

    Supports two control modes:
    - Velocity mode: set_command(vx, vy, wz) or keyboard
    - Pose mode: set_target_pose(x, y, yaw) — PID drives to world-frame pose (via LocomotionPoseTracker)
    """

    def __init__(
        self,
        robot,
        physics_dt: float = 1.0 / 500.0,
        collision_avoidance: bool = True,
        device: str = "cpu",
        enable: bool = True,
    ) -> None:
        """
        Args:
            robot: SingleArticulation — the 19-DOF Relic Spot robot.
            physics_dt: Physics timestep (1/500). forward() is called at 50 Hz by the caller.
            collision_avoidance: Enable depth-camera collision avoidance filter.
            enable: If False, skip policy/PID. Navigation moves via instant teleport.
        """
        self._robot = robot
        self._physics_dt = physics_dt
        self._device = device
        self._enable = enable
        self._action_scale = 0.2
        self._command = np.zeros(3)
        self._action = np.zeros(12)
        self._previous_action = np.zeros(12)

        self._tracker = LocomotionPoseTracker(robot)

        self._kb_held: set[str] = set()
        self._kb_vel = np.zeros(3, dtype=np.float64)

        self._collision_avoidance_enabled: bool = collision_avoidance
        self._collision_avoidance = None

        if robot.num_dof != 19:
            raise RuntimeError(
                f"[LOCO] Expected 19-DOF robot, got {robot.num_dof}. "
                f"Verify spot_with_arm URDF is loaded."
            )

        if enable:
            if not _POLICY_PATH.exists():
                raise RuntimeError(f"[LOCO] Policy not found: {_POLICY_PATH}")
            self._policy = torch.jit.load(str(_POLICY_PATH)).to(device)
            self._policy.eval()
            print(f"[LOCO] Loaded policy: {_POLICY_PATH.name} (12-DOF legs only)")
            self._decimation = self._parse_decimation()
            print(f"[LOCO] Decimation: {self._decimation}  "
                  f"(forward() called at {1.0 / (self._decimation * physics_dt):.0f} Hz, "
                  f"policy dt = {self._decimation * physics_dt:.4f}s)")
        else:
            self._policy = None
            self._collision_avoidance_enabled = False
            self._decimation = 10
            print("[LOCO] Locomotion disabled — teleport mode (50 Hz)")

    def _parse_decimation(self) -> int:
        """Extract decimation from env.yaml (file uses !!python/tuple tags incompatible with safe_load)."""
        import re
        if not _ENV_YAML_PATH.exists():
            raise RuntimeError(f"[LOCO] env.yaml not found: {_ENV_YAML_PATH}")
        text = _ENV_YAML_PATH.read_text()
        match = re.search(r"^decimation:\s*(\d+)", text, re.MULTILINE)
        if not match:
            raise RuntimeError(f"[LOCO] 'decimation' key not found in {_ENV_YAML_PATH}")
        return int(match.group(1))

    @property
    def decimation(self) -> int:
        return self._decimation

    def initialize(self) -> None:
        """Call after world.reset() and apply_all_physics(). Reads drive gains from USD."""
        import omni.usd
        from pxr import Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        articulation_root = stage.GetPrimAtPath(self._robot.prim_path)
        robot_prim = articulation_root.GetParent()
        robot_parent_path: str = str(robot_prim.GetPath())

        self._tracker.initialize(robot_parent_path)

        # Build name→prim map for all joints
        dof_names = list(self._robot.dof_names)
        joint_prim_map = {}
        for prim in Usd.PrimRange(robot_prim):
            if prim.GetName() in dof_names:
                joint_prim_map[prim.GetName()] = prim

        deg_to_rad = 180.0 / math.pi
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

        self._last_targets_relic = self._robot.get_joint_positions().copy()

        if not self._enable:
            from spot_config.physics.drive_gains import apply_leg_drive_multiplier
            apply_leg_drive_multiplier(robot_parent_path, multiplier=50.0)
            print("[LOCO] Leg joints locked (50× gains)")

        print("[LOCO] Initialization complete")

    # ------------------------------------------------------------------
    # Public control interface
    # ------------------------------------------------------------------

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        """Set velocity command directly (velocity mode — disables PID)."""
        self._tracker.clear_target()
        self._command = np.array([vx, vy, wz])

    def set_target_pose(self, x: float, y: float, yaw: float) -> None:
        """Set world-frame position and yaw target (PID mode)."""
        self._tracker.set_target(x, y, yaw)

    def set_cameras(self, cameras: dict) -> None:
        """Provide depth cameras for collision avoidance."""
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
        """Run one locomotion update. Called at 50 Hz (every decimation physics steps)."""
        from isaacsim.core.utils.types import ArticulationAction

        if np.any(self._kb_vel != 0.0):
            self._tracker.nudge(self._kb_vel)

        # --- Teleport mode (enable=False): no policy, no PID ---
        if not self._enable:
            target = self._tracker.target
            if target is not None:
                pos, _ = self._robot.get_world_pose()
                pos_err = hypot(target[0] - float(pos[0]), target[1] - float(pos[1]))
                yaw_err = abs(normalize_angle(target[2] - self._tracker.get_world_yaw()))
                if pos_err > _TELEPORT_POS_TOL or yaw_err > _TELEPORT_YAW_TOL:
                    self._tracker.teleport_to(target[0], target[1], float(pos[2]), target[2])
                self._command = np.zeros(3, dtype=np.float32)
            self._robot.apply_action(ArticulationAction(
                joint_positions=_DEFAULT_LEG_POS,
                joint_indices=_RELIC_LEG_INDICES,
            ))
            return

        # --- Locked: hold _LEG_ON_LOCK_POS, bypass policy ---
        if self._tracker.is_locked:
            self._tracker.tick_lock_ramp(dt)
            if self._tracker.target is None or self._tracker.check_drift():
                self._robot.apply_action(ArticulationAction(
                    joint_positions=_LEG_ON_LOCK_POS,
                    joint_indices=_RELIC_LEG_INDICES,
                ))
                return
            # drifted — set_locked(False) already called by check_drift(); fall through to PID

        # --- PID update ---
        if self._tracker.target is not None:
            self._command = self._tracker.update_pid_command(dt)
            if self._tracker.check_lock_trigger():
                self._tracker.tick_lock_ramp(dt)
                self._robot.apply_action(ArticulationAction(
                    joint_positions=_DEFAULT_LEG_POS,
                    joint_indices=_RELIC_LEG_INDICES,
                ))
                return

        # --- Collision avoidance ---
        if self._collision_avoidance is not None:
            self._command = self._collision_avoidance.apply(self._command)

        # --- RL policy ---
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
    # Observation
    # ------------------------------------------------------------------

    def _compute_observation(self) -> np.ndarray:
        """Build 48-dim observation: lin_vel(3)+ang_vel(3)+gravity(3)+cmd(3)+joint_pos(12)+joint_vel(12)+prev_action(12)."""
        from isaacsim.core.utils.rotations import quat_to_rot_matrix

        lin_vel_I = self._robot.get_linear_velocity()
        ang_vel_I = self._robot.get_angular_velocity()
        _, q_IB = self._robot.get_world_pose()

        R_BI = quat_to_rot_matrix(q_IB).T
        lin_vel_b = R_BI @ lin_vel_I
        ang_vel_b = R_BI @ ang_vel_I
        gravity_b = R_BI @ np.array([0.0, 0.0, -1.0])

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
    # Keyboard
    # ------------------------------------------------------------------

    def on_key_event(self, key_name: str, pressed: bool) -> None:
        if key_name not in _KEY_POSE_MAP:
            return
        if pressed:
            self._kb_held.add(key_name)
        else:
            self._kb_held.discard(key_name)
        vel = np.zeros(3, dtype=np.float64)
        for k in self._kb_held:
            vel += _KEY_POSE_MAP[k]
        self._kb_vel = vel
