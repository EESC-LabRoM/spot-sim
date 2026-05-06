"""
SpotController — owns all loco-manip controller lifecycle for Spot in Isaac Sim.

Creates, initializes, and drives:
  - SpotLocomotionController  (base locomotion policy)
  - NavigationExecutor        (ROS /goal_pose subscriber)
  - SpotArmController         (Lula IK)
  - GraspExecutor             (ROS grasp service interface)

Grasp state machine:
  IDLE / REACHING_EE / APPROACHING / GRASPING / WAITING / CLOSING / RETRIEVING / STOWING / HOMING / TRAJECTORY

Usage:
  controller = SpotController(robot, world, scene_cfg, cameras, device)
  # per physics step:
  controller.update()
  # direct control:
  controller.set_grasp_pose(GraspCmd(...))
  controller.set_arm_pose(ArmPoseCmd(...))
  controller.set_body_pose(BodyPoseCmd(...))
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.spot_isaacsim.spot_config.cfg.constants import (
    SPOT_RESTING_ARM_JOINT_POSITION,
    SPOT_STANDING_ARM_JOINT_POSITION,
    APPROACH_DISTANCE_M,
    RETRIEVE_LIFT_M,
    APPROACH_THRESH,
    GRASPING_THRESH,
    RETRIEVING_THRESH,
    HOME_JOINT_THRESH,
    GRIPPER_OPEN,
    GRIPPER_CLOSED,
    WAIT_STEPS,
    CLOSE_STEPS,
    BASE_REACH_M,
    WARMUP_STEPS,
    ARM_JOINT_NAMES,
    EE_TARGET_OFFSET,
)
from scripts.spot_isaacsim.spot_config.cfg.commands import (
    BodyPoseCmd,
    ArmPoseCmd,
    GraspCmd,
    TrajectoryCmd,
    HomeCmd,
)
from scripts.spot_isaacsim.utils.math import (
    advance_along_local_x,
    euclidean_distance_2d,
    euclidean_distance_3d,
    bearing_from_positions,
)

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
IDLE        = "IDLE"
REACHING_EE = "REACHING_EE"
APPROACHING = "APPROACHING"
GRASPING    = "GRASPING"
WAITING     = "WAITING"
CLOSING     = "CLOSING"
RETRIEVING  = "RETRIEVING"
STOWING     = "STOWING"
HOMING      = "HOMING"
TRAJECTORY  = "TRAJECTORY"


class SpotController:
    """
    Owns all loco-manip subsystem lifecycle: creation, initialization,
    per-step update, and reinitialization.

    Exposes a typed Cmd API (set_body_pose, set_arm_pose, set_grasp_pose,
    run_trajectory) backed by an internal step-based state machine.
    The ROS service path (GraspExecutor + NavigationExecutor) is also
    updated here each step.
    """

    def __init__(self, robot, world, scene_cfg, cameras, device: str = "cpu"):
        # Deferred imports — must happen after SimulationApp + extensions are ready.
        from scripts.spot_isaacsim.spot_config.physics import apply_all_physics
        from scripts.spot_isaacsim.control.components.arm_controller import SpotArmController
        from scripts.spot_isaacsim.control.components.locomotion_controller import SpotLocomotionController
        from scripts.spot_isaacsim.control.interfaces.ros_controller import GraspExecutor, NavigationExecutor

        self._apply_all_physics = apply_all_physics
        self._scene_cfg         = scene_cfg
        self._robot             = robot

        # ---- Locomotion ----
        self.locomotion = SpotLocomotionController(
            robot,
            physics_dt=scene_cfg.physics_dt,
            device=device,
            enable=scene_cfg.robot.enable_locomotion,
            collision_avoidance=scene_cfg.robot.enable_collision_avoidance,
        )
        self.locomotion.initialize()
        x, y, yaw = scene_cfg.robot.initial_goal_pose
        self.locomotion.set_target_pose(x, y, yaw)
        self.locomotion.set_cameras(cameras)

        # ---- ROS navigation ----
        self.navigation_executor = NavigationExecutor(self.locomotion, physics_dt=scene_cfg.physics_dt)

        # ---- Arm IK + ROS grasp executor ----
        if scene_cfg.robot.enable_arm_ik:
            self.arm_controller = SpotArmController(robot, scene_cfg.robot)
            self.arm_controller.initialize(world)
            self.executor = GraspExecutor(self, robot)
            if scene_cfg.robot.enable_ros_controllers:
                self.executor.enable_ros2(self.navigation_executor)
                self.navigation_executor.enable_ros2()
            else:
                print("[INFO] ROS controllers disabled")
        else:
            self.arm_controller = None
            self.executor       = None
            print("[INFO] Arm IK disabled")

        # ---- Locomotion step counters ----
        self._step_count   = 0
        self._loco_counter = 0

        # ---- State machine ----
        self._state        = IDLE
        self._step_counter = 0

        self._pre_grasp_pos: np.ndarray | None = None
        self._grasp_pos:     np.ndarray | None = None
        self._quat_wxyz:     np.ndarray | None = None
        self._lift_pos:      np.ndarray | None = None

        self._chain_grasp    = False
        self._chain_retrieve = False

        self._pending_ee_pos: np.ndarray | None = None
        self._pending_ee_ori: np.ndarray | None = None

        self._pending_after_nav: str | None = None  # "grasp" or "ee"

        # Trajectory state
        self._traj_points:        list               = []
        self._traj_joint_indices: np.ndarray | None  = None
        self._traj_elapsed_s:     float              = 0.0
        self._traj_waypoint_idx:  int                = 0
        self._chain_traj_pick:    bool               = False

        # ---- Joint indices (resolved once) ----
        dof_names = list(robot.dof_names)
        self._gripper_idx        = dof_names.index("arm_f1x")
        self._arm_indices        = np.array([dof_names.index(n) for n in ARM_JOINT_NAMES], dtype=np.int32)
        self._resting_positions  = np.array([SPOT_RESTING_ARM_JOINT_POSITION[n]  for n in ARM_JOINT_NAMES])
        self._standing_positions = np.array([SPOT_STANDING_ARM_JOINT_POSITION[n] for n in ARM_JOINT_NAMES])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reinitialize(self) -> None:
        """Re-run controller init after a world.reset() call."""
        self._apply_all_physics(self._scene_cfg.robot.prim_path)
        self.locomotion.initialize()
        x, y, yaw = self._scene_cfg.robot.initial_goal_pose
        self.locomotion.set_target_pose(x, y, yaw)
        self._step_count   = 0
        self._loco_counter = 0

    # ------------------------------------------------------------------
    # Main update — call every physics step
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Drive locomotion, arm IK, state machine, and ROS executor paths."""
        self._step_count += 1
        if self._step_count >= WARMUP_STEPS:
            self._loco_counter += 1
            if self._loco_counter % self.locomotion.decimation == 0:
                self.locomotion.forward(self._scene_cfg.physics_dt * self.locomotion.decimation)

        if self.arm_controller is not None:
            self.arm_controller.update()
            self.tick()
            if not self.arm_controller.is_tracking:
                self.executor.update()

        self.navigation_executor.update()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    def can_accept_command(self, cmd_type: type = None) -> bool:
        """Return True if the controller is IDLE and can accept a new command."""
        return self._state == IDLE

    def set_body_pose(self, cmd: BodyPoseCmd) -> None:
        """Navigate base to (x, y, yaw). Homes arm in parallel."""
        self.locomotion.set_target_pose(cmd.x, cmd.y, cmd.yaw)
        self.transition(HOMING)

    def set_arm_pose(self, cmd: ArmPoseCmd) -> None:
        """Track target EE pose every step. Navigates base to standoff first if needed."""
        robot_pos, _ = self._robot.get_world_pose()
        dist = euclidean_distance_2d(cmd.position, robot_pos)
        self._pending_ee_pos = cmd.position.copy()
        self._pending_ee_ori = cmd.orientation.copy() if cmd.orientation is not None else None

        if dist <= BASE_REACH_M:
            self.transition(REACHING_EE)
            return

        if cmd.orientation is not None:
            R = Rotation.from_quat([
                cmd.orientation[1], cmd.orientation[2],
                cmd.orientation[3], cmd.orientation[0],
            ]).as_matrix()
            theta_g = math.atan2(R[:, 0][1], R[:, 0][0])
        else:
            theta_g = bearing_from_positions(robot_pos, cmd.position)
        new_x = float(cmd.position[0]) - BASE_REACH_M * math.cos(theta_g)
        new_y = float(cmd.position[1]) - BASE_REACH_M * math.sin(theta_g)
        self.locomotion.set_target_pose(new_x, new_y, theta_g)
        self._pending_after_nav = "ee"
        print(f"[CTRL] Base too far ({dist:.2f} m), navigating to ({new_x:.3f}, {new_y:.3f}) then reaching EE.")

    def set_grasp_pose(self, cmd: GraspCmd) -> None:
        """Open gripper → approach pre-grasp → advance to grasp → close gripper."""
        self._pre_grasp_pos  = cmd.pre_grasp_pos
        self._grasp_pos      = cmd.grasp_pos
        self._quat_wxyz      = cmd.orientation
        self._chain_grasp    = cmd.chain_grasp
        self._chain_retrieve = cmd.chain_retrieve

        robot_pos, _ = self._robot.get_world_pose()
        dist = euclidean_distance_2d(cmd.grasp_pos, robot_pos)

        if dist <= BASE_REACH_M:
            self._set_gripper(GRIPPER_OPEN)
            self.transition(APPROACHING)
            return

        theta_g = bearing_from_positions(robot_pos, cmd.grasp_pos)
        new_x = float(cmd.grasp_pos[0]) - BASE_REACH_M * math.cos(theta_g)
        new_y = float(cmd.grasp_pos[1]) - BASE_REACH_M * math.sin(theta_g)
        self.locomotion.set_target_pose(new_x, new_y, theta_g)
        self._pending_after_nav = "grasp"
        print(f"[CTRL] Base too far ({dist:.2f} m), navigating to ({new_x:.3f}, {new_y:.3f}) then grasping.")

    def run_trajectory(self, cmd: TrajectoryCmd) -> None:
        """Execute a joint trajectory. If cmd.chain_pick, close gripper and retrieve at end."""
        self._traj_points        = cmd.points
        self._traj_joint_indices = cmd.joint_indices
        self._traj_elapsed_s     = 0.0
        self._traj_waypoint_idx  = 0
        self._chain_traj_pick    = cmd.chain_pick
        self.transition(TRAJECTORY)

    def home(self) -> None:
        """Move arm to resting position and open gripper."""
        self.transition(HOMING)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def tick(self) -> None:
        # Navigation gate: fire pending action once base has arrived
        if self._pending_after_nav is not None and self.locomotion._locked:
            action, self._pending_after_nav = self._pending_after_nav, None
            if action == "grasp":
                self._set_gripper(GRIPPER_OPEN)
                self.transition(APPROACHING)
            elif action == "ee":
                self.transition(REACHING_EE)
            return

        if self._state == IDLE:
            pass

        elif self._state == REACHING_EE:
            self.arm_controller.move_to(self._pending_ee_pos, self._pending_ee_ori)

        elif self._state == APPROACHING:
            self.arm_controller.move_to(self._pre_grasp_pos, self._quat_wxyz)
            if euclidean_distance_3d(self.arm_controller.get_ee_pose()[0], self._pre_grasp_pos) < APPROACH_THRESH:
                if self._chain_grasp:
                    self._chain_grasp = False
                    self.transition(GRASPING)
                else:
                    self.transition(IDLE)
                    print("[CTRL] At pre-grasp.")

        elif self._state == GRASPING:
            self.arm_controller.move_to(self._grasp_pos, self._quat_wxyz)
            if euclidean_distance_3d(self.arm_controller.get_ee_pose()[0], self._grasp_pos) < GRASPING_THRESH:
                self.transition(WAITING)

        elif self._state == WAITING:
            self.arm_controller.move_to(self._grasp_pos, self._quat_wxyz)
            self._step_counter += 1
            if self._step_counter >= WAIT_STEPS:
                self._set_gripper(GRIPPER_CLOSED)
                self.transition(CLOSING)

        elif self._state == CLOSING:
            self._step_counter += 1
            if self._step_counter >= CLOSE_STEPS:
                if self._chain_retrieve:
                    self._chain_retrieve = False
                    self._lift_pos = self._grasp_pos + np.array([0.0, 0.0, RETRIEVE_LIFT_M])
                    self.transition(RETRIEVING)
                else:
                    self.transition(IDLE)

        elif self._state == RETRIEVING:
            self.arm_controller.move_to(self._lift_pos, self._quat_wxyz)
            if euclidean_distance_3d(self.arm_controller.get_ee_pose()[0], self._lift_pos) < RETRIEVING_THRESH:
                self.transition(STOWING)

        elif self._state == STOWING:
            from isaacsim.core.utils.types import ArticulationAction
            self._robot.apply_action(ArticulationAction(
                joint_positions=self._standing_positions,
                joint_indices=self._arm_indices,
            ))
            current = self._robot.get_joint_positions()[self._arm_indices]
            if float(np.max(np.abs(current - self._standing_positions))) < HOME_JOINT_THRESH:
                self.transition(IDLE)

        elif self._state == HOMING:
            from isaacsim.core.utils.types import ArticulationAction
            self._robot.apply_action(ArticulationAction(
                joint_positions=self._resting_positions,
                joint_indices=self._arm_indices,
            ))
            current = self._robot.get_joint_positions()[self._arm_indices]
            if float(np.max(np.abs(current - self._resting_positions))) < HOME_JOINT_THRESH:
                self._set_gripper(GRIPPER_OPEN)
                self.transition(IDLE)

        elif self._state == TRAJECTORY:
            from scripts.spot_isaacsim.utils.ros import duration_to_s
            from isaacsim.core.utils.types import ArticulationAction
            self._traj_elapsed_s += self._scene_cfg.physics_dt
            while (self._traj_waypoint_idx + 1 < len(self._traj_points) and
                   duration_to_s(self._traj_points[self._traj_waypoint_idx + 1].time_from_start)
                   <= self._traj_elapsed_s):
                self._traj_waypoint_idx += 1
            pt = self._traj_points[self._traj_waypoint_idx]
            self._robot.apply_action(ArticulationAction(
                joint_positions=np.array(pt.positions),
                joint_indices=self._traj_joint_indices,
            ))
            if self._traj_elapsed_s >= duration_to_s(self._traj_points[-1].time_from_start):
                if self._chain_traj_pick:
                    self._chain_traj_pick = False
                    ee_pos, _ = self.arm_controller.get_ee_pose()
                    self._lift_pos = ee_pos + np.array([0.0, 0.0, RETRIEVE_LIFT_M])
                    self._set_gripper(GRIPPER_CLOSED)
                    self.transition(CLOSING)
                else:
                    self.transition(IDLE)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def transition(self, new_state: str) -> None:
        print(f"[CTRL] {self._state} → {new_state}")
        self._state        = new_state
        self._step_counter = 0

    def _set_gripper(self, angle: float) -> None:
        from isaacsim.core.utils.types import ArticulationAction
        self._robot.apply_action(ArticulationAction(
            joint_positions=np.array([angle]),
            joint_indices=np.array([self._gripper_idx], dtype=np.int32),
        ))
