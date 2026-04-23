"""
ROS Controller — consolidates GraspExecutor and NavigationExecutor.

GraspExecutor mirrors the real robot's service interface:
  ~/approach            Open gripper + move arm to pre-grasp pose.
  ~/grasp               Advance along approach axis + close gripper.
  ~/retrieve            Lift 30 cm above grasp, then stow arm (gripper stays closed).
  ~/approach_and_grasp  ~/approach then ~/grasp in one call.
  ~/pick                Full sequence: approach → grasp → retrieve.
  ~/home                Stow arm to SPOT_RESTING_ARM_JOINT_POSITION (opens gripper).
  ~/move_to             No-op in simulation (navigation is separate).
  ~/trajectory          Replay a trajectory_msgs/JointTrajectory from /graspgen/arm_joint_command.

NavigationExecutor drives the Spot base to a goal pose via the locomotion controller:
  IDLE → NAVIGATING → ARRIVED → IDLE
"""

import math
import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from scipy.spatial.transform import Rotation

from scripts.spot_isaacsim.spot_config.robot import SPOT_RESTING_ARM_JOINT_POSITION, SPOT_STANDING_ARM_JOINT_POSITION

# ---------------------------------------------------------------------------
# GraspExecutor — constants and states
# ---------------------------------------------------------------------------

APPROACH_DISTANCE_M = 0.15
RETRIEVE_LIFT_M     = 0.20
APPROACH_THRESH     = 0.02
HOME_JOINT_THRESH   = 0.05

GRIPPER_OPEN   = -1.5
GRIPPER_CLOSED =  0.0

ROBOT_BASE_TARGET_DISTANCE = 1.0

_WAIT_DURATION_STEPS  = 50
_CLOSE_DURATION_STEPS = 50

IDLE        = "IDLE"
APPROACHING = "APPROACHING"
GRASPING    = "GRASPING"
WAITING     = "WAITING"
CLOSING     = "CLOSING"
RETRIEVING  = "RETRIEVING"
STOWING     = "STOWING"
HOMING      = "HOMING"
TRAJECTORY  = "TRAJECTORY"

_ARM_JOINT_NAMES = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"]


def _advance_along_local_x(pos: np.ndarray, quat_wxyz: np.ndarray, distance: float) -> np.ndarray:
    """Translate pos by distance along the pose's local X-axis."""
    R = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return pos + distance * R.as_matrix()[:, 0]


def _duration_to_s(duration) -> float:
    """Convert builtin_interfaces/Duration to seconds."""
    return duration.sec + duration.nanosec * 1e-9


# ---------------------------------------------------------------------------
# GraspExecutor
# ---------------------------------------------------------------------------


class GraspExecutor:
    """Step-based state machine matching the real robot's grasp executor service API."""

    def __init__(self, arm_controller, robot, physics_dt: float = 0.01):
        self._arm        = arm_controller
        self._robot      = robot
        self._state      = IDLE
        self._physics_dt = physics_dt

        self._pre_grasp_pos:  np.ndarray | None = None
        self._grasp_pos:      np.ndarray | None = None
        self._quat_wxyz:      np.ndarray | None = None
        self._lift_pos:       np.ndarray | None = None
        self._arm_at_pregrasp = False
        self._chain_grasp     = False
        self._chain_retrieve  = False

        self._step_counter = 0
        self._wait_steps   = _WAIT_DURATION_STEPS
        self._close_steps  = _CLOSE_DURATION_STEPS
        self._state_pub    = None

        self._nav             = None
        self._cmd_lock        = None
        self._pending_cmd:    tuple | None = None
        self._pose_lock       = None
        self._latest_pose_msg = None

        # Trajectory fields
        self._traj_lock         = None
        self._latest_traj_msg   = None
        self._traj_points       = []
        self._traj_joint_indices: np.ndarray | None = None
        self._traj_elapsed_s    = 0.0
        self._traj_waypoint_idx = 0

        dof_names = list(robot.dof_names)
        self._gripper_idx        = dof_names.index("arm_f1x")
        self._arm_indices        = np.array([dof_names.index(n) for n in _ARM_JOINT_NAMES], dtype=np.int32)
        self._resting_positions  = np.array([SPOT_RESTING_ARM_JOINT_POSITION[n]  for n in _ARM_JOINT_NAMES])
        self._standing_positions = np.array([SPOT_STANDING_ARM_JOINT_POSITION[n] for n in _ARM_JOINT_NAMES])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def do_approach(self, pose_msg) -> tuple[bool, str]:
        if self._state != IDLE:
            return False, "EXECUTOR_BUSY: executor is busy."
        if pose_msg is None:
            return False, "NO_DATA_CLOUD: no grasp pose — publish to /grasp/selected_grasp first."
        self._parse_pose(pose_msg)
        self._arm_at_pregrasp = False
        self._chain_grasp     = False
        self._chain_retrieve  = False
        self._set_gripper(GRIPPER_OPEN)
        self._transition(APPROACHING)
        return True, f"Approaching pre-grasp {self._pre_grasp_pos.round(3)}"

    def do_grasp(self) -> tuple[bool, str]:
        if self._state != IDLE:
            return False, "EXECUTOR_BUSY: executor is busy."
        if not self._arm_at_pregrasp:
            return False, "NO_DATA_CLOUD: arm not at pre-grasp — call ~/approach first."
        self._chain_retrieve = False
        self._transition(GRASPING)
        return True, f"Grasping at {self._grasp_pos.round(3)}"

    def do_approach_and_grasp(self, pose_msg) -> tuple[bool, str]:
        if self._state != IDLE:
            return False, "EXECUTOR_BUSY: executor is busy."
        if pose_msg is None:
            return False, "NO_DATA_CLOUD: no grasp pose — publish to /grasp/selected_grasp first."
        self._parse_pose(pose_msg)
        self._arm_at_pregrasp = False
        self._chain_grasp     = True
        self._chain_retrieve  = False
        self._set_gripper(GRIPPER_OPEN)
        self._transition(APPROACHING)
        return True, "Approach+grasp sequence started."

    def do_retrieve(self) -> tuple[bool, str]:
        if self._state != IDLE:
            return False, "EXECUTOR_BUSY: executor is busy."
        if self._grasp_pos is None:
            return False, "NO_DATA_CLOUD: no grasp pose — perform a grasp first."
        self._lift_pos = self._grasp_pos + np.array([0.0, 0.0, RETRIEVE_LIFT_M])
        self._transition(RETRIEVING)
        return True, f"Retrieving — lifting to {self._lift_pos.round(3)}"

    def do_pick(self, pose_msg) -> tuple[bool, str]:
        if self._state != IDLE:
            return False, "EXECUTOR_BUSY: executor is busy."
        if pose_msg is None:
            return False, "NO_DATA_CLOUD: no grasp pose — publish to /grasp/selected_grasp first."
        self._parse_pose(pose_msg)
        self._arm_at_pregrasp = False
        self._chain_grasp     = True
        self._chain_retrieve  = True
        self._set_gripper(GRIPPER_OPEN)
        self._transition(APPROACHING)
        return True, "Pick sequence started (approach → grasp → retrieve)."

    def do_home(self) -> tuple[bool, str]:
        if self._state != IDLE:
            return False, "EXECUTOR_BUSY: executor is busy."
        self._transition(HOMING)
        return True, "Homing arm to resting position."

    def do_move_to(self, _pose_msg) -> tuple[bool, str]:
        return True, "move_to: no-op in simulation (use /goal_pose for base navigation)."

    def do_trajectory(self) -> tuple[bool, str]:
        """Replay latest trajectory_msgs/JointTrajectory from /graspgen/arm_joint_command."""
        if self._state != IDLE:
            return False, "EXECUTOR_BUSY: executor is busy."
        traj = self._latest_traj()
        if traj is None:
            return False, "NO_DATA_TRAJ: no trajectory received — publish to /graspgen/arm_joint_command first."
        if len(traj.points) == 0:
            return False, "NO_DATA_TRAJ: trajectory has no waypoints."
        dof_names = list(self._robot.dof_names)
        try:
            indices = np.array([dof_names.index(n) for n in traj.joint_names], dtype=np.int32)
        except ValueError as e:
            return False, f"NO_DATA_TRAJ: joint name not found in robot DOFs — {e}"
        self._traj_points        = traj.points
        self._traj_joint_indices = indices
        self._traj_elapsed_s     = 0.0
        self._traj_waypoint_idx  = 0
        self._transition(TRAJECTORY)
        return True, f"Executing trajectory with {len(traj.points)} waypoints."

    @property
    def state(self):
        return self._state

    # ------------------------------------------------------------------
    # ROS2 integration
    # ------------------------------------------------------------------

    def enable_ros2(self, navigation_executor=None) -> None:
        """Start ROS2 services and subscriptions. Call after world.reset()."""
        import threading
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import String
        from std_srvs.srv import Trigger
        from trajectory_msgs.msg import JointTrajectory
        from rclpy.qos import QoSProfile, DurabilityPolicy
        from rclpy.executors import SingleThreadedExecutor

        self._nav       = navigation_executor
        self._cmd_lock  = threading.Lock()
        self._pose_lock = threading.Lock()
        self._traj_lock = threading.Lock()

        if not rclpy.ok():
            rclpy.init()
        self._ros_node = rclpy.create_node("grasp_executor")

        latch_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._state_pub = self._ros_node.create_publisher(String, "/grasp_executor/state", latch_qos)
        self._ros_node.create_subscription(PoseStamped,     "/grasp/selected_grasp",       self._pose_cb, 10)
        self._ros_node.create_subscription(JointTrajectory, "/graspgen/arm_joint_command",  self._traj_cb, 10)

        self._ros_node.create_service(Trigger, "~/approach",           self._queue("approach",           use_pose=True))
        self._ros_node.create_service(Trigger, "~/grasp",              self._queue("grasp"))
        self._ros_node.create_service(Trigger, "~/retrieve",           self._queue("retrieve"))
        self._ros_node.create_service(Trigger, "~/approach_and_grasp", self._queue("approach_and_grasp", use_pose=True))
        self._ros_node.create_service(Trigger, "~/pick",               self._queue("pick",               use_pose=True))
        self._ros_node.create_service(Trigger, "~/home",               self._queue("home"))
        self._ros_node.create_service(Trigger, "~/move_to",            self._queue("move_to",            use_pose=True))
        self._ros_node.create_service(Trigger, "~/trajectory",         self._queue("trajectory"))

        ros_exec = SingleThreadedExecutor()
        ros_exec.add_node(self._ros_node)
        threading.Thread(target=ros_exec.spin, daemon=True).start()
        print("[EXEC] ROS2 services: ~/approach | ~/grasp | ~/retrieve | ~/approach_and_grasp | ~/pick | ~/home | ~/move_to | ~/trajectory")

    def _queue(self, name: str, use_pose: bool = False):
        def handler(_req, resp):
            with self._cmd_lock:
                self._pending_cmd = (name, self._latest_pose() if use_pose else None)
            resp.success, resp.message = True, f"{name} queued"
            return resp
        return handler

    def _pose_cb(self, msg) -> None:
        with self._pose_lock:
            self._latest_pose_msg = msg

    def _latest_pose(self):
        with self._pose_lock:
            return self._latest_pose_msg

    def _traj_cb(self, msg) -> None:
        with self._traj_lock:
            self._latest_traj_msg = msg

    def _latest_traj(self):
        with self._traj_lock:
            return self._latest_traj_msg

    # ------------------------------------------------------------------
    # Step loop
    # ------------------------------------------------------------------

    _DISPATCH = {
        "approach":           lambda self, pose: self.do_approach(pose),
        "grasp":              lambda self, _:    self.do_grasp(),
        "retrieve":           lambda self, _:    self.do_retrieve(),
        "approach_and_grasp": lambda self, pose: self.do_approach_and_grasp(pose),
        "pick":               lambda self, pose: self.do_pick(pose),
        "home":               lambda self, _:    self.do_home(),
        "move_to":            lambda self, pose: self._do_move_to_nav(pose),
        "trajectory":         lambda self, _:    self.do_trajectory(),
    }

    def update(self):
        self._consume_pending()

        if self._state == IDLE:
            pass

        elif self._state == APPROACHING:
            self._arm.move_to(self._pre_grasp_pos, self._quat_wxyz)
            if self._ee_error(self._pre_grasp_pos) < APPROACH_THRESH * 2:
                self._arm_at_pregrasp = True
                if self._chain_grasp:
                    self._chain_grasp = False
                    self._transition(GRASPING)
                else:
                    self._transition(IDLE)
                    print("[EXEC] At pre-grasp — call ~/grasp to proceed.")

        elif self._state == GRASPING:
            self._arm.move_to(self._grasp_pos, self._quat_wxyz)
            if self._ee_error(self._grasp_pos) < APPROACH_THRESH:
                self._transition(WAITING)

        elif self._state == WAITING:
            self._arm.move_to(self._grasp_pos, self._quat_wxyz)
            self._step_counter += 1
            if self._step_counter >= self._wait_steps:
                self._set_gripper(GRIPPER_CLOSED)
                self._transition(CLOSING)

        elif self._state == CLOSING:
            self._step_counter += 1
            if self._step_counter >= self._close_steps:
                if self._chain_retrieve:
                    self._chain_retrieve = False
                    self._lift_pos = self._grasp_pos + np.array([0.0, 0.0, RETRIEVE_LIFT_M])
                    self._transition(RETRIEVING)
                else:
                    self._transition(IDLE)

        elif self._state == RETRIEVING:
            self._arm.move_to(self._lift_pos, self._quat_wxyz)
            if self._ee_error(self._lift_pos) < APPROACH_THRESH:
                self._transition(STOWING)

        elif self._state == STOWING:
            self._robot.apply_action(ArticulationAction(
                joint_positions=self._standing_positions,
                joint_indices=self._arm_indices,
            ))
            current = self._robot.get_joint_positions()[self._arm_indices]
            if float(np.max(np.abs(current - self._standing_positions))) < HOME_JOINT_THRESH:
                self._transition(IDLE)

        elif self._state == HOMING:
            self._robot.apply_action(ArticulationAction(
                joint_positions=self._resting_positions,
                joint_indices=self._arm_indices,
            ))
            current = self._robot.get_joint_positions()[self._arm_indices]
            if float(np.max(np.abs(current - self._resting_positions))) < HOME_JOINT_THRESH:
                self._set_gripper(GRIPPER_OPEN)
                self._transition(IDLE)

        elif self._state == TRAJECTORY:
            self._traj_elapsed_s += self._physics_dt
            # Advance waypoint index while the next waypoint's time has been reached
            while (self._traj_waypoint_idx + 1 < len(self._traj_points) and
                   _duration_to_s(self._traj_points[self._traj_waypoint_idx + 1].time_from_start)
                   <= self._traj_elapsed_s):
                self._traj_waypoint_idx += 1
            pt = self._traj_points[self._traj_waypoint_idx]
            self._robot.apply_action(ArticulationAction(
                joint_positions=np.array(pt.positions),
                joint_indices=self._traj_joint_indices,
            ))
            if self._traj_elapsed_s >= _duration_to_s(self._traj_points[-1].time_from_start):
                self._transition(IDLE)
                print("[EXEC] Trajectory complete.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _consume_pending(self) -> None:
        if self._cmd_lock is None:
            return
        with self._cmd_lock:
            cmd, self._pending_cmd = self._pending_cmd, None
        if cmd is not None:
            self._DISPATCH[cmd[0]](self, cmd[1])

    def _do_move_to_nav(self, pose) -> tuple[bool, str]:
        if self._nav is None:
            return False, "Navigation not available (locomotion not enabled)."
        if pose is None:
            return False, "NO_DATA_CLOUD: no grasp pose — publish to /grasp/selected_grasp first."
        p, q = pose.pose.position, pose.pose.orientation
        local_x = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()[:, 0]
        theta_g = math.atan2(local_x[1], local_x[0])
        robot_pos, _ = self._robot.get_world_pose()
        dist  = min(math.hypot(p.x - float(robot_pos[0]), p.y - float(robot_pos[1])), ROBOT_BASE_TARGET_DISTANCE)
        new_x = p.x - dist * math.cos(theta_g)
        new_y = p.y - dist * math.sin(theta_g)
        self._nav.accept_goal(new_x, new_y, theta_g)
        return True, "Move-to accepted."

    def _parse_pose(self, pose_msg) -> None:
        from scripts.spot_isaacsim.control.arm_controller import _EE_TARGET_OFFSET
        p, q = pose_msg.pose.position, pose_msg.pose.orientation
        q_urdf    = (Rotation.from_quat([q.x, q.y, q.z, q.w]) * Rotation.from_euler('x', np.pi / 2)).as_quat()
        quat_wxyz = np.array([q_urdf[3], q_urdf[0], q_urdf[1], q_urdf[2]])
        grasp_pos     = np.array([p.x, p.y, p.z])
        pre_grasp_pos = _advance_along_local_x(grasp_pos, quat_wxyz, -APPROACH_DISTANCE_M)
        R             = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
        offset_world  = R @ _EE_TARGET_OFFSET
        grasp_pos    += offset_world
        pre_grasp_pos += offset_world
        self._pre_grasp_pos = pre_grasp_pos
        self._grasp_pos     = grasp_pos
        self._quat_wxyz     = quat_wxyz
        print(f"[EXEC] Pose parsed. Pre-grasp={pre_grasp_pos.round(3)}, Grasp={grasp_pos.round(3)}")

    def _transition(self, new_state: str) -> None:
        print(f"[EXEC] {self._state} → {new_state}")
        self._state        = new_state
        self._step_counter = 0
        if self._state_pub is not None:
            from std_msgs.msg import String
            msg = String(); msg.data = new_state
            self._state_pub.publish(msg)

    def _set_gripper(self, angle: float) -> None:
        self._robot.apply_action(ArticulationAction(
            joint_positions=np.array([angle]),
            joint_indices=np.array([self._gripper_idx], dtype=np.int32),
        ))

    def _ee_error(self, target_pos: np.ndarray) -> float:
        ee_pos, _ = self._arm.get_ee_pose()
        return float(np.linalg.norm(ee_pos - target_pos))


# ---------------------------------------------------------------------------
# NavigationExecutor — constants
# ---------------------------------------------------------------------------

_NAV_IDLE       = "IDLE"
_NAV_NAVIGATING = "NAVIGATING"
_NAV_ARRIVED    = "ARRIVED"

_CMD_ZERO_THRESH  = 0.05
_ARRIVED_STEPS    = 150
_IDLE_RESET_STEPS = 150


def _yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw (rotation about Z) from a quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# ---------------------------------------------------------------------------
# NavigationExecutor
# ---------------------------------------------------------------------------


class NavigationExecutor:
    """State machine that drives the Spot base to a goal pose."""

    def __init__(self, locomotion_controller, physics_dt: float = 0.002):
        self._locomotion    = locomotion_controller
        self._state         = _NAV_IDLE
        self._still_steps   = 0
        self._arrived_steps = 0
        self._pending_goal: tuple | None = None

    def set_goal(self, pose_msg) -> tuple:
        """Set a navigation goal from a geometry_msgs/PoseStamped message."""
        p   = pose_msg.pose.position
        q   = pose_msg.pose.orientation
        x   = p.x
        y   = p.y
        yaw = _yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self._pending_goal = (x, y, yaw)
        print(f"[NAV] Goal queued: x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f}°")
        return True, f"Goal queued ({x:.3f}, {y:.3f}, {math.degrees(yaw):.1f}°)"

    def enable_ros2(self) -> None:
        """Subscribe to /goal_pose (RViz 2D Goal Pose). Call after world.reset()."""
        import threading
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.executors import SingleThreadedExecutor

        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("spot_sim_nav_goal_receiver")
        node.create_subscription(PoseStamped, "/goal_pose", self.set_goal, 10)
        ros_exec = SingleThreadedExecutor()
        ros_exec.add_node(node)
        threading.Thread(target=ros_exec.spin, daemon=True).start()
        print("[NAV] ROS2 active: /goal_pose (RViz 2D Goal Pose)")

    def accept_goal(self, x: float, y: float, yaw: float) -> None:
        """Accept a pre-computed goal directly from the main thread."""
        self._locomotion.set_target_pose(x, y, yaw)
        self._transition(_NAV_NAVIGATING)
        print(f"[NAV] Goal accepted: x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f}°")

    def cancel(self):
        """Cancel active navigation and stop the robot."""
        if self._state != _NAV_IDLE:
            self._locomotion.set_command(0.0, 0.0, 0.0)
            self._transition(_NAV_IDLE)
            print("[NAV] Navigation cancelled")

    @property
    def state(self) -> str:
        return self._state

    def update(self):
        if self._pending_goal is not None:
            x, y, yaw = self._pending_goal
            self._pending_goal = None
            self._locomotion.set_target_pose(x, y, yaw)
            self._transition(_NAV_NAVIGATING)
            print(f"[NAV] Goal accepted: x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f}°")

        if self._state == _NAV_IDLE:
            pass

        elif self._state == _NAV_NAVIGATING:
            cmd = self._locomotion.command
            if max(abs(cmd[0]), abs(cmd[1]), abs(cmd[2])) < _CMD_ZERO_THRESH:
                self._still_steps += 1
            else:
                self._still_steps = 0
            if self._still_steps >= _ARRIVED_STEPS:
                self._transition(_NAV_ARRIVED)
                print("[NAV] Arrived at goal")

        elif self._state == _NAV_ARRIVED:
            self._arrived_steps += 1
            if self._arrived_steps >= _IDLE_RESET_STEPS:
                self._transition(_NAV_IDLE)

    def _transition(self, new_state):
        print(f"[NAV] {self._state} → {new_state}")
        self._state         = new_state
        self._still_steps   = 0
        self._arrived_steps = 0
