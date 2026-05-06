"""
ROS Controller — thin ROS adapters for GraspExecutor and NavigationExecutor.

GraspExecutor services (validate + delegate to SpotController only):
  ~/approach            Move arm to pre-grasp pose.
  ~/grasp               Advance to grasp pose.
  ~/retrieve            Lift 20 cm above grasp.
  ~/approach_and_grasp  Approach then grasp in one call.
  ~/pick                Full sequence: approach → grasp → retrieve.
  ~/home                Stow arm to resting position.
  ~/move_to             Drive base to a pose via NavigationExecutor.
  ~/trajectory          Replay a trajectory_msgs/JointTrajectory.
  ~/trajectory_pick     Replay trajectory then close gripper and retrieve.

NavigationExecutor drives the Spot base to a goal pose:
  IDLE → NAVIGATING → ARRIVED → IDLE
"""

import math
import threading

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.spot_isaacsim.spot_config.cfg.constants import (
    APPROACH_DISTANCE_M,
    RETRIEVE_LIFT_M,
    EE_TARGET_OFFSET,
    NAV_CMD_ZERO_THRESH,
    NAV_ARRIVED_STEPS,
    NAV_IDLE_RESET_STEPS,
    BASE_REACH_M,
)
from scripts.spot_isaacsim.spot_config.cfg.commands import (
    GraspCmd,
    TrajectoryCmd,
)
from scripts.spot_isaacsim.utils.math import (
    yaw_from_quaternion,
    euclidean_distance_2d,
    bearing_from_positions,
)
from scripts.spot_isaacsim.utils.ros import parse_pose_stamped


# ---------------------------------------------------------------------------
# GraspExecutor — thin ROS adapter
# ---------------------------------------------------------------------------


class GraspExecutor:
    """Thin ROS service adapter. Validates requests and delegates to SpotController."""

    def __init__(self, controller, robot, physics_dt: float = 0.01):
        self._ctrl       = controller
        self._robot      = robot
        self._nav             = None
        self._cmd_lock        = None
        self._pose_lock       = threading.Lock()
        self._traj_lock       = threading.Lock()
        self._latest_pose_msg = None
        self._latest_traj_msg = None
        self._pending_cmd: tuple | None = None
        self._state_pub       = None

    # ------------------------------------------------------------------
    # Public do_* API — called from _consume_pending
    # ------------------------------------------------------------------

    def do_approach(self, pose_msg) -> tuple[bool, str]:
        if not self._ctrl.can_accept_command():
            return False, "EXECUTOR_BUSY: controller is busy."
        if pose_msg is None:
            return False, "NO_DATA_CLOUD: no grasp pose — publish to /grasp/selected_grasp first."
        pre, grasp, quat = parse_pose_stamped(pose_msg, EE_TARGET_OFFSET, APPROACH_DISTANCE_M)
        cmd = GraspCmd(pre_grasp_pos=pre, grasp_pos=grasp, orientation=quat,
                       chain_grasp=False, chain_retrieve=False)
        self._ctrl.set_grasp_pose(cmd)
        return True, f"Approaching pre-grasp {pre.round(3)}"

    def do_grasp(self, _pose_msg=None) -> tuple[bool, str]:
        if not self._ctrl.can_accept_command():
            return False, "EXECUTOR_BUSY: controller is busy."
        from scripts.spot_isaacsim.control.spot_controller import GRASPING
        self._ctrl.transition(GRASPING)
        return True, "Grasping."

    def do_approach_and_grasp(self, pose_msg) -> tuple[bool, str]:
        if not self._ctrl.can_accept_command():
            return False, "EXECUTOR_BUSY: controller is busy."
        if pose_msg is None:
            return False, "NO_DATA_CLOUD: no grasp pose — publish to /grasp/selected_grasp first."
        pre, grasp, quat = parse_pose_stamped(pose_msg, EE_TARGET_OFFSET, APPROACH_DISTANCE_M)
        cmd = GraspCmd(pre_grasp_pos=pre, grasp_pos=grasp, orientation=quat,
                       chain_grasp=True, chain_retrieve=False)
        self._ctrl.set_grasp_pose(cmd)
        return True, "Approach+grasp sequence started."

    def do_retrieve(self, _pose_msg=None) -> tuple[bool, str]:
        if not self._ctrl.can_accept_command():
            return False, "EXECUTOR_BUSY: controller is busy."
        from scripts.spot_isaacsim.control.spot_controller import RETRIEVING
        if self._ctrl._grasp_pos is None:
            return False, "NO_DATA_CLOUD: no grasp pose — perform a grasp first."
        self._ctrl._lift_pos = self._ctrl._grasp_pos + np.array([0.0, 0.0, RETRIEVE_LIFT_M])
        self._ctrl.transition(RETRIEVING)
        return True, f"Retrieving — lifting to {self._ctrl._lift_pos.round(3)}"

    def do_pick(self, pose_msg) -> tuple[bool, str]:
        if not self._ctrl.can_accept_command():
            return False, "EXECUTOR_BUSY: controller is busy."
        if pose_msg is None:
            return False, "NO_DATA_CLOUD: no grasp pose — publish to /grasp/selected_grasp first."
        pre, grasp, quat = parse_pose_stamped(pose_msg, EE_TARGET_OFFSET, APPROACH_DISTANCE_M)
        cmd = GraspCmd(pre_grasp_pos=pre, grasp_pos=grasp, orientation=quat,
                       chain_grasp=True, chain_retrieve=True)
        self._ctrl.set_grasp_pose(cmd)
        return True, "Pick sequence started (approach → grasp → retrieve)."

    def do_home(self, _pose_msg=None) -> tuple[bool, str]:
        if not self._ctrl.can_accept_command():
            return False, "EXECUTOR_BUSY: controller is busy."
        self._ctrl.home()
        return True, "Homing arm to resting position."

    def do_move_to(self, pose_msg) -> tuple[bool, str]:
        if self._nav is None:
            return False, "Navigation not available (locomotion not enabled)."
        if pose_msg is None:
            return False, "NO_DATA_CLOUD: no pose message."
        p, q = pose_msg.pose.position, pose_msg.pose.orientation
        robot_pos, _ = self._robot.get_world_pose()
        theta_g = bearing_from_positions(np.array([float(robot_pos[0]), float(robot_pos[1])]),
                                         np.array([p.x, p.y]))
        dist    = min(euclidean_distance_2d(np.array([p.x, p.y]),
                                            np.array([float(robot_pos[0]), float(robot_pos[1])])),
                      BASE_REACH_M)
        new_x   = p.x - dist * math.cos(theta_g)
        new_y   = p.y - dist * math.sin(theta_g)
        self._nav.accept_goal(new_x, new_y, theta_g)
        return True, "Move-to accepted."

    def do_trajectory(self, _pose_msg=None) -> tuple[bool, str]:
        if not self._ctrl.can_accept_command():
            return False, "EXECUTOR_BUSY: controller is busy."
        traj = self._latest_traj()
        if traj is None:
            return False, "NO_DATA_TRAJ: no trajectory — publish to /graspgen/arm_joint_command first."
        if len(traj.points) == 0:
            return False, "NO_DATA_TRAJ: trajectory has no waypoints."
        dof_names = list(self._robot.dof_names)
        try:
            indices = np.array(
                [dof_names.index(n) for n in traj.joint_names], dtype=np.int32
            )
        except ValueError as e:
            return False, f"NO_DATA_TRAJ: joint name not found — {e}"
        self._ctrl.run_trajectory(TrajectoryCmd(points=traj.points, joint_indices=indices, chain_pick=False))
        return True, f"Executing trajectory with {len(traj.points)} waypoints."

    def do_trajectory_pick(self, _pose_msg=None) -> tuple[bool, str]:
        if not self._ctrl.can_accept_command():
            return False, "EXECUTOR_BUSY: controller is busy."
        traj = self._latest_traj()
        if traj is None:
            return False, "NO_DATA_TRAJ: no trajectory — publish to /graspgen/arm_joint_command first."
        if len(traj.points) == 0:
            return False, "NO_DATA_TRAJ: trajectory has no waypoints."
        dof_names = list(self._robot.dof_names)
        try:
            indices = np.array([dof_names.index(n) for n in traj.joint_names], dtype=np.int32)
        except ValueError as e:
            return False, f"NO_DATA_TRAJ: joint name not found — {e}"
        self._ctrl.run_trajectory(TrajectoryCmd(points=traj.points, joint_indices=indices, chain_pick=True))
        return True, f"Trajectory pick: {len(traj.points)} waypoints."

    # ------------------------------------------------------------------
    # update() — no-op; state machine lives in SpotController
    # ------------------------------------------------------------------

    def update(self) -> None:
        self._consume_pending()

    # ------------------------------------------------------------------
    # ROS2 integration
    # ------------------------------------------------------------------

    def enable_ros2(self, navigation_executor=None) -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import String
        from std_srvs.srv import Trigger
        from trajectory_msgs.msg import JointTrajectory
        from rclpy.qos import QoSProfile, DurabilityPolicy
        from rclpy.executors import SingleThreadedExecutor

        self._nav      = navigation_executor
        self._cmd_lock = threading.Lock()

        if not rclpy.ok():
            rclpy.init()
        self._ros_node = rclpy.create_node("grasp_executor")

        latch_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._state_pub = self._ros_node.create_publisher(String, "/grasp_executor/state", latch_qos)
        self._ros_node.create_subscription(PoseStamped,     "/grasp/selected_grasp",      self._pose_cb, 10)
        self._ros_node.create_subscription(JointTrajectory, "/graspgen/arm_joint_command", self._traj_cb, 10)

        self._ros_node.create_service(Trigger, "~/approach",           self._queue("approach",           use_pose=True))
        self._ros_node.create_service(Trigger, "~/grasp",              self._queue("grasp"))
        self._ros_node.create_service(Trigger, "~/retrieve",           self._queue("retrieve"))
        self._ros_node.create_service(Trigger, "~/approach_and_grasp", self._queue("approach_and_grasp", use_pose=True))
        self._ros_node.create_service(Trigger, "~/pick",               self._queue("pick",               use_pose=True))
        self._ros_node.create_service(Trigger, "~/home",               self._queue("home"))
        self._ros_node.create_service(Trigger, "~/move_to",            self._queue("move_to",            use_pose=True))
        self._ros_node.create_service(Trigger, "~/trajectory",         self._queue("trajectory"))
        self._ros_node.create_service(Trigger, "~/trajectory_pick",    self._queue("trajectory_pick"))

        ros_exec = SingleThreadedExecutor()
        ros_exec.add_node(self._ros_node)
        threading.Thread(target=ros_exec.spin, daemon=True).start()
        print("[EXEC] ROS2 services: ~/approach | ~/grasp | ~/retrieve | ~/approach_and_grasp | ~/pick | ~/home | ~/move_to | ~/trajectory | ~/trajectory_pick")

    _DISPATCH = {
        "approach":           lambda self, pose: self.do_approach(pose),
        "grasp":              lambda self, _:    self.do_grasp(),
        "retrieve":           lambda self, _:    self.do_retrieve(),
        "approach_and_grasp": lambda self, pose: self.do_approach_and_grasp(pose),
        "pick":               lambda self, pose: self.do_pick(pose),
        "home":               lambda self, _:    self.do_home(),
        "move_to":            lambda self, pose: self.do_move_to(pose),
        "trajectory":         lambda self, _:    self.do_trajectory(),
        "trajectory_pick":    lambda self, _:    self.do_trajectory_pick(),
    }

    def _queue(self, name: str, use_pose: bool = False):
        def handler(_req, resp):
            with self._cmd_lock:
                self._pending_cmd = (name, self._latest_pose() if use_pose else None)
            resp.success, resp.message = True, f"{name} queued"
            return resp

        return handler

    def _consume_pending(self) -> None:
        if self._cmd_lock is None:
            return
        with self._cmd_lock:
            cmd, self._pending_cmd = self._pending_cmd, None
        if cmd is not None:
            ok, msg = self._DISPATCH[cmd[0]](self, cmd[1])
            if not ok:
                print(f"[EXEC] {cmd[0]} failed: {msg}")

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


# ---------------------------------------------------------------------------
# NavigationExecutor
# ---------------------------------------------------------------------------

_NAV_IDLE = "IDLE"
_NAV_NAVIGATING = "NAVIGATING"
_NAV_ARRIVED    = "ARRIVED"


class NavigationExecutor:
    """Drives the Spot base to a goal pose from ROS /goal_pose topic."""

    def __init__(self, locomotion_controller, physics_dt: float = 0.002):
        self.locomotion     = locomotion_controller
        self._state         = _NAV_IDLE
        self._still_steps   = 0
        self._arrived_steps = 0
        self._pending_goal: tuple | None = None

    def set_goal(self, pose_msg) -> tuple:
        p, q = pose_msg.pose.position, pose_msg.pose.orientation
        yaw  = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self._pending_goal = (p.x, p.y, yaw)
        print(f"[NAV] Goal queued: x={p.x:.3f} y={p.y:.3f} yaw={math.degrees(yaw):.1f}°")
        return True, f"Goal queued ({p.x:.3f}, {p.y:.3f}, {math.degrees(yaw):.1f}°)"

    def enable_ros2(self) -> None:
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
        self.locomotion.set_target_pose(x, y, yaw)
        self.transition(_NAV_NAVIGATING)
        print(f"[NAV] Goal accepted: x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f}°")

    def cancel(self):
        if self._state != _NAV_IDLE:
            self.locomotion.set_command(0.0, 0.0, 0.0)
            self.transition(_NAV_IDLE)

    @property
    def state(self) -> str:
        return self._state

    def update(self):
        if self._pending_goal is not None:
            x, y, yaw = self._pending_goal
            self._pending_goal = None
            self.accept_goal(x, y, yaw)

        if self._state == _NAV_NAVIGATING:
            cmd = self.locomotion.command
            if max(abs(cmd[0]), abs(cmd[1]), abs(cmd[2])) < NAV_CMD_ZERO_THRESH:
                self._still_steps += 1
            else:
                self._still_steps = 0
            if self._still_steps >= NAV_ARRIVED_STEPS:
                self.transition(_NAV_ARRIVED)
                print("[NAV] Arrived at goal")

        elif self._state == _NAV_ARRIVED:
            self._arrived_steps += 1
            if self._arrived_steps >= NAV_IDLE_RESET_STEPS:
                self.transition(_NAV_IDLE)

    def transition(self, new_state: str) -> None:
        self._state         = new_state
        self._still_steps   = 0
        self._arrived_steps = 0
