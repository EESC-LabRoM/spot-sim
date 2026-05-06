"""
Spot Arm Controller using Lula Kinematics Solver.

Uses Isaac Sim's built-in LulaKinematicsSolver + ArticulationKinematicsSolver
to compute and apply IK solutions for the Spot arm.

End-effector: arm_link_wr1 (6 DOF: arm_sh0 → arm_wr1, gripper excluded)

Modes (toggle with Numpad 5):
    IDLE     — cube hidden, silently follows EE every step.
               GraspExecutor / ROS2 grasp commands are active.
    TRACKING — cube appears at EE position and is fixed in world space.
               Drag the green cube in the Isaac Sim viewport to guide the arm.
               GraspExecutor is bypassed while tracking is active.
"""

import numpy as np
from scripts.spot_isaacsim.spot_config.cfg.constants import (
    SPOT_RESTING_ARM_JOINT_POSITION,
    SPOT_STANDING_ARM_JOINT_POSITION,
    SPOT_GRIPPER_OPEN,
    SPOT_GRIPPER_CLOSED,
    EE_TARGET_OFFSET,
)
from scripts.spot_isaacsim.control.interfaces.keyboard_controller import (
    ARM_KEY_TOGGLE,
    ARM_KEY_RESTING,
    ARM_KEY_STANDING,
    ARM_KEY_GRIPPER,
)

_TRACKING_CUBE_PATH  = "/World/ArmTrackingTarget"
_TRACKING_CUBE_SIZE  = 0.05                        # m — edge length
_TRACKING_CUBE_COLOR = np.array([0.1, 0.9, 0.4])  # bright green


class SpotArmController:
    END_EFFECTOR = "arm_link_wr1"

    def __init__(self, robot, robot_config):
        """
        Args:
            robot: SingleArticulation object for the Spot robot.
            robot_config: RobotConfig instance (provides URDF and Lula config paths).
        """
        from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
        self._robot = robot
        self._lula = LulaKinematicsSolver(
            robot_description_path=robot_config.lula_config_path,
            urdf_path=robot_config.urdf_path,
        )
        self._art_ik = ArticulationKinematicsSolver(robot, self._lula, self.END_EFFECTOR)

        self._mode = "idle"         # "idle" | "tracking"
        self._tracking_cube = None  # FixedCuboid — created in initialize()
        self._gripper_open  = True  # tracks current gripper state for toggle


    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, world) -> None:
        """Create the tracking cube and register it with the world.

        Call after world.reset() — right after constructing this controller.
        The cube starts hidden; it is repositioned to the EE on every step
        while in IDLE mode so the transition to TRACKING is seamless.
        """
        from isaacsim.core.api.objects import VisualCuboid
        self._tracking_cube = VisualCuboid(
            prim_path=_TRACKING_CUBE_PATH,
            name="arm_tracking_target",
            size=1.0,
            scale=np.array([_TRACKING_CUBE_SIZE] * 3),
            position=np.array([0.0, 0.0, 0.0]),
            color=_TRACKING_CUBE_COLOR,

        )
        world.scene.add(self._tracking_cube)
        self._set_cube_visible(False)
        print("[ARM] Initialized — mode: IDLE  (Numpad 5 to toggle tracking)")

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Call every physics step from play.py.

        IDLE     — silently syncs the hidden cube to the current EE position.
        TRACKING — reads the cube world position and drives the arm toward it.
        """
        if self._tracking_cube is None:
            return
        self._update_lula_base()
        ee_pos, _ = self.get_ee_pose()
        if self._mode == "idle":
            self._tracking_cube.set_world_pose(position=ee_pos)
        else:  # tracking
            cube_pos, _ = self._tracking_cube.get_world_pose()
            self.move_to(self._apply_ee_offset(cube_pos))

    # ------------------------------------------------------------------
    # Public control interface
    # ------------------------------------------------------------------

    def _update_lula_base(self) -> None:
        """Sync the Lula solver's base pose with the robot's current world pose.

        Must be called before any get_ee_pose() query to get world-frame EE positions.
        move_to() calls this internally; update() and _toggle_mode() call it explicitly.
        """
        pos, ori = self._robot.get_world_pose()
        self._lula.set_robot_base_pose(pos, ori)

    def move_to(self, target_position: np.ndarray, target_orientation: np.ndarray = None) -> bool:
        """Compute IK and apply joint position targets. Returns True on success.

        Args:
            target_position: World-frame target position [x, y, z] in meters.
            target_orientation: World-frame target orientation as quaternion [w, x, y, z].
                                 If None, orientation is unconstrained.
        """
        self._update_lula_base()
        action, success = self._art_ik.compute_inverse_kinematics(target_position, target_orientation)
        if success:
            self._robot.apply_action(action)
        return success

    def get_ee_pose(self):
        """Returns (position [3], rotation_matrix [3,3]) of arm_link_wr1 in world frame."""
        return self._art_ik.compute_end_effector_pose()

    def _apply_ee_offset(self, cube_world_pos: np.ndarray) -> np.ndarray:
        """Rotate EE_TARGET_OFFSET from EE local frame to world frame and add to cube pos."""
        _, R = self.get_ee_pose()   # R is rotation_matrix [3,3], already world-frame
        return cube_world_pos + R @ EE_TARGET_OFFSET

    @property
    def is_tracking(self) -> bool:
        """True when arm is in TRACKING mode (executor bypassed)."""
        return self._mode == "tracking"

    # ------------------------------------------------------------------
    # Mode toggle
    # ------------------------------------------------------------------

    def _toggle_mode(self) -> None:
        if self._mode == "idle":
            # Place cube so that (cube + offset) == current EE — arm won't jerk on activation
            self._update_lula_base()
            ee_pos, R = self.get_ee_pose()
            self._tracking_cube.set_world_pose(position=ee_pos - R @ EE_TARGET_OFFSET)
            self._set_cube_visible(True)
            self._mode = "tracking"
            print("[ARM] Mode → TRACKING  (drag green cube in viewport to guide arm)")
        else:
            self._set_cube_visible(False)
            self._mode = "idle"
            print("[ARM] Mode → IDLE  (ROS2 grasp commands active)")

    def _apply_arm_pose(self, pose_dict: dict) -> None:
        """Apply a named arm pose by direct joint position target."""
        from isaacsim.core.utils.types import ArticulationAction
        joint_names = list(self._robot.dof_names)
        indices, positions = [], []
        for name, pos in pose_dict.items():
            if name in joint_names:
                indices.append(joint_names.index(name))
                positions.append(pos)
        if indices:
            self._robot.apply_action(ArticulationAction(
                joint_positions=np.array(positions),
                joint_indices=np.array(indices, dtype=np.int32),
            ))

    def set_resting(self) -> None:
        """Switch to IDLE mode and apply the resting arm pose."""
        self._apply_arm_pose(SPOT_RESTING_ARM_JOINT_POSITION)
        print("[ARM] Set to RESTING pose")

    def set_standing(self) -> None:
        """Switch to IDLE mode and apply the standing arm pose."""
        self._apply_arm_pose(SPOT_STANDING_ARM_JOINT_POSITION)
        print("[ARM] Set to STANDING pose")

    def set_gripper_open(self) -> None:
        """Open the gripper."""
        self._apply_arm_pose(SPOT_GRIPPER_OPEN)
        self._gripper_open = True
        print("[ARM] Gripper OPEN")

    def set_gripper_closed(self) -> None:
        """Close the gripper."""
        self._apply_arm_pose(SPOT_GRIPPER_CLOSED)
        self._gripper_open = False
        print("[ARM] Gripper CLOSED")

    def toggle_gripper(self) -> None:
        """Toggle gripper between open and closed."""
        if self._gripper_open:
            self.set_gripper_closed()
        else:
            self.set_gripper_open()

    def _set_cube_visible(self, visible: bool) -> None:
        import omni
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(_TRACKING_CUBE_PATH)
        if prim and prim.IsValid():
            img = UsdGeom.Imageable(prim)
            img.MakeVisible() if visible else img.MakeInvisible()

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def on_key_event(self, key_name: str, pressed: bool) -> None:
        """Called by KeyboardController on each key press/release."""
        if not pressed:
            return
        if key_name == ARM_KEY_TOGGLE:
            self._toggle_mode()
        elif key_name == ARM_KEY_RESTING:
            self.set_resting()
        elif key_name == ARM_KEY_STANDING:
            self.set_standing()
        elif key_name == ARM_KEY_GRIPPER:
            self.toggle_gripper()
