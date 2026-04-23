import torch
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils

from dataset_generation.utils.grasp_data_loader import (
    GraspDataLoader,
    compute_grasp_orientation_from_normal,
)


# TODO: debug normal alignment -> is the normal wrong or the algebric manipulation?
class GripperGraspingAction(ActionTerm):
    """
    Combined Kinematic Controller:
    1. teleports base (wrist) to a target grasp point from loaded point cloud data.
    2. closes fingers continuously until contact is detected.
    """

    cfg: "GripperGraspingActionCfg"

    def __init__(self, cfg: "GripperGraspingActionCfg", env):
        super().__init__(cfg, env)
        # robot and sensors
        self.robot = env.scene[cfg.asset_name]
        self.contact_sensor: ContactSensor = env.scene[cfg.contact_sensor_name]
        self.joint_ids, _ = self.robot.find_joints(cfg.joint_names)
        # target object for reference
        self.target_object = env.scene[cfg.target_object_name]

        # Initialize grasp data loader
        # Testing phase: loads view_000 only
        self.grasp_loader = GraspDataLoader(
            dataset_dir=cfg.dataset_dir,
            device=self.device,
            view_id=0,  # Always load view_000 for testing
        )

        # Load initial grasp poses for all environments
        self._load_grasp_poses()

        # Convert gripper offset to tensor
        self.gripper_offset = torch.tensor(
            cfg.gripper_offset_from_point, device=self.device, dtype=torch.float32
        ).unsqueeze(0)  # Shape: (1, 3) for broadcasting

        # internal state
        self.current_finger_pos = torch.zeros(
            self.num_envs, len(self.joint_ids), device=self.device
        )
        self.is_stopped = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # storage for action data
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )

    def _load_grasp_poses(self):
        """
        Load grasp poses from the dataset for all environments.

        Each environment gets one grasp point from the loaded point cloud.

        COORDINATE-FRAME NOTE
        ---------------------
        The dataset stores all coordinates in env-local frame
        (world - env.scene.env_origins at save time).  To get world-frame
        positions suitable for write_root_pose_to_sim we simply add back
        this env's grid origin:

            point_world = point_env_local + env.scene.env_origins

        This is correct for any num_envs / env_spacing combination.
        Normals are direction vectors and need no translation.
        """
        # Get grasp poses for all environments (returned in env-local frame)
        grasp_points, grasp_normals, approach_points = (
            self.grasp_loader.get_grasp_poses_for_envs(
                num_envs=self.num_envs,
                selection_mode=self.cfg.point_selection_mode,
                grasp_offset=self.cfg.approach_offset,
            )
        )

        # env-local → world frame
        # env_origins: (num_envs, 3) — each row is this env's grid origin in world
        env_origins = self._env.scene.env_origins  # (num_envs, 3)
        grasp_points = grasp_points + env_origins  # (num_envs, 3)
        approach_points = approach_points + env_origins  # (num_envs, 3)
        # normals are directions — no translation needed

        # Store grasp points and normals
        self.grasp_points = grasp_points  # (num_envs, 3)
        self.grasp_normals = grasp_normals  # (num_envs, 3)
        self.approach_points = approach_points  # (num_envs, 3)

        # Compute grasp orientations from normals
        if self.cfg.use_normal_orientation:
            self.grasp_orientations = compute_grasp_orientation_from_normal(
                self.grasp_normals, up_axis="z"
            )
        else:
            # Use fixed orientation from config
            fixed_quat = torch.tensor(
                self.cfg.grasp_orientation_quat, device=self.device
            )
            self.grasp_orientations = fixed_quat.repeat(self.num_envs, 1)

        print(f"[GripperAction] Loaded {self.num_envs} grasp poses from dataset")
        print(f"[GripperAction] First grasp point: {self.grasp_points[0]}")
        print(f"[GripperAction] First grasp normal: {self.grasp_normals[0]}")

    @property
    def action_dim(self) -> int:
        """Dimension of the action space: 1 (close/open command)."""
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the term."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The actions computed by the term after processing."""
        return self._processed_actions

    def apply_actions(self):
        """Apply the processed actions to the robot."""
        # The articulation actions are already being applied in process_actions
        # This method is called by the action manager after process_actions
        pass

    def process_actions(self, actions: torch.Tensor):
        """
        Kinematic Teleportation -> Base/Root
        Uses loaded grasp points and normals from point cloud data.
        """
        # Store raw actions
        self._raw_actions[:] = actions

        # Compute target positions for gripper base
        # Use approach points (offset along normal) as the gripper target
        target_pos = self.approach_points - self.gripper_offset

        # Pose tensor: position + quaternion
        grasp_root_pose = torch.cat([target_pos, self.grasp_orientations], dim=-1)

        # Write to simulation -> Teleport
        self.robot.write_root_pose_to_sim(grasp_root_pose)

        # Zero velocity -> kill drift
        self.robot.write_root_velocity_to_sim(
            torch.zeros_like(self.robot.data.root_vel_w)
        )

        """
        Finger Control -> stop on contact
        """
        # parse input: > 0.5 is Close)
        want_close = actions[:, 0] > 0.0
        # check contact
        forces = self.contact_sensor.data.net_forces_w
        if forces.ndim == 3:
            forces = forces[:, 0, :]
        has_contact = torch.norm(forces, dim=-1) > self.cfg.force_threshold
        # update state
        self.is_stopped = torch.where(want_close, self.is_stopped | has_contact, False)
        # calculate motion
        step = self.cfg.close_speed_rad_per_step
        delta = torch.where(self.is_stopped.unsqueeze(1), 0.0, -step)
        next_closing_pos = self.current_finger_pos + delta
        # reset if opening
        target_if_opening = torch.full_like(self.current_finger_pos, self.cfg.open_pos)
        # final command selection
        self.current_finger_pos = torch.where(
            want_close.unsqueeze(1), next_closing_pos, target_if_opening
        )
        # clamp limits
        self.current_finger_pos = torch.clamp(
            self.current_finger_pos,
            min=self.cfg.closed_pos_limit,
            max=self.cfg.open_pos,
        )

        # Store processed actions
        self._processed_actions[:] = actions

        # Set the joint position targets using the robot's articulation interface
        self.robot.set_joint_position_target(
            self.current_finger_pos, joint_ids=self.joint_ids
        )

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)

        self.current_finger_pos[env_ids] = self.cfg.open_pos
        self.is_stopped[env_ids] = False

        # Reload grasp poses on reset if configured
        # Testing phase: always uses view_000, just resamples points
        if self.cfg.reload_on_reset:
            if env_ids is slice(None):
                # Full reset - reload all poses
                self._load_grasp_poses()
            else:
                # Partial reset — reload only specific environments
                grasp_points, grasp_normals, approach_points = (
                    self.grasp_loader.get_grasp_poses_for_envs(
                        num_envs=len(env_ids),
                        selection_mode=self.cfg.point_selection_mode,
                        grasp_offset=self.cfg.approach_offset,
                    )
                )

                # env-local → world frame for the reset envs only
                env_origins = self._env.scene.env_origins[env_ids]  # (len(env_ids), 3)
                grasp_points = grasp_points + env_origins
                approach_points = approach_points + env_origins

                self.grasp_points[env_ids] = grasp_points
                self.grasp_normals[env_ids] = grasp_normals
                self.approach_points[env_ids] = approach_points

                if self.cfg.use_normal_orientation:
                    self.grasp_orientations[env_ids] = (
                        compute_grasp_orientation_from_normal(
                            grasp_normals, up_axis="z"
                        )
                    )


# define the Config Class -> isaac is really annoying about this


@configclass
class GripperGraspingActionCfg(ActionTermCfg):
    class_type = GripperGraspingAction
    asset_name: str = MISSING
    joint_names: list[str] = MISSING

    # object to follow
    target_object_name: str = MISSING
    contact_sensor_name: str = MISSING

    # motion params
    open_pos: float = 0.0
    closed_pos_limit: float = -1.57
    close_speed_rad_per_step: float = 0.05
    force_threshold: float = 1.0

    # grasp data params
    dataset_dir: str = "raw_dataset"
    point_selection_mode: str = "random"  # "random", "sequential", "uniform"
    approach_offset: float = 0.05  # Offset distance along normal for approach
    use_normal_orientation: bool = True  # Use normal to compute orientation
    reload_on_reset: bool = True  # Reload grasp poses on environment reset

    # gripper offset from grasp point
    gripper_offset_from_point: tuple = (0.0, 0.0, 0.0)

    # fallback orientation if not using normal
    grasp_orientation_quat: tuple = (0.707, 0.707, 0.0, 0.0)  # (w, x, y, z)


# Action Configuration Instance


@configclass
class ActionsCfg:
    # use Gripper Action
    gripper_action: GripperGraspingActionCfg = GripperGraspingActionCfg(
        asset_name="robot",
        target_object_name="target_object",
        contact_sensor_name="contact_sensor",
        joint_names=["arm_f1x"],
        # motion params
        open_pos=0.0,
        closed_pos_limit=-1.57,
        close_speed_rad_per_step=0.05,
        force_threshold=1.0,  # [N]
        # grasp data params (testing phase: view_000 only)
        dataset_dir="raw_dataset",
        point_selection_mode="random",
        approach_offset=0.05,
        use_normal_orientation=True,
        reload_on_reset=True,
        # gripper offset
        gripper_offset_from_point=(0.0, 0.0, 0.0),
        # fallback orientation
        grasp_orientation_quat=(0.707, 0.707, 0.0, 0.0),
    )
