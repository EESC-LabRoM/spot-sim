"""
Isaac Lab ObservationTerm implementations for Spot robot.
Provides structured observations through ObservationManager.
"""
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv


def arm_joint_positions(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Get current arm joint positions (6 DOF).

    Args:
        env: Environment instance
        asset_cfg: Asset configuration

    Returns:
        [num_envs, 6] arm joint positions
    """
    robot = env.scene[asset_cfg.name]
    arm_names = [n for n in robot.data.joint_names if n.startswith("arm")][:6]
    arm_indices = [robot.data.joint_names.index(n) for n in arm_names]
    return robot.data.joint_pos[:, arm_indices]


def arm_joint_velocities(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Get current arm joint velocities (6 DOF).

    Args:
        env: Environment instance
        asset_cfg: Asset configuration

    Returns:
        [num_envs, 6] arm joint velocities
    """
    robot = env.scene[asset_cfg.name]
    arm_names = [n for n in robot.data.joint_names if n.startswith("arm")][:6]
    arm_indices = [robot.data.joint_names.index(n) for n in arm_names]
    return robot.data.joint_vel[:, arm_indices]


def ee_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Get end-effector pose [position(3) + quaternion(4)].

    Args:
        env: Environment instance
        asset_cfg: Asset configuration

    Returns:
        [num_envs, 7] end-effector pose (pos + quat)
    """
    robot = env.scene[asset_cfg.name]
    kin_state = robot.root_physx_view.get_link_transforms()

    # Find EE link index
    if "arm_link_wr1" in robot.body_names:
        ee_idx = robot.body_names.index("arm_link_wr1")
    else:
        raise ValueError("End-effector link 'arm_link_wr1' not found")

    # Return [pos(3), quat(4)]
    return kin_state[:, ee_idx, :]


def ee_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Get end-effector position only.

    Args:
        env: Environment instance
        asset_cfg: Asset configuration

    Returns:
        [num_envs, 3] end-effector position
    """
    robot = env.scene[asset_cfg.name]
    kin_state = robot.root_physx_view.get_link_transforms()

    if "arm_link_wr1" in robot.body_names:
        ee_idx = robot.body_names.index("arm_link_wr1")
    else:
        raise ValueError("End-effector link 'arm_link_wr1' not found")

    return kin_state[:, ee_idx, :3]


def ee_orientation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Get end-effector orientation (quaternion).

    Args:
        env: Environment instance
        asset_cfg: Asset configuration

    Returns:
        [num_envs, 4] end-effector quaternion
    """
    robot = env.scene[asset_cfg.name]
    kin_state = robot.root_physx_view.get_link_transforms()

    if "arm_link_wr1" in robot.body_names:
        ee_idx = robot.body_names.index("arm_link_wr1")
    else:
        raise ValueError("End-effector link 'arm_link_wr1' not found")

    return kin_state[:, ee_idx, 3:]


def base_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Get robot base position.

    Args:
        env: Environment instance
        asset_cfg: Asset configuration

    Returns:
        [num_envs, 3] base position
    """
    robot = env.scene[asset_cfg.name]
    return robot.data.root_pos_w


def base_orientation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Get robot base orientation (quaternion).

    Args:
        env: Environment instance
        asset_cfg: Asset configuration

    Returns:
        [num_envs, 4] base quaternion
    """
    robot = env.scene[asset_cfg.name]
    return robot.data.root_quat_w


def camera_rgb(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera")) -> torch.Tensor:
    """
    Get RGB image from camera.

    Args:
        env: Environment instance
        sensor_cfg: Sensor configuration

    Returns:
        [num_envs, H, W, 3] RGB image
    """
    camera = env.scene[sensor_cfg.name]
    return camera.data.output["rgb"]


def camera_depth(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera")) -> torch.Tensor:
    """
    Get depth map from camera.

    Args:
        env: Environment instance
        sensor_cfg: Sensor configuration

    Returns:
        [num_envs, H, W, 1] depth map
    """
    camera = env.scene[sensor_cfg.name]
    return camera.data.output["distance_to_image_plane"]


def vlm_tracking_state(env: ManagerBasedEnv) -> torch.Tensor:
    """
    Get VLM tracking state (custom observation).

    Args:
        env: Environment instance

    Returns:
        [num_envs, 3] tracking state: [tracking_mode, consecutive_failures, fallback_active]
    """
    if hasattr(env, "state"):
        tracking_mode = float(env.state.vlm.tracking_mode)
        consecutive_failures = float(env.state.vlm.consecutive_failures)
        fallback_active = float(env.state.fallback.active)
    else:
        tracking_mode = 0.0
        consecutive_failures = 0.0
        fallback_active = 0.0

    # Stack into tensor
    state = torch.tensor(
        [[tracking_mode, consecutive_failures, fallback_active]],
        dtype=torch.float32,
        device=env.device,
    )

    # Repeat for all environments
    return state.repeat(env.num_envs, 1)


def arm_joint_positions_relative(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Get arm joint positions relative to default/initial configuration.

    Args:
        env: Environment instance
        asset_cfg: Asset configuration

    Returns:
        [num_envs, 6] relative arm joint positions
    """
    robot = env.scene[asset_cfg.name]
    arm_names = [n for n in robot.data.joint_names if n.startswith("arm")][:6]
    arm_indices = [robot.data.joint_names.index(n) for n in arm_names]

    current_pos = robot.data.joint_pos[:, arm_indices]
    default_pos = robot.data.default_joint_pos[:, arm_indices]

    return current_pos - default_pos
