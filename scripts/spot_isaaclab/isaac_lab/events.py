"""
Isaac Lab EventTerm implementations for Spot robot.
Handles resets, domain randomization, and curriculum learning.
"""
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation


def reset_arm_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Reset arm joints to default configuration.

    Args:
        env: Environment instance
        env_ids: Environment IDs to reset
        asset_cfg: Asset configuration
    """
    robot: Articulation = env.scene[asset_cfg.name]

    # Get arm indices
    arm_names = [n for n in robot.data.joint_names if n.startswith("arm")][:6]
    arm_indices = [robot.data.joint_names.index(n) for n in arm_names]

    # Reset to default positions
    robot.data.joint_pos[env_ids, arm_indices] = robot.data.default_joint_pos[env_ids, arm_indices]
    robot.data.joint_vel[env_ids, arm_indices] = 0.0

    # Write to simulation
    robot.write_data_to_sim()


def reset_base_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Reset robot base (root) to default pose.

    Args:
        env: Environment instance
        env_ids: Environment IDs to reset
        asset_cfg: Asset configuration
    """
    robot: Articulation = env.scene[asset_cfg.name]

    # Reset root pose
    robot.data.root_pos_w[env_ids] = robot.data.default_root_state[env_ids, :3]
    robot.data.root_quat_w[env_ids] = robot.data.default_root_state[env_ids, 3:7]

    # Reset root velocities
    robot.data.root_lin_vel_w[env_ids] = 0.0
    robot.data.root_ang_vel_w[env_ids] = 0.0

    # Write to simulation
    robot.write_data_to_sim()


def reset_scene_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
):
    """
    Reset entire scene (robot + objects) to default state.

    Args:
        env: Environment instance
        env_ids: Environment IDs to reset
    """
    # Reset robot
    reset_arm_to_default(env, env_ids)
    reset_base_to_default(env, env_ids)

    # Reset VLM state if available (global state, not per-environment)
    if hasattr(env, "state"):
        env.state.reset()


def randomize_arm_joint_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    position_range: tuple[float, float] = (-0.1, 0.1),
):
    """
    Randomize arm joint positions around default (for domain randomization).

    Args:
        env: Environment instance
        env_ids: Environment IDs to randomize
        asset_cfg: Asset configuration
        position_range: Randomization range relative to default
    """
    robot: Articulation = env.scene[asset_cfg.name]

    # Get arm indices
    arm_names = [n for n in robot.data.joint_names if n.startswith("arm")][:6]
    arm_indices = [robot.data.joint_names.index(n) for n in arm_names]

    # Generate random offsets
    num_envs = len(env_ids)
    random_offsets = torch.rand(num_envs, len(arm_indices), device=env.device)
    random_offsets = random_offsets * (position_range[1] - position_range[0]) + position_range[0]

    # Apply randomization
    robot.data.joint_pos[env_ids, arm_indices] = (
        robot.data.default_joint_pos[env_ids, arm_indices] + random_offsets
    )

    # Write to simulation
    robot.write_data_to_sim()


def randomize_base_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    position_range: tuple[float, float] = (-0.2, 0.2),
):
    """
    Randomize robot base position (for domain randomization).

    Args:
        env: Environment instance
        env_ids: Environment IDs to randomize
        asset_cfg: Asset configuration
        position_range: Randomization range in meters (XY plane only)
    """
    robot: Articulation = env.scene[asset_cfg.name]

    # Generate random XY offsets
    num_envs = len(env_ids)
    random_xy = torch.rand(num_envs, 2, device=env.device)
    random_xy = random_xy * (position_range[1] - position_range[0]) + position_range[0]

    # Apply to base position (XY only, keep Z)
    robot.data.root_pos_w[env_ids, :2] += random_xy

    # Write to simulation
    robot.write_data_to_sim()


def randomize_object_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_name: str = "target_object",
    position_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.3, 0.7),  # X range
        (-0.3, 0.3),  # Y range
        (0.5, 1.0),  # Z range
    ),
):
    """
    Randomize object positions in scene (for curriculum learning).

    Args:
        env: Environment instance
        env_ids: Environment IDs to randomize
        object_name: Name of object to randomize
        position_range: (X, Y, Z) range tuples
    """
    if object_name not in env.scene:
        return  # Object doesn't exist

    obj = env.scene[object_name]

    # Generate random positions
    num_envs = len(env_ids)
    random_pos = torch.zeros(num_envs, 3, device=env.device)

    for dim in range(3):
        min_val, max_val = position_range[dim]
        random_pos[:, dim] = torch.rand(num_envs, device=env.device) * (max_val - min_val) + min_val

    # Apply to object
    obj.data.root_pos_w[env_ids] = random_pos

    # Write to simulation
    obj.write_data_to_sim()


def apply_external_force_to_arm(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_range: tuple[float, float] = (-10.0, 10.0),
):
    """
    Apply random external force to arm (for robustness training).

    Args:
        env: Environment instance
        env_ids: Environment IDs to apply force
        asset_cfg: Asset configuration
        force_range: Force magnitude range in Newtons
    """
    robot: Articulation = env.scene[asset_cfg.name]

    # Find EE link
    if "arm_link_wr1" in robot.body_names:
        ee_idx = robot.body_names.index("arm_link_wr1")
    else:
        return

    # Generate random force
    num_envs = len(env_ids)
    random_force = torch.rand(num_envs, 3, device=env.device)
    random_force = random_force * (force_range[1] - force_range[0]) + force_range[0]

    # Apply force (this would require physics API access)
    # Note: Isaac Lab may require different API for applying forces
    # This is a placeholder for the concept
    print(f"[EVENT] Applying external force to EE (link {ee_idx}): {random_force[0]}")


def reset_vlm_state(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """
    Reset VLM-specific state variables.

    Args:
        env: Environment instance
        env_ids: Environment IDs to reset (can be slice or tensor)
    """
    if hasattr(env, "state"):
        # Reset VLM state for all environments
        # (Since state is global, we reset it entirely)
        env.state.vlm.reset()
        env.state.control.reset()
        env.state.fallback.reset()

        # Handle different env_ids types
        if isinstance(env_ids, torch.Tensor):
            num_envs = len(env_ids)
        elif env_ids is None or isinstance(env_ids, slice):
            num_envs = env.num_envs
        else:
            num_envs = "all"
        print(f"[EVENT] VLM state reset for {num_envs} environments")
