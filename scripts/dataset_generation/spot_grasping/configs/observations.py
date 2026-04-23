"""
Observation configurations for Spot Grasping environment.

Simplified version without camera - data collection happens in separate environment.
"""

import torch
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
# from isaaclab.utils.math import quat_apply, quat_conjugate

from dataset_generation import Colors


def get_pos(env, asset_cfg: RigidObjectCfg, debug: bool = False):
    """get asset position in ??? frame"""
    asset = env.scene[asset_cfg.name]
    asset_pose = asset.data.root_pose_w
    asset_pos = asset_pose[:, 0:3]
    asset_root_pos = asset.data.root_link_pos_w
    if debug:
        print(f"\n{Colors.YELLOW}[DEBUG POS] {asset_pos[0]}{Colors.END}")
        print(f"{Colors.YELLOW}[DEBUG ROOT POS]\n {asset_root_pos[0]}{Colors.END}")
    return asset_pos


def joint_pos_rel(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint positions relative to default position."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos - asset.data.default_joint_pos


def joint_vel_rel(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint velocities."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_vel


@configclass
class ObservationsCfg:
    """Observation specifications for the grasping environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos_rel = ObsTerm(func=joint_pos_rel)
        joint_vel_rel = ObsTerm(func=joint_vel_rel)
        object_pos = ObsTerm(
            func=get_pos,
            params={
                "asset_cfg": SceneEntityCfg("target_object"),
                "debug": True,
            },
        )

        # TODO: debug wth
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
