from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class BodyPoseCmd:
    x: float
    y: float
    yaw: float


@dataclass
class ArmPoseCmd:
    position: np.ndarray
    orientation: np.ndarray | None = None


@dataclass
class GraspCmd:
    """Pre-computed grasp geometry. Build via GraspCmd.from_raw() for Python callers."""
    pre_grasp_pos: np.ndarray
    grasp_pos: np.ndarray
    orientation: np.ndarray          # wxyz
    chain_grasp: bool = True
    chain_retrieve: bool = False

    @classmethod
    def from_raw(
        cls,
        position: np.ndarray,
        orientation: np.ndarray,
        ee_offset: np.ndarray,
        approach_dist: float,
        do_pick: bool = False,
    ) -> "GraspCmd":
        """Build GraspCmd from raw target position + wxyz orientation.
        Applies ee_offset in the grasp frame and backs off by approach_dist.
        """
        from scripts.spot_isaacsim.utils.math import advance_along_local_x
        R         = Rotation.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]])
        grasp_pos = position.copy() + R.as_matrix() @ ee_offset
        pre_grasp_pos = advance_along_local_x(grasp_pos, orientation, -approach_dist)
        return cls(
            pre_grasp_pos=pre_grasp_pos,
            grasp_pos=grasp_pos,
            orientation=orientation.copy(),
            chain_grasp=True,
            chain_retrieve=do_pick,
        )


@dataclass
class TrajectoryCmd:
    points: list                      # list of trajectory_msgs/JointTrajectoryPoint
    joint_indices: np.ndarray
    chain_pick: bool = False


@dataclass
class HomeCmd:
    pass
