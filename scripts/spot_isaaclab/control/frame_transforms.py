"""
Frame transformation utilities (Camera ↔ EE ↔ World).
Handles coordinate transformations with camera rotation compensation.

Camera rotation relative to EE link: (π/2, π, π/2)
This means:
- Camera +X (forward) → EE -Z
- Camera +Y (right) → EE -X
- Camera +Z (up) → EE +Y
"""
import torch
import numpy as np
from isaaclab.utils.math import quat_from_euler_xyz, quat_inv, quat_apply


class FrameTransformer:
    """Handles coordinate frame transformations for Spot arm"""

    # Camera rotation relative to EE: (pi/2, pi, pi/2)
    CAMERA_TO_EE_EULER = (np.pi / 2, np.pi, np.pi / 2)

    def __init__(self, device="cuda:0"):
        """
        Initialize frame transformer.

        Args:
            device: Torch device
        """
        self.device = device
        self._camera_to_ee_quat = None  # Lazy initialization (precomputed)
        self._ee_to_camera_quat = None  # Cached inverse

    @property
    def camera_to_ee_quat(self) -> torch.Tensor:
        """
        Quaternion for camera → EE transform.
        Precomputed once to avoid repeated tensor allocations.

        Returns:
            Quaternion [4] representing camera rotation relative to EE
        """
        if self._camera_to_ee_quat is None:
            self._camera_to_ee_quat = quat_from_euler_xyz(
                torch.tensor([self.CAMERA_TO_EE_EULER[0]], device=self.device),
                torch.tensor([self.CAMERA_TO_EE_EULER[1]], device=self.device),
                torch.tensor([self.CAMERA_TO_EE_EULER[2]], device=self.device),
            ).squeeze(0)
        return self._camera_to_ee_quat

    @property
    def ee_to_camera_quat(self) -> torch.Tensor:
        """
        Inverse quaternion for EE → camera transform.
        Precomputed and cached.

        Returns:
            Quaternion [4] representing inverse rotation
        """
        if self._ee_to_camera_quat is None:
            self._ee_to_camera_quat = quat_inv(self.camera_to_ee_quat.unsqueeze(0)).squeeze(0)
        return self._ee_to_camera_quat

    def camera_displacement_to_world(
        self, camera_disp: torch.Tensor, ee_quat: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform displacement from camera frame to world frame.

        Args:
            camera_disp: [3] displacement in camera frame
            ee_quat: [4] EE orientation in world frame

        Returns:
            [3] displacement in world frame
        """
        # Ensure correct device
        camera_disp = camera_disp.to(self.device)
        ee_quat = ee_quat.to(self.device)

        # Camera → EE frame
        ee_disp = quat_apply(
            self.ee_to_camera_quat.unsqueeze(0), camera_disp.unsqueeze(0)
        ).squeeze(0)

        # EE → World frame
        if ee_quat.dim() == 1:
            ee_quat = ee_quat.unsqueeze(0)
        world_disp = quat_apply(ee_quat, ee_disp.unsqueeze(0)).squeeze(0)

        return world_disp

    def camera_displacement_to_ee(self, camera_disp: torch.Tensor) -> torch.Tensor:
        """
        Transform displacement from camera frame to EE frame.

        Args:
            camera_disp: [3] displacement in camera frame

        Returns:
            [3] displacement in EE frame
        """
        camera_disp = camera_disp.to(self.device)
        ee_disp = quat_apply(
            self.ee_to_camera_quat.unsqueeze(0), camera_disp.unsqueeze(0)
        ).squeeze(0)
        return ee_disp

    def ee_displacement_to_world(
        self, ee_disp: torch.Tensor, ee_quat: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform displacement from EE frame to world frame.

        Args:
            ee_disp: [3] displacement in EE frame
            ee_quat: [4] EE orientation in world frame

        Returns:
            [3] displacement in world frame
        """
        ee_disp = ee_disp.to(self.device)
        ee_quat = ee_quat.to(self.device)

        if ee_quat.dim() == 1:
            ee_quat = ee_quat.unsqueeze(0)

        world_disp = quat_apply(ee_quat, ee_disp.unsqueeze(0)).squeeze(0)
        return world_disp
