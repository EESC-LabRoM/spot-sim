"""
Locomotion collision avoidance filter.

Reads body-facing depth cameras at a configurable rate and removes the
velocity component pointing toward any detected obstacle from the locomotion
command, then rescales to preserve the original travel speed.

Usage:
    ca = LocomotionCollisionAvoidance(
        cameras=cameras,
        robot_prim_path=config.robot.prim_path,
        check_frequency_hz=10.0,
        obstacle_distance_m=0.30,
        physics_dt=config.physics_dt,
    )
    locomotion.set_collision_avoidance(ca)
"""

from typing import Dict, List
import numpy as np
import torch

# Body-frame blocking directions per camera name.
# Spot body frame: X = forward, Y = left.
_CAMERA_LOOKATS: Dict[str, List[float]] = {
    "frontleft":  [ 0.74757313,  -0.66417951],   # front face → block +X
    "frontright": [ 0.74757313,  0.66417951],   # front face → block +X
    "left":       [ 0.0,  1.0],   # left face  → block +Y
    "right":      [ 0.0, -1.0],   # right face → block -Y
    "rear":       [-1.0,  0.0],   # rear face  → block -X
}


class LocomotionCollisionAvoidance:
    """Depth-camera-based velocity filter for locomotion commands.

    At each `check_frequency_hz` interval, scans the body-facing depth cameras.
    For every camera that sees an object closer than `obstacle_distance_m`, the
    component of the incoming [vx, vy, wz] command that points toward that camera
    is removed.  The remaining XY velocity is rescaled to the original magnitude
    so the robot continues at normal speed in the safe direction.

    The yaw component (wz) is never modified — the robot can always rotate.

    Depth tensors stay on `device` throughout; only the final 2-element XY result
    is transferred back to CPU at the numpy boundary.
    """

    # Body-mounted cameras only; hand camera (on arm) is excluded
    BODY_CAMERA_NAMES = {"frontleft", "frontright", "left", "right", "rear"}

    def __init__(
        self,
        cameras: Dict[str, "Camera"],
        obstacle_distance_m: float = 0.30,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            cameras: Full cameras dict from play.py (may include hand camera).
            obstacle_distance_m: Closest allowed depth reading before avoidance
                activates (metres).
            device: Torch device for depth tensors and math ("cpu" or "cuda:0").
        """
        self._cameras: Dict[str, "Camera"] = {
            k: v for k, v in cameras.items() if k in self.BODY_CAMERA_NAMES
        }
        self._threshold: float = obstacle_distance_m
        self._device: str = device

        # Pre-allocate lookat tensors once — reused every call
        self._lookats: Dict[str, torch.Tensor] = {
            name: torch.tensor(_CAMERA_LOOKATS[name], dtype=torch.float64, device=device)
            for name in self._cameras
            if name in _CAMERA_LOOKATS
        }

        self._blocked_lookats: List[torch.Tensor] = []

        print(
            f"[COLLISION_AVOID] Initialized — "
            f"{len(self._cameras)} body cameras, "
            f"threshold={obstacle_distance_m:.2f} m, device={device}, check every step"
        )
        for name, lookat in self._lookats.items():
            print(f"[COLLISION_AVOID]   {name}: lookat_body_xy={lookat.tolist()}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Cameras whose bottom rows are cropped to ignore feet during locomotion.
    _FOOT_CROP_CAMERAS = {"frontleft", "frontright"}
    _FOOT_CROP_FRACTION = 0.20  # ignore bottom

    def _check_obstacles(self) -> None:
        """Sample all body cameras and update the blocked-lookat list."""
        blocked: List[torch.Tensor] = []
        for name, camera in self._cameras.items():
            depth = camera.get_depth(device=self._device)
            if depth is None:
                continue
            if not isinstance(depth, torch.Tensor):
                import warp as wp
                depth = wp.to_torch(depth).float()   # zero-copy via __cuda_array_interface__
            if name in self._FOOT_CROP_CAMERAS and depth.ndim >= 2:
                crop_rows = int(depth.shape[0] * (1.0 - self._FOOT_CROP_FRACTION))
                depth = depth[:crop_rows, :]
            valid = depth[torch.isfinite(depth) & (depth > 0.0)]
            if valid.numel() > 0 and valid.min().item() < self._threshold:
                blocked.append(self._lookats[name])
                print(f"[COLLISION_AVOID] obstacle detected by '{name}' camera "
                      f"(min depth={valid.min().item():.3f} m)")
        self._blocked_lookats = blocked

    def apply(self, command: np.ndarray) -> np.ndarray:
        """Filter a locomotion command to avoid obstacles.

        Should be called every physics step.

        Args:
            command: Shape-(3,) array [vx, vy, wz] in robot body frame.

        Returns:
            Shape-(3,) array with the obstacle-toward component of XY velocity
            removed and the remaining XY velocity rescaled to the original
            magnitude.  The yaw component (index 2) is never modified.
        """
        self._check_obstacles()

        if not self._blocked_lookats:
            return command

        cmd = torch.tensor(command[:2], dtype=torch.float64, device=self._device)
        original_mag = cmd.norm()

        for lookat in self._blocked_lookats:
            proj = torch.dot(cmd, lookat)
            if proj > 0.0:
                cmd = cmd - proj * lookat

        safe_mag = cmd.norm()
        if original_mag > 1e-6:
            if safe_mag > 1e-6:
                cmd = cmd * (original_mag / safe_mag)   # preserve travel speed
            else:
                cmd = torch.zeros(2, device=self._device)   # fully blocked — stop XY, wz active

        # Return numpy at the boundary; only 8 bytes cross back to CPU
        result = command.copy()
        result[:2] = cmd.cpu().numpy()
        return result.astype(np.float32)
