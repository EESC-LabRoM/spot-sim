import torch
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab.sensors import CameraCfg, Camera

camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/arm_link_fngr/hand_cam",
    update_period=0,
    height=480,
    width=640,
    data_types=[
        "rgb",
        "distance_to_image_plane",
        "semantic_segmentation",
    ],  # RGB + Depth + Segmentation
    colorize_semantic_segmentation=True,
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=18.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 1000.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.05, 0.0, 0.0),
        rot=quat_from_euler_xyz(
            torch.tensor(torch.pi / 2),
            torch.tensor(torch.pi),
            torch.tensor(torch.pi / 2),
        ),
    ),
)
