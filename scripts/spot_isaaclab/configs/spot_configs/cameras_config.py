"""
Camera configuration for Spot robot
Defines camera sensors attached to the robot
"""
import torch
import numpy as np
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_from_euler_xyz
from ..global_config import USE_URDF
arm_prefix = "arm" if USE_URDF else "arm0"

# Helper function to convert RPY to quaternion list
def rpy_to_quat_list(roll, pitch, yaw):
    """Convert RPY angles (in radians) to quaternion list [w, x, y, z]"""
    quat = quat_from_euler_xyz(
        torch.tensor(roll),
        torch.tensor(pitch),
        torch.tensor(yaw)
    )
    return quat.tolist()

# Calibrated fisheye camera parameters (from rosbag data)
# Averaged from front left (fx=330.56, fy=330.33) and front right (fx=330.93, fy=330.59)
# All Spot fisheye cameras use the same hardware model
FISHEYE_FOCAL_LENGTH = 0.517  # cm (calculated from avg focal length in pixels)
FISHEYE_HORIZONTAL_APERTURE = 1.0  # cm (640 pixels * pixel_size)
FISHEYE_VERTICAL_APERTURE = 0.75  # cm (480 pixels * pixel_size)

"""
TOGGLE IT TO SWIT
"""
USE_FISHEYE_CFG = False

if USE_FISHEYE_CFG:
  BODY_CAMS_CONFIG = sim_utils.FisheyeCameraCfg(
        projection_type="fisheyePolynomial",
        focal_length=FISHEYE_FOCAL_LENGTH,
        horizontal_aperture=FISHEYE_HORIZONTAL_APERTURE,
        vertical_aperture=FISHEYE_VERTICAL_APERTURE,
        clipping_range=(0.01, 1000.0),
        fisheye_nominal_width=640.0,
        fisheye_nominal_height=480.0,
        fisheye_max_fov=220.0,
        fisheye_polynomial_b=0.00245,
    )
else:
  BODY_CAMS_CONFIG = sim_utils.PinholeCameraCfg(
        focal_length=18.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 10.0),
    )


hand_camera_config =  CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/" + arm_prefix + "_link_wr1/hand_cam",
    update_period=0,
    height=336,
    width=336,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=18.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 10.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.15, 0.0, 0.02),
        rot=rpy_to_quat_list(np.pi/2, np.pi, np.pi / 2)
    )
)

# Fisheye cameras mounted on robot body
# Front left fisheye camera (calibrated)
frontleft_camera_config = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/frontleft_fisheye/camera",
    update_period=0,
    height=640,
    width=480,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=BODY_CAMS_CONFIG,
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.01),
        rot=rpy_to_quat_list(np.pi, np.pi, np.pi / 2)
    )
)

# Front right fisheye camera (calibrated)
frontright_camera_config = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/frontright_fisheye/camera",
    update_period=0,
    height=640,
    width=480,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=BODY_CAMS_CONFIG,
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.01),
        rot=rpy_to_quat_list(np.pi, np.pi, np.pi / 2)
    )
)

# Left fisheye camera (calibrated - same hardware as front cameras)
left_camera_config = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_fisheye/camera",
    update_period=0,
    height=480,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=BODY_CAMS_CONFIG,
    offset=CameraCfg.OffsetCfg(
        pos=(0.05, 0.0, 0.02),
        rot=rpy_to_quat_list(np.pi / 2, np.pi, np.pi / 2)
    )
)

# Right fisheye camera (calibrated - same hardware as front cameras)
right_camera_config = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_fisheye/camera",
    update_period=0,
    height=480,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=BODY_CAMS_CONFIG,
    offset=CameraCfg.OffsetCfg(
        pos=(0.05, 0.0, 0.02),
        rot=rpy_to_quat_list(np.pi / 2, np.pi, np.pi / 2)
    )
)

# Back fisheye camera (calibrated - same hardware as front cameras)
back_camera_config = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/back_fisheye/camera",
    update_period=0,
    height=480,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=BODY_CAMS_CONFIG,
    offset=CameraCfg.OffsetCfg(
        pos=(0.06, 0.0, 0.0),
        rot=rpy_to_quat_list(np.pi / 2, np.pi, np.pi / 2)
    )
)
