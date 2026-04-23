"""
ZED Camera Helper OmniGraph node spec for the ROS2 bridge.

The ZED Camera Helper streams simulated ZED data to the ZED ROS2 Wrapper,
which must be launched separately with:
    ros2 launch zed_wrapper zed_camera.launch.py \
        camera_model:=zedx sim_mode:=true use_sim_time:=true sim_port:=<streaming_port>

Node type and input names sourced from the OGN definition:
    zed-isaac-sim/exts/sl.sensor.camera/sl/sensor/camera/nodes/SlCameraStreamer.ogn
"""
import math
import usdrt.Sdf
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.spot_isaacsim.spot_config.cameras.mounted_zed import ZedConfig


def get_zed_spec(
    zed_prim_path: str,
    config: "ZedConfig",
) -> Tuple[List, List, List]:
    """
    Return (create_nodes, connect, set_values) spec for the ZED Camera Helper node.

    Connects to OnPlaybackTick (must already exist in the graph).

    Args:
        zed_prim_path:  Full USD path to the top-level ZED prim,
                        e.g. "/World/Robot/body/ZED_X"
        config:         ZedConfig — streaming_port and transport_mode are read from here.
    """
    create_nodes = [
        ("ZEDCameraHelper", "sl.sensor.camera.ZED_Camera"),
    ]
    connect = [
        ("OnPlaybackTick.outputs:tick", "ZEDCameraHelper.inputs:execIn"),
    ]
    set_values = [
        ("ZEDCameraHelper.inputs:cameraPrim", [usdrt.Sdf.Path(zed_prim_path)]),
        ("ZEDCameraHelper.inputs:streamingPort", config.streaming_port),
        ("ZEDCameraHelper.inputs:transportLayerMode", config.transport_mode),
    ]
    return create_nodes, connect, set_values


def get_zed_tf_spec(config: "ZedConfig") -> Tuple[List, List, List]:
    """
    Return (create_nodes, connect, set_values) spec for a static TF publisher.

    Publishes the mount transform from the last segment of config.reference_frame
    (e.g. "body") to config.ros_frame_id (e.g. "zed_camera_link") on /tf_static.

    Connects to OnPlaybackTick and ReadSimTime (must already exist in the graph).

    Args:
        config: ZedConfig — reference_frame, translation, rotation_rpy, and
                ros_frame_id are all read from here.
    """
    r, p, y = config.rotation_rpy
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    parent_frame = config.reference_frame.split("/")[-1]

    create_nodes = [("ZEDStaticTF", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree")]
    connect = [
        ("OnPlaybackTick.outputs:tick", "ZEDStaticTF.inputs:execIn"),
        ("ReadSimTime.outputs:simulationTime", "ZEDStaticTF.inputs:timeStamp"),
    ]
    set_values = [
        ("ZEDStaticTF.inputs:parentFrameId", parent_frame),
        ("ZEDStaticTF.inputs:childFrameId", config.ros_frame_id),
        ("ZEDStaticTF.inputs:translation", list(config.translation)),
        ("ZEDStaticTF.inputs:rotation", [qx, qy, qz, qw]),
        ("ZEDStaticTF.inputs:staticPublisher", True),
        ("ZEDStaticTF.inputs:topicName", "tf_static"),
    ]
    return create_nodes, connect, set_values
