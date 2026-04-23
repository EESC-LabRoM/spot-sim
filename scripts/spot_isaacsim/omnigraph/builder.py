"""
ROS 2 OmniGraph Action Graph builder for spot_isaacsim.

Publishes:
- /clock (Clock message — required for use_sim_time nodes)
- /joint_states (JointState message)
- /tf (TransformTree message)
- /spot/camera/{name}/image (RGB Image messages)
- /spot/depth_registered/{name}/image (Depth Image messages)
- /spot/camera/{name}/camera_info (RGB CameraInfo messages)
- /spot/depth_registered/{name}/camera_info (Depth CameraInfo messages)

ZED streaming (via ZED Camera Helper → ZED ROS2 Wrapper):
- Streams data to the ZED ROS2 Wrapper on zed_streaming_port (default 30000)
- Launch wrapper with: ros2 launch zed_wrapper zed_camera.launch.py \
      camera_model:=zedx sim_mode:=true use_sim_time:=true sim_port:=30000

Usage:
    bridge = ROSBridgeBuilder(robot=robot, cameras=cameras)
    if bridge.success:
        print("ROS2 bridge created!")
    else:
        print(f"Bridge error: {bridge.error}")
"""
import omni.graph.core as og
import usdrt.Sdf
from typing import Tuple, Optional, Dict, Any, List

from .config import BridgeConfig


class ROSBridgeBuilder:
    """ROS 2 Action Graph builder for Spot in native Isaac Sim."""

    def __init__(
        self,
        robot,
        cameras: Dict[str, Any],
        enable_joint_state: bool = True,
        enable_tf: bool = True,
        graph_path: str = "/ActionGraph_ROS",
        camera_step: int = 1,
        zed_prim_path: str = None,
        enable_zed: bool = True,
        zed_config=None,
    ):
        self.cameras = cameras or {}
        self.zed_prim_path = zed_prim_path
        self.zed_config = zed_config
        if enable_zed and zed_prim_path and zed_config is None:
            raise ValueError(
                "[ROSBridgeBuilder] enable_zed=True and zed_prim_path is set, "
                "but zed_config is None. Pass a ZedConfig instance."
            )
        self.config = BridgeConfig(
            cameras=self._make_camera_topics(),
            enable_joint_state=enable_joint_state,
            enable_tf=enable_tf,
            graph_path=graph_path,
            camera_step=camera_step,
            enable_zed=enable_zed,
        )

        self.robot_prim_path = robot.prim_path
        self.camera_paths = self._extract_all_camera_paths()
        self._validate_prims()

        self.success, self.error = self._build()

    def _make_camera_topics(self) -> List[Dict[str, str]]:
        """Generate default camera topic specs from camera dict."""
        topics = []
        for name, camera in self.cameras.items():
            cam_prim_name = camera.prim_path.split('/')[-1]
            topics.append({
                "name": name,
                "camera_key": name,
                "frame_id": cam_prim_name,
                "rgb_topic": f"/spot/camera/{name}/image",
                "depth_topic": f"/spot/depth_registered/{name}/image",
                "rgb_camera_info_topic": f"/spot/camera/{name}/camera_info",
                "depth_camera_info_topic": f"/spot/depth_registered/{name}/camera_info",
            })
        return topics

    def _build(self) -> Tuple[bool, Optional[str]]:
        """Build the ROS bridge action graph."""
        try:
            success, error = self._ensure_graph_container()
            if not success:
                return False, error

            all_create_nodes, all_connect, all_set_values = [], [], []

            for spec_fn in [self._get_common_nodes_spec]:
                create, connect, values = spec_fn()
                all_create_nodes.extend(create)
                all_connect.extend(connect)
                all_set_values.extend(values)

            if self.config.enable_joint_state:
                create, connect, values = self._get_joint_state_spec()
                all_create_nodes.extend(create)
                all_connect.extend(connect)
                all_set_values.extend(values)

            if self.config.enable_tf:
                create, connect, values = self._get_tf_spec()
                all_create_nodes.extend(create)
                all_connect.extend(connect)
                all_set_values.extend(values)

            for cam_spec in self.config.cameras:
                camera_key = cam_spec.get("camera_key", cam_spec["name"])
                if camera_key not in self.camera_paths:
                    print(f"[WARN] Camera '{cam_spec['name']}' not found, skipping")
                    continue

                _, render_path = self.camera_paths[camera_key]

                create, connect, values = self._get_camera_specs(
                    camera_name=cam_spec["name"],
                    render_path=render_path,
                    frame_id=cam_spec["frame_id"],
                    rgb_topic=cam_spec["rgb_topic"],
                    depth_topic=cam_spec["depth_topic"],
                )
                all_create_nodes.extend(create)
                all_connect.extend(connect)
                all_set_values.extend(values)

                rgb_info_topic = cam_spec.get("rgb_camera_info_topic")
                depth_info_topic = cam_spec.get("depth_camera_info_topic")
                if rgb_info_topic and depth_info_topic:
                    create, connect, values = self._get_camera_info_specs(
                        camera_name=cam_spec["name"],
                        render_path=render_path,
                        frame_id=cam_spec["frame_id"],
                        rgb_info_topic=rgb_info_topic,
                        depth_info_topic=depth_info_topic,
                    )
                    all_create_nodes.extend(create)
                    all_connect.extend(connect)
                    all_set_values.extend(values)

            if self.config.enable_zed and self.zed_prim_path and self.zed_config:
                from .zed import get_zed_spec, get_zed_tf_spec
                for fn, kwargs in [
                    (get_zed_spec, {"zed_prim_path": self.zed_prim_path, "config": self.zed_config}),
                    (get_zed_tf_spec, {"config": self.zed_config}),
                ]:
                    create, connect, values = fn(**kwargs)
                    all_create_nodes.extend(create)
                    all_connect.extend(connect)
                    all_set_values.extend(values)

            keys = og.Controller.Keys
            og.Controller.edit(
                {"graph_path": self.config.graph_path},
                {
                    keys.CREATE_NODES: all_create_nodes,
                    keys.CONNECT: all_connect,
                    keys.SET_VALUES: all_set_values,
                },
            )
            return True, None

        except Exception as e:
            return False, f"Failed to create graph: {e}"

    # =========================================================================
    # Path Extraction / Validation
    # =========================================================================

    def _extract_all_camera_paths(self) -> Dict[str, Tuple[str, str]]:
        """Extract (prim_path, render_product_path) for each camera."""
        camera_paths = {}
        for name, camera in self.cameras.items():
            try:
                camera_paths[name] = (camera.prim_path, camera.get_render_product_path())
            except Exception as e:
                print(f"[WARN] Failed to get paths for camera '{name}': {e}")
        return camera_paths

    def _validate_prims(self) -> None:
        """Validate that robot and camera prims exist in USD stage."""
        import omni.usd
        stage = omni.usd.get_context().get_stage()

        if not stage.GetPrimAtPath(self.robot_prim_path).IsValid():
            raise ValueError(f"Robot prim invalid at: {self.robot_prim_path}")

        for name, (camera_prim_path, _) in self.camera_paths.items():
            if not stage.GetPrimAtPath(camera_prim_path).IsValid():
                print(f"[WARN] Camera prim invalid at: {camera_prim_path} (name: {name})")

    def _ensure_graph_container(self) -> Tuple[bool, Optional[str]]:
        """Remove existing graph (if any) so we start clean."""
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            existing = stage.GetPrimAtPath(self.config.graph_path)
            if existing.IsValid():
                stage.RemovePrim(self.config.graph_path)
            return True, None
        except Exception as e:
            return False, f"Failed to ensure graph container: {e}"

    # =========================================================================
    # Node Specification Methods
    # =========================================================================

    def _get_common_nodes_spec(self) -> Tuple[List, List, List]:
        create_nodes = [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("ReadSimTime",    "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("PubClock",       "isaacsim.ros2.bridge.ROS2PublishClock"),
            ("CameraGate",     "isaacsim.core.nodes.IsaacSimulationGate"),
        ]
        connect = [
            ("OnPlaybackTick.outputs:tick", "PubClock.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", "PubClock.inputs:timeStamp"),
            ("OnPlaybackTick.outputs:tick", "CameraGate.inputs:execIn"),
        ]
        set_values = [
            ("CameraGate.inputs:step", self.config.camera_step),
        ]
        return create_nodes, connect, set_values

    def _get_joint_state_spec(self) -> Tuple[List, List, List]:
        create_nodes = [("PubJointState", "isaacsim.ros2.bridge.ROS2PublishJointState")]
        connect = [
            ("OnPlaybackTick.outputs:tick", "PubJointState.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", "PubJointState.inputs:timeStamp"),
        ]
        set_values = [
            ("PubJointState.inputs:targetPrim", [usdrt.Sdf.Path(self.robot_prim_path)]),
            ("PubJointState.inputs:topicName", self.config.joint_state_topic),
        ]
        return create_nodes, connect, set_values

    def _get_tf_spec(self) -> Tuple[List, List, List]:
        create_nodes = [("PubTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree")]
        connect = [
            ("OnPlaybackTick.outputs:tick", "PubTF.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", "PubTF.inputs:timeStamp"),
        ]
        target_prims = [usdrt.Sdf.Path(self.robot_prim_path)]
        for camera_prim_path, _ in self.camera_paths.values():
            target_prims.append(usdrt.Sdf.Path(camera_prim_path))
        set_values = [
            ("PubTF.inputs:targetPrims", target_prims),
            ("PubTF.inputs:topicName", self.config.tf_topic),
        ]
        return create_nodes, connect, set_values

    def _get_camera_specs(
        self,
        camera_name: str,
        render_path: str,
        frame_id: str,
        rgb_topic: str,
        depth_topic: str,
    ) -> Tuple[List, List, List]:
        rgb_node = f"PubCamera_{camera_name}_rgb"
        depth_node = f"PubCamera_{camera_name}_depth"
        create_nodes = [
            (rgb_node, "isaacsim.ros2.bridge.ROS2CameraHelper"),
            (depth_node, "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ]
        connect = [
            ("CameraGate.outputs:execOut", f"{rgb_node}.inputs:execIn"),
            ("CameraGate.outputs:execOut", f"{depth_node}.inputs:execIn"),
        ]
        set_values = [
            (f"{rgb_node}.inputs:frameId", frame_id),
            (f"{rgb_node}.inputs:topicName", rgb_topic),
            (f"{rgb_node}.inputs:renderProductPath", render_path),
            (f"{rgb_node}.inputs:type", "rgb"),
            (f"{depth_node}.inputs:frameId", frame_id),
            (f"{depth_node}.inputs:topicName", depth_topic),
            (f"{depth_node}.inputs:renderProductPath", render_path),
            (f"{depth_node}.inputs:type", "depth"),
        ]
        return create_nodes, connect, set_values

    def _get_camera_info_specs(
        self,
        camera_name: str,
        render_path: str,
        frame_id: str,
        rgb_info_topic: str,
        depth_info_topic: str,
    ) -> Tuple[List, List, List]:
        rgb_node = f"PubCameraInfo_{camera_name}_rgb"
        depth_node = f"PubCameraInfo_{camera_name}_depth"
        create_nodes = [
            (rgb_node, "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            (depth_node, "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
        ]
        connect = [
            ("CameraGate.outputs:execOut", f"{rgb_node}.inputs:execIn"),
            ("CameraGate.outputs:execOut", f"{depth_node}.inputs:execIn"),
        ]
        set_values = [
            (f"{rgb_node}.inputs:frameId", frame_id),
            (f"{rgb_node}.inputs:topicName", rgb_info_topic),
            (f"{rgb_node}.inputs:renderProductPath", render_path),
            (f"{depth_node}.inputs:frameId", frame_id),
            (f"{depth_node}.inputs:topicName", depth_info_topic),
            (f"{depth_node}.inputs:renderProductPath", render_path),
        ]
        return create_nodes, connect, set_values
