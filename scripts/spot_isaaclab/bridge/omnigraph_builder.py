"""
ROS 2 Omnigraph Action Graph builder for NVBlox integration.
Refactored to use single og.Controller.edit() call for efficiency.

Publishes:
- /joint_states (JointState message)
- /tf (TransformTree message)
- /spot/camera/hand/image (RGB Image message)
- /spot/camera/hand/depth (Depth Image message)
- /robot_description (String message, URDF content - optional)

Usage:
    # Basic usage (all topics enabled)
    builder = ROSBridgeBuilder(robot=robot, scene=scene)
    success, error = builder.build()

    # Selective topics
    builder = ROSBridgeBuilder(
        robot=robot,
        scene=scene,
        enable_depth_camera=False,
        enable_tf=False
    )
    success, error = builder.build()

    # With URDF publishing
    builder = ROSBridgeBuilder(
        robot=robot,
        scene=scene,
        enable_robot_description=True,
        urdf_path="/path/to/robot.urdf"
    )
    success, error = builder.build()

    # Backward compatibility (legacy)
    success, error = ROSBridgeBuilder.create_nvblox_graph(robot, scene)
"""
import omni.graph.core as og
import usdrt.Sdf
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List


@dataclass
class BridgeConfig:
    """Configuration for ROS bridge topics"""

    enable_joint_state: bool = True
    enable_tf: bool = True
    enable_robot_description: bool = False

    # Camera configuration - list of camera specs
    cameras: Optional[List[Dict[str, str]]] = None  # [{"name": "hand", "scene_key": "robot_camera_hand", "frame_id": "hand_camera", "rgb_topic": "...", "depth_topic": "..."}, ...]

    joint_state_topic: str = "/joint_states"
    tf_topic: str = "/tf"
    robot_description_topic: str = "/robot_description"
    graph_path: str = "/ActionGraph_ROS"
    urdf_path: Optional[str] = None
    robot_description_qos_profile: str = '{"durability": "transientLocal", "reliability": "reliable", "history": "keepLast", "depth": 1, "deadline": 0.0, "lifespan": 0.0, "liveliness": "automatic", "leaseDuration": 0.0}'  # QoS for RViz2 compatibility (transient local = latching)

    # Legacy fields (deprecated but kept for backward compatibility)
    enable_rgb_camera: bool = True
    enable_depth_camera: bool = True
    rgb_topic: str = "/spot/camera/hand/image"
    depth_topic: str = "/spot/camera/hand/depth"
    camera_frame_id: str = "spot_camera_frame"


class ROSBridgeBuilder:
    """
    Instance-based ROS 2 Action Graph builder for Spot NVBlox integration.

    Uses single og.Controller.edit() call for efficient graph creation.
    Supports optional topic configuration and fail-fast error handling.

    Example:
        # Basic usage (all topics enabled)
        builder = ROSBridgeBuilder(robot, scene)
        success, error = builder.build()

        # Custom configuration
        builder = ROSBridgeBuilder(
            robot, scene,
            enable_depth_camera=False,
            enable_rgb_camera=True
        )
        success, error = builder.build()

        # Using config object
        config = BridgeConfig(enable_depth_camera=False)
        builder = ROSBridgeBuilder(robot, scene, config=config)
        success, error = builder.build()
    """

    def __init__(
        self,
        robot,
        scene,
        cameras: Optional[List[Dict[str, str]]] = None,
        enable_joint_state: bool = True,
        enable_tf: bool = True,
        enable_rgb_camera: bool = True,
        enable_depth_camera: bool = True,
        enable_robot_description: bool = False,
        urdf_path: Optional[str] = None,
        graph_path: str = "/ActionGraph_ROS",
        config: Optional[BridgeConfig] = None,
    ):
        """
        Initialize ROS bridge builder.

        Args:
            robot: Robot articulation object (from env.robot)
            scene: Scene object containing camera sensors (from env.scene)
            cameras: List of camera specifications with keys: name, scene_key, frame_id, rgb_topic, depth_topic
            enable_joint_state: Enable /joint_states topic
            enable_tf: Enable /tf topic
            enable_rgb_camera: Enable RGB camera topic (legacy, used if cameras not provided)
            enable_depth_camera: Enable depth camera topic (legacy, used if cameras not provided)
            enable_robot_description: Enable /robot_description topic (URDF)
            urdf_path: Path to URDF file (required if enable_robot_description=True)
            graph_path: OmniGraph path to create
            config: Optional BridgeConfig object (overrides individual parameters)

        Raises:
            ValueError: If robot or camera prims are invalid, or if URDF path is missing/invalid
        """
        # Default to hand camera only if cameras not specified
        if cameras is None:
            cameras = [{
                "name": "hand",
                "scene_key": "robot_camera_hand",
                "frame_id": "hand_camera",
                "rgb_topic": "/spot/camera/hand/image",
                "depth_topic": "/spot/depth_registered/hand/image"
            }]

        # Configuration
        if config is not None:
            self.config = config
        else:
            self.config = BridgeConfig(
                cameras=cameras,
                enable_joint_state=enable_joint_state,
                enable_tf=enable_tf,
                enable_rgb_camera=enable_rgb_camera,
                enable_depth_camera=enable_depth_camera,
                enable_robot_description=enable_robot_description,
                urdf_path=urdf_path,
                graph_path=graph_path,
            )

        # Read URDF content if enabled
        self.urdf_content = None
        if self.config.enable_robot_description:
            if self.config.urdf_path is None:
                raise ValueError("urdf_path must be provided when enable_robot_description=True")
            try:
                with open(self.config.urdf_path, 'r') as f:
                    self.urdf_content = f.read()
                print(f"[BRIDGE] ✓ Loaded URDF from {self.config.urdf_path} ({len(self.urdf_content)} bytes)")
            except FileNotFoundError:
                raise ValueError(f"URDF file not found: {self.config.urdf_path}")

        # Extract and validate paths
        self.robot_prim_path = self._extract_robot_path(robot)
        self.camera_paths = self._extract_all_camera_paths(scene)

        # Legacy single camera paths (for backward compatibility)
        if "robot_camera_hand" in self.camera_paths:
            self.camera_prim_path, self.render_product_path = self.camera_paths["robot_camera_hand"]
        else:
            self.camera_prim_path = None
            self.render_product_path = None

        # Validate prims exist
        self._validate_prims()

        # Node storage (populated during build)
        self.nodes: Dict[str, Any] = {}
        self.graph = None

        # Build state
        self._is_built = False

        self.success, self.error = self.build()

        if self.success:
            self._is_built = True


    def build(self) -> Tuple[bool, Optional[str]]:
        """
        Build the ROS bridge action graph with configured topics.

        Uses single og.Controller.edit() call for efficiency.

        Returns:
            (success: bool, error_msg: Optional[str])
        """
        if self._is_built:
            return False, "Graph already built. Create a new instance to rebuild."

        print(f"[BRIDGE] Creating ROS 2 Action Graph:")
        print(f"  Robot:  {self.robot_prim_path}")
        if self.config.cameras:
            print(f"  Cameras: {len(self.config.cameras)} camera(s)")
            for cam_spec in self.config.cameras:
                print(f"    - {cam_spec['name']}: {cam_spec['rgb_topic']}, {cam_spec['depth_topic']}")
        else:
            print(f"  Camera: {self.camera_prim_path if self.camera_prim_path else 'None'}")
            print(f"  Render: {self.render_product_path if self.render_product_path else 'None'}")
        print(
            f"  Topics: JointState={self.config.enable_joint_state}, "
            f"TF={self.config.enable_tf}, "
            f"RobotDescription={self.config.enable_robot_description}"
        )

        try:
            # Step 1: Ensure graph container
            success, error = self._ensure_graph_container()
            if not success:
                return False, error

            print("[BRIDGE] ✓ Graph container creation is safe")

            # Step 2: Collect all node specs from enabled topics
            all_create_nodes = []
            all_connect = []
            all_set_values = []

            # Common nodes (always required)
            create, connect, values = self._get_common_nodes_spec()
            all_create_nodes.extend(create)
            all_connect.extend(connect)
            all_set_values.extend(values)

            # Joint state topic
            if self.config.enable_joint_state:
                create, connect, values = self._get_joint_state_spec()
                all_create_nodes.extend(create)
                all_connect.extend(connect)
                all_set_values.extend(values)

            # TF topic
            if self.config.enable_tf:
                create, connect, values = self._get_tf_spec()
                all_create_nodes.extend(create)
                all_connect.extend(connect)
                all_set_values.extend(values)

            # Camera topics (RGB + Depth for all cameras)
            if self.config.cameras:
                for cam_spec in self.config.cameras:
                    scene_key = cam_spec["scene_key"]
                    if scene_key in self.camera_paths:
                        prim_path, render_path = self.camera_paths[scene_key]
                        create, connect, values = self._get_camera_specs(
                            camera_name=cam_spec["name"],
                            prim_path=prim_path,
                            render_path=render_path,
                            frame_id=cam_spec["frame_id"],
                            rgb_topic=cam_spec["rgb_topic"],
                            depth_topic=cam_spec["depth_topic"]
                        )
                        all_create_nodes.extend(create)
                        all_connect.extend(connect)
                        all_set_values.extend(values)
                    else:
                        print(f"[WARN] Camera '{cam_spec['name']}' not found in scene, skipping")
            # Legacy single camera support (deprecated)
            elif self.config.enable_rgb_camera or self.config.enable_depth_camera:
                if self.config.enable_rgb_camera:
                    create, connect, values = self._get_rgb_camera_spec()
                    all_create_nodes.extend(create)
                    all_connect.extend(connect)
                    all_set_values.extend(values)

                if self.config.enable_depth_camera:
                    create, connect, values = self._get_depth_camera_spec()
                    all_create_nodes.extend(create)
                    all_connect.extend(connect)
                    all_set_values.extend(values)

            # Robot description topic
            if self.config.enable_robot_description:
                create, connect, values = self._get_robot_description_spec()
                all_create_nodes.extend(create)
                all_connect.extend(connect)
                all_set_values.extend(values)

            print(f"[BRIDGE] Collected specs for {len(all_create_nodes)} nodes")

            # Step 3: Single og.Controller.edit() call with all accumulated specs
            keys = og.Controller.Keys
            (_, nodes, _, _) = og.Controller.edit(
                {"graph_path": self.config.graph_path},
                {
                    keys.CREATE_NODES: all_create_nodes,
                    keys.CONNECT: all_connect,
                    keys.SET_VALUES: all_set_values,
                },
            )

            print("[BRIDGE] ✓ All nodes created in single operation")

            # Step 4: Store node references for debugging
            self._populate_node_references(nodes, all_create_nodes)

            self._is_built = True
            print("[BRIDGE] ✓ ROS 2 Action Graph created successfully!")
            return True, None

        except Exception as e:
            error_msg = f"Failed to create graph: {e}"
            print(f"[ERROR] {error_msg}")
            return False, error_msg

    # Private extraction methods
    def _extract_robot_path(self, robot) -> str:
        """Extract robot prim path from robot object."""
        try:
            return robot.root_physx_view.prim_paths[0]
        except Exception as e:
            raise ValueError(f"Failed to extract robot prim path: {e}")

    def _extract_camera_paths(self, scene) -> Tuple[str, str]:
        """Extract camera prim and render product paths from scene (legacy single camera)."""
        try:
            camera = scene["robot_camera_hand"]
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Failed to find hand_camera in scene: {e}")

        try:
            camera_prim_path = camera._view.prim_paths[0]
            render_product_path = camera.render_product_paths[0]
            return camera_prim_path, render_product_path
        except Exception as e:
            raise ValueError(f"Failed to extract camera paths: {e}")

    def _extract_all_camera_paths(self, scene) -> Dict[str, Tuple[str, str]]:
        """Extract camera prim and render product paths for all cameras.

        Returns:
            Dict mapping camera scene keys to (prim_path, render_product_path) tuples
        """
        camera_paths = {}

        if self.config.cameras:
            for cam_spec in self.config.cameras:
                scene_key = cam_spec["scene_key"]
                try:
                    camera = scene[scene_key]
                    camera_prim_path = camera._view.prim_paths[0]
                    render_product_path = camera.render_product_paths[0]
                    camera_paths[scene_key] = (camera_prim_path, render_product_path)
                    print(f"[BRIDGE] ✓ Found camera '{cam_spec['name']}': {camera_prim_path}")
                except (KeyError, AttributeError) as e:
                    print(f"[WARN] Failed to find camera '{scene_key}': {e}")

        return camera_paths

    def _validate_prims(self):
        """Validate that robot and camera prims exist in USD stage."""
        import omni.usd

        stage = omni.usd.get_context().get_stage()

        robot_prim = stage.GetPrimAtPath(self.robot_prim_path)
        if not robot_prim.IsValid():
            raise ValueError(f"Robot prim invalid at: {self.robot_prim_path}")

        # Validate all camera prims
        for scene_key, (camera_prim_path, _) in self.camera_paths.items():
            camera_prim = stage.GetPrimAtPath(camera_prim_path)
            if not camera_prim.IsValid():
                print(f"[WARN] Camera prim invalid at: {camera_prim_path} (scene_key: {scene_key})")
                # Don't raise error, just warn - some cameras might be optional

    # Graph creation methods
    def _ensure_graph_container(self) -> Tuple[bool, Optional[str]]:
        """Create the base omnigraph container."""
        try:
            import omni.usd

            # Check if graph already exists and delete it
            stage = omni.usd.get_context().get_stage()
            existing_graph = stage.GetPrimAtPath(self.config.graph_path)
            if existing_graph.IsValid():
                print(f"[BRIDGE] Removing existing graph at {self.config.graph_path}")
                stage.RemovePrim(self.config.graph_path)

            return True, None
        except Exception as e:
            return False, f"Failed to ensure graph container: {e}"

    def _populate_node_references(self, nodes: List, create_specs: List):
        """
        Map returned nodes to their names for debugging.

        Args:
            nodes: List of created nodes from og.Controller.edit()
            create_specs: List of (name, type) tuples used in CREATE_NODES
        """
        for i, (node_name, _) in enumerate(create_specs):
            if i < len(nodes):
                self.nodes[node_name] = nodes[i]

    # Node specification methods (return tuples for accumulation)
    def _get_common_nodes_spec(self) -> Tuple[List, List, List]:
        """
        Get specs for OnPlaybackTick and ReadSimTime nodes.

        Returns:
            (create_nodes, connect, set_values)
        """
        create_nodes = [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ]

        connect = []  # Common nodes don't connect to each other

        set_values = []  # No configuration needed

        return create_nodes, connect, set_values

    def _get_joint_state_spec(self) -> Tuple[List, List, List]:
        """
        Get specs for /joint_states publisher.

        Returns:
            (create_nodes, connect, set_values)
        """
        create_nodes = [
            ("PubJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
        ]

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
        """
        Get specs for /tf publisher.

        Returns:
            (create_nodes, connect, set_values)
        """
        create_nodes = [
            ("PubTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
        ]

        connect = [
            ("OnPlaybackTick.outputs:tick", "PubTF.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", "PubTF.inputs:timeStamp"),
        ]

        # Collect all camera prims for TF tree
        target_prims = [usdrt.Sdf.Path(self.robot_prim_path)]
        for scene_key, (camera_prim_path, _) in self.camera_paths.items():
            target_prims.append(usdrt.Sdf.Path(camera_prim_path))

        set_values = [
            ("PubTF.inputs:targetPrims", target_prims),
            ("PubTF.inputs:topicName", self.config.tf_topic),
        ]

        return create_nodes, connect, set_values

    def _get_camera_specs(self, camera_name: str, prim_path: str, render_path: str,
                          frame_id: str, rgb_topic: str, depth_topic: str) -> Tuple[List, List, List]:
        """Get specs for a single camera's RGB and depth publishers.

        Args:
            camera_name: Identifier for the camera (e.g., "hand", "frontleft")
            prim_path: USD path to camera prim
            render_path: Render product path
            frame_id: ROS frame ID
            rgb_topic: ROS RGB topic name
            depth_topic: ROS depth topic name

        Returns:
            (create_nodes, connect, set_values) tuples
        """
        # Generate unique node names
        rgb_node = f"PubCamera_{camera_name}_rgb"
        depth_node = f"PubCamera_{camera_name}_depth"

        create_nodes = [
            (rgb_node, "isaacsim.ros2.bridge.ROS2CameraHelper"),
            (depth_node, "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ]

        connect = [
            ("OnPlaybackTick.outputs:tick", f"{rgb_node}.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", f"{depth_node}.inputs:execIn"),
        ]

        set_values = [
            # RGB camera
            (f"{rgb_node}.inputs:frameId", frame_id),
            (f"{rgb_node}.inputs:topicName", rgb_topic),
            (f"{rgb_node}.inputs:renderProductPath", render_path),
            (f"{rgb_node}.inputs:type", "rgb"),
            # Depth camera
            (f"{depth_node}.inputs:frameId", frame_id),
            (f"{depth_node}.inputs:topicName", depth_topic),
            (f"{depth_node}.inputs:renderProductPath", render_path),
            (f"{depth_node}.inputs:type", "depth"),
        ]

        return create_nodes, connect, set_values

    def _get_rgb_camera_spec(self) -> Tuple[List, List, List]:
        """
        Get specs for RGB camera publisher.

        Returns:
            (create_nodes, connect, set_values)
        """
        create_nodes = [
            ("PubCamera", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ]

        connect = [
            ("OnPlaybackTick.outputs:tick", "PubCamera.inputs:execIn"),
        ]

        set_values = [
            ("PubCamera.inputs:frameId", self.config.camera_frame_id),
            ("PubCamera.inputs:topicName", self.config.rgb_topic),
            ("PubCamera.inputs:renderProductPath", self.render_product_path),
            ("PubCamera.inputs:type", "rgb"),
        ]

        return create_nodes, connect, set_values

    def _get_depth_camera_spec(self) -> Tuple[List, List, List]:
        """
        Get specs for depth camera publisher.

        Returns:
            (create_nodes, connect, set_values)
        """
        create_nodes = [
            ("PubDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ]

        connect = [
            ("OnPlaybackTick.outputs:tick", "PubDepth.inputs:execIn"),
        ]

        set_values = [
            ("PubDepth.inputs:frameId", self.config.camera_frame_id),
            ("PubDepth.inputs:topicName", self.config.depth_topic),
            ("PubDepth.inputs:renderProductPath", self.render_product_path),
            ("PubDepth.inputs:type", "depth"),
        ]

        return create_nodes, connect, set_values

    def _get_robot_description_spec(self) -> Tuple[List, List, List]:
        """
        Get specs for /robot_description publisher (std_msgs/String).

        Publishes URDF content to ROS2 topic for robot_state_publisher integration.

        Returns:
            (create_nodes, connect, set_values)
        """
        create_nodes = [
            ("PubRobotDescription", "isaacsim.ros2.bridge.ROS2Publisher"),
        ]

        # Connect to OnPlaybackTick for publication every frame
        # Note: For efficiency, consider using OnImpulseEvent for one-time publication
        connect = [
            ("OnPlaybackTick.outputs:tick", "PubRobotDescription.inputs:execIn"),
        ]

        set_values = [
            # Configure message type (order matters: type first, then data)
            ("PubRobotDescription.inputs:messagePackage", "std_msgs"),
            ("PubRobotDescription.inputs:messageName", "String"),
            ("PubRobotDescription.inputs:messageSubfolder", "msg"),
            ("PubRobotDescription.inputs:topicName", self.config.robot_description_topic),
            # Set URDF content as message data
            ("PubRobotDescription.inputs:data", self.urdf_content),
        ]

        # Add QoS profile if specified (for latching behavior)
        if self.config.robot_description_qos_profile:
            set_values.append(
                ("PubRobotDescription.inputs:qosProfile", self.config.robot_description_qos_profile)
            )

        return create_nodes, connect, set_values
