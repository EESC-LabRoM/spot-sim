"""
Robot configuration for spot_isaacsim.

Robot URDF path, prim path, initial pose, and joint positions.
Physics parameters (drive gains, friction) are in spot_config/physics/.
Scene objects (cubes, assets, lights) are configured in scene/scene_cfg.yaml.
Simulation config (physics, gravity) is in spot_config/sim_config.py.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from scripts.spot_isaacsim.spot_config.cfg.constants import (
    SPOT_STANDING_ARM_JOINT_POSITION,
    SPOT_RESTING_ARM_JOINT_POSITION,
    SPOT_STANDING_JOINT_POSITIONS,
    SPOT_DEFAULT_JOINT_POSITIONS,
)

# Project root for resolving relative paths (3 levels up from spot_config/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

@dataclass
class RobotConfig:
    """Robot configuration for URDF-based loading."""

    urdf_path: str = "external/relic/source/relic/relic/assets/spot/spot_with_arm.urdf"
    lula_config_path: str = "scripts/spot_isaacsim/spot_config/cfg/spot_lula_config.yaml"
    prim_path: str = "/World/Robot"
    position: Tuple[float, float, float] = (-2.5, 0.0, 0.7)
    orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz
    initial_goal_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, yaw (metres, metres, radians)
    initial_joint_positions: Dict[str, float] = field(default_factory=lambda: dict(SPOT_DEFAULT_JOINT_POSITIONS))

    # Subsystem feature flags
    enable_locomotion: bool = True
    enable_arm_ik: bool = True
    enable_ros_bridge: bool = True
    enable_ros_controllers: bool = True
    enable_keyboard_controller: bool = True
    enable_collision_avoidance: bool = True

    # Camera creation filter — None means create all cameras; list restricts to named cameras
    cameras: list = field(default_factory=lambda: ["hand", "front_left", "front_right", "rear_left", "rear", "left", "right"])

    def __post_init__(self) -> None:
        urdf = Path(self.urdf_path)
        if not urdf.is_absolute():
            urdf = PROJECT_ROOT / urdf
        if not urdf.exists():
            raise FileNotFoundError(f"Robot URDF not found: {urdf}")
        self.urdf_path = str(urdf)

        lula = Path(self.lula_config_path)
        if not lula.is_absolute():
            lula = PROJECT_ROOT / lula
        if not lula.exists():
            raise FileNotFoundError(f"Lula config not found: {lula}")
        self.lula_config_path = str(lula)

_BASE_REACH_M = 1.0  # max XY distance at which the arm can reach a target


class SpotRobot:
    """
    Encapsulates all robot subsystem setup: cameras, physics, ROS2 bridge,
    locomotion, arm IK, and grasp executor.

    Instantiate after the World and scene are built (robot articulation available).
    All Isaac Sim / local imports are deferred inside __init__ to avoid circular
    imports and to respect the SimulationApp initialization requirement.
    """

    def __init__(self, robot, world,
                 scene_cfg: "SceneConfig",
                 bridge_cfg: "BridgeConfig | None" = None,
                 zed_cfg: "ZedConfig | None" = None,
                 device: str = "cpu",
                 headless: bool = False):
        # Deferred imports — must happen after SimulationApp + extensions are ready,
        # and keeps this module free of circular dependencies with control/.
        from scripts.spot_isaacsim.spot_config import all_rgb_camera_configs
        from scripts.spot_isaacsim.scene import create_all_cameras, initialize_cameras
        from scripts.spot_isaacsim.spot_config.physics import apply_all_physics
        from scripts.spot_isaacsim.omnigraph import ROSBridgeBuilder
        from scripts.spot_isaacsim.control.spot_controller import SpotController

        self._world = world
        self._scene_cfg = scene_cfg
        self._apply_all_physics = apply_all_physics
        self._robot = robot

        # Cameras (created before world.reset())
        print("[INFO] Creating RGB cameras (with depth enabled)...")
        active_configs = [c for c in all_rgb_camera_configs
                          if scene_cfg.robot.cameras is None or c.name in scene_cfg.robot.cameras]
        self.cameras = create_all_cameras(scene_cfg.robot.prim_path, active_configs)
        print(f"[INFO] Created {len(self.cameras)} RGB camera(s): {list(self.cameras)}")

        # ZED camera USD asset (mounted before world.reset())
        self.zed_prim_path = None
        if zed_cfg is not None:
            from scripts.spot_isaacsim.spot_config.cameras import add_zed_to_stage
            print("[INFO] Adding ZED X camera to stage...")
            self.zed_prim_path = add_zed_to_stage(zed_cfg)
            print(f"[INFO] ZED X mounted at: {self.zed_prim_path}")

        # World reset + physics + camera init
        world.reset()
        print("[INFO] World reset complete")
        apply_all_physics(scene_cfg.robot.prim_path)
        initialize_cameras(list(self.cameras.values()), enable_depth=True)

        # ROS2 bridge
        if scene_cfg.robot.enable_ros_bridge:
            if bridge_cfg is None:
                raise ValueError("enable_ros_bridge is True but no BridgeConfig was provided")
            print("[INFO] Creating ROS2 bridge...")
            pub_cameras = {k: v for k, v in self.cameras.items() if k in bridge_cfg.publishing_cameras}
            self.ros2_bridge = ROSBridgeBuilder(
                robot=robot,
                cameras=pub_cameras,
                camera_step=bridge_cfg.camera_step,
                zed_prim_path=self.zed_prim_path,
                zed_config=zed_cfg if self.zed_prim_path else None,
            )
            if self.ros2_bridge.success:
                print("[INFO] ROS2 bridge created successfully")
            else:
                print(f"[ERROR] ROS2 bridge failed: {self.ros2_bridge.error}")
        else:
            self.ros2_bridge = None
            print("[INFO] ROS2 bridge disabled")

        # Initial joint positions
        joint_names = list(robot.dof_names)
        positions = robot.get_joint_positions()
        applied_count = 0
        for name, pos in scene_cfg.robot.initial_joint_positions.items():
            if name in joint_names:
                positions[joint_names.index(name)] = pos
                applied_count += 1
            else:
                print(f"[WARN] Joint '{name}' not found in robot")
        robot.set_joint_positions(positions)
        print(f"[INFO] Applied initial positions to {applied_count}/{robot.num_dof} joints")

        # All loco-manip controllers (locomotion, arm IK, ROS executors, state machine)
        self.controller = SpotController(robot, world, scene_cfg, cameras=self.cameras, device=device)

        # Keyboard controller
        if scene_cfg.robot.enable_keyboard_controller:
            from scripts.spot_isaacsim.control.interfaces.keyboard_controller import create_keyboard_controller
            self.kb = create_keyboard_controller(headless=headless)
            self.kb.subscribe(self.controller.locomotion.on_key_event)
            if self.controller.arm_controller is not None:
                self.kb.subscribe(self.controller.arm_controller.on_key_event)
        else:
            self.kb = None
            print("[INFO] Keyboard controller disabled")

    def update(self) -> None:
        """Step all robot controllers."""
        if self.kb is not None and hasattr(self.kb, 'update'):
            self.kb.update()
        self.controller.update()

    def reinitialize(self) -> None:
        """Re-run physics and controller init after a GUI stop → Play cycle."""
        self._world.reset()
        self.controller.reinitialize()

    def set_base_pose(self, pose_2d: Tuple[float, float, float]) -> None:
        """Navigate base to (x, y, yaw) and home arm."""
        self.controller.set_base_pose(*pose_2d)

    def set_ee_pose(self, target_position: np.ndarray,
                    target_orientation: np.ndarray | None,
                    reach: float = _BASE_REACH_M) -> None:
        """Track target EE pose. Navigates base to standoff first if needed."""
        self.controller.set_ee_pose(target_position, target_orientation, reach)

    def set_grasp_pose(self, target_position: np.ndarray,
                       target_orientation: np.ndarray,
                       do_pick: bool = False) -> None:
        """Approach pre-grasp → advance to grasp → close gripper. do_pick also retrieves."""
        self.controller.set_grasp_pose(target_position, target_orientation, do_pick)



def load_robot_from_urdf(
    world,
    urdf_path: str,
    prim_path: str,
    position: Tuple[float, float, float],
    orientation: Tuple[float, float, float, float],
):
    """
    Load robot from URDF at runtime using the Isaac Sim URDF importer.

    Imports the URDF directly into the live stage (no intermediate USD file),
    moves the prim to the requested prim_path, and wraps it as a SingleArticulation.

    Args:
        world: Isaac Sim World object
        urdf_path: Absolute path to URDF file
        prim_path: Desired USD prim path for the robot (e.g. '/World/Robot')
        position: Initial position (x, y, z)
        orientation: Initial orientation quaternion (w, x, y, z)

    Returns:
        SingleArticulation object for the robot
    """
    import omni.kit.app
    import omni.kit.commands
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.extensions import enable_extension

    print(f"[Robot] Importing from URDF: {urdf_path}")

    # Enable URDF importer extension
    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled("isaacsim.asset.importer.urdf"):
        enable_extension("isaacsim.asset.importer.urdf")

    # Create import configuration (same settings as convert_urdf_to_usd.py)
    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.set_distance_scale(1.0)
    import_config.set_make_default_prim(True)
    import_config.set_create_physics_scene(True)
    import_config.set_density(0.0)
    import_config.set_convex_decomp(False)
    import_config.set_collision_from_visuals(False)
    import_config.set_merge_fixed_joints(False)
    import_config.set_fix_base(False)
    import_config.set_self_collision(False)
    import_config.set_parse_mimic(False)
    import_config.set_replace_cylinders_with_capsules(False)

    # Import into live stage (dest_path="" = open stage, no file written).
    # Temporarily change CWD so the importer writes materials/textures under
    # spot_config/cfg/ instead of the project root.
    cfg_dir = Path(__file__).parent / "cfg"
    cfg_dir.mkdir(exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(str(cfg_dir))
    try:
        status, imported_prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(urdf_path),
            import_config=import_config,
            dest_path="",
        )
    finally:
        os.chdir(original_cwd)

    if not status or not imported_prim_path:
        raise RuntimeError(f"[Robot] URDF import failed for: {urdf_path}")

    # The importer stores material texture references as relative paths (e.g.
    # "./materials/textures/foo.png").  For an in-memory stage these are resolved
    # against CWD at render time, not against cfg_dir.  Patch them to absolute
    # paths so Hydra can find the textures wherever CWD happens to be.
    import omni.usd
    from pxr import Sdf
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() == Sdf.ValueTypeNames.Asset:
                val = attr.Get()
                if val and str(val.path).startswith("./materials/textures/"):
                    filename = Path(str(val.path)).name
                    abs_path = str(cfg_dir / "materials" / "textures" / filename)
                    attr.Set(Sdf.AssetPath(abs_path))
    print(f"[Robot] Patched material texture paths → {cfg_dir / 'materials' / 'textures'}")

    print(f"[Robot] URDF imported at: {imported_prim_path}")

    # Move prim to desired location if it differs from import location
    if imported_prim_path != prim_path:
        omni.kit.commands.execute(
            "MovePrimCommand",
            path_from=imported_prim_path,
            path_to=prim_path,
            keep_world_transform=True,
        )
        print(f"[Robot] Moved prim: {imported_prim_path} -> {prim_path}")

    articulation = SingleArticulation(
        prim_path=prim_path,
        name="spot_robot",
        position=np.array(position),
        orientation=np.array(orientation),
    )

    world.scene.add(articulation)
    print(f"[Robot] Added at {prim_path}")
    return articulation


def load_robot_from_usd(
    world,
    usd_path: str,
    prim_path: str,
    position: Tuple[float, float, float],
    orientation: Tuple[float, float, float, float],
):
    """
    Load robot from pre-converted USD file (backward compatibility).

    Args:
        world: Isaac Sim World object
        usd_path: Absolute path to USD file
        prim_path: USD prim path for the robot
        position: Initial position (x, y, z)
        orientation: Initial orientation quaternion (w, x, y, z)

    Returns:
        SingleArticulation object for the robot
    """
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage

    print(f"[Robot] Loading from USD: {usd_path}")

    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

    articulation = SingleArticulation(
        prim_path=prim_path,
        name="spot_robot",
        position=np.array(position),
        orientation=np.array(orientation),
    )

    world.scene.add(articulation)
    print(f"[Robot] Added at {prim_path}")
    return articulation
