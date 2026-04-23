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
import torch

# Project root for resolving relative paths (3 levels up from spot_config/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Projected Arm Position
initial_curvature = {
    "shoulder": torch.pi / 180.0 * 110.0,  # Initial shoulder curvature
    "elbow": torch.pi / 180.0 * 45.0,  # Initial elbow curvature
}
initial_curvature["wrist"] = 180 - (
    (180 - initial_curvature["shoulder"]) + initial_curvature["elbow"]
)  # Initial wrist curvature

# Values for the 7 arm joints
SPOT_STANDING_ARM_JOINT_POSITION = {
    "arm_sh0": 0.0,
    "arm_sh1": -3.141 + initial_curvature["shoulder"],  # Shoulder 0/1
    "arm_el0": 3.141 - initial_curvature["elbow"],
    "arm_el1": 0.0,  # Elbow 0/1
    "arm_wr0": 0.0 - initial_curvature["wrist"],
    "arm_wr1": 0.0,  # Wrist 0/1
    "arm_f1x": -1.0,  # Gripper
}

SPOT_RESTING_ARM_JOINT_POSITION = {
    "arm_sh0": 0.0,
    "arm_sh1": -3.141,  # Shoulder 0/1
    "arm_el0": 3.141,
    "arm_el1": 0.0,  # Elbow 0/1
    "arm_wr0": 0.0,
    "arm_wr1": 0.0,  # Wrist 0/1
    "arm_f1x": -1.0,  # Gripper
}

# Default standing pose
# Standing spot configuration used without locomotion
SPOT_STANDING_JOINT_POSITIONS: Dict[str, float] = {
    # Front left leg
    "fl_hx":  0.1,
    "fl_hy":  0.9,
    "fl_kn": -1.503,
    # Front right leg
    "fr_hx": -0.1,
    "fr_hy":  0.9,
    "fr_kn": -1.503,
    # Hind left leg
    "hl_hx":  0.1,
    "hl_hy":  1.1,
    "hl_kn": -1.503,
    # Hind right leg
    "hr_hx": -0.1,
    "hr_hy":  1.1,
    "hr_kn": -1.503,
}

SPOT_DEFAULT_JOINT_POSITIONS = SPOT_STANDING_JOINT_POSITIONS | SPOT_RESTING_ARM_JOINT_POSITION

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

class SpotRobot:
    """
    Encapsulates all robot subsystem setup: cameras, physics, ROS2 bridge,
    locomotion, arm IK, and grasp executor.

    Instantiate after the World and scene are built (robot articulation available).
    All Isaac Sim / local imports are deferred inside __init__ to avoid circular
    imports and to respect the SimulationApp initialization requirement.
    """

    def __init__(self, robot, world, config, device: str = "cpu",
                 enabled_cameras: set = None, gate_step: int = 1,
                 zed_config="default", headless: bool = False):
        # Deferred imports — must happen after SimulationApp + extensions are ready,
        # and keeps this module free of circular dependencies with control/.
        from scripts.spot_isaacsim.spot_config import all_rgb_camera_configs
        from scripts.spot_isaacsim.scene import create_all_cameras, initialize_cameras
        from scripts.spot_isaacsim.spot_config.physics import apply_all_physics
        from scripts.spot_isaacsim.omnigraph import ROSBridgeBuilder
        from scripts.spot_isaacsim.control import (
            SpotLocomotionController,
            NavigationExecutor,
            SpotArmController,
            GraspExecutor,
        )

        self._world = world
        self._config = config
        self._apply_all_physics = apply_all_physics
        self._step_count = 0
        self._loco_counter = 0

        # Cameras (created before world.reset())
        print("[INFO] Creating RGB cameras (with depth enabled)...")
        active_configs = [c for c in all_rgb_camera_configs
                          if enabled_cameras is None or c.name in enabled_cameras]
        self.cameras = create_all_cameras(config.robot.prim_path, active_configs)
        print(f"[INFO] Created {len(self.cameras)} RGB camera(s): {list(self.cameras)}")

        # ZED camera USD asset (mounted before world.reset())
        self.zed_prim_path = None
        if zed_config is not None:
            from scripts.spot_isaacsim.spot_config.cameras import ZedConfig, add_zed_to_stage
            if zed_config == "default":
                zed_config = ZedConfig()
            print("[INFO] Adding ZED X camera to stage...")
            self.zed_prim_path = add_zed_to_stage(zed_config)
            print(f"[INFO] ZED X mounted at: {self.zed_prim_path}")

        # World reset + physics + camera init
        world.reset()
        print("[INFO] World reset complete")
        apply_all_physics(config.robot.prim_path)
        initialize_cameras(list(self.cameras.values()), enable_depth=True)

        # ROS2 bridge
        print("[INFO] Creating ROS2 bridge...")
        self.ros2_bridge = ROSBridgeBuilder(
            robot=robot,
            cameras=self.cameras,
            camera_step=gate_step,
            zed_prim_path=self.zed_prim_path,
            zed_config=zed_config if self.zed_prim_path else None,
        )
        if self.ros2_bridge.success:
            print("[INFO] ROS2 bridge created successfully")
        else:
            print(f"[ERROR] ROS2 bridge failed: {self.ros2_bridge.error}")

        # Initial joint positions
        joint_names = list(robot.dof_names)
        positions = robot.get_joint_positions()
        applied_count = 0
        for name, pos in config.robot.initial_joint_positions.items():
            if name in joint_names:
                positions[joint_names.index(name)] = pos
                applied_count += 1
            else:
                print(f"[WARN] Joint '{name}' not found in robot")
        robot.set_joint_positions(positions)
        print(f"[INFO] Applied initial positions to {applied_count}/{robot.num_dof} joints")

        # Locomotion + navigation
        self.locomotion = SpotLocomotionController(
            robot,
            physics_dt=config.physics_dt,
            device=device,
        )
        self.locomotion.initialize()
        x, y, yaw = config.robot.initial_goal_pose
        self.locomotion.set_target_pose(x, y, yaw)
        self.locomotion.set_cameras(self.cameras)
        self.navigation_executor = NavigationExecutor(self.locomotion, physics_dt=config.physics_dt)

        # Arm IK + grasp executor
        self.arm_controller = SpotArmController(robot, config.robot)
        self.arm_controller.initialize(world)
        self.executor = GraspExecutor(self.arm_controller, robot)
        self.executor.enable_ros2(self.navigation_executor)
        self.navigation_executor.enable_ros2()

        # Keyboard controller — auto-selects carb (headed) or stdin (headless)
        from scripts.spot_isaacsim.control.keyboard_controller import create_keyboard_controller
        self.kb = create_keyboard_controller(headless=headless)
        self.kb.subscribe(self.locomotion.on_key_event)
        self.kb.subscribe(self.arm_controller.on_key_event)

    _WARMUP_STEPS = 100  # physics steps before locomotion policy activates

    def update(self) -> None:
        """Step all robot controllers. Locomotion is gated until warmup is complete."""
        if hasattr(self.kb, 'update'):
            self.kb.update()
        self._step_count += 1
        if self._step_count >= self._WARMUP_STEPS:
            self._loco_counter += 1
            if self._loco_counter % self.locomotion.decimation == 0:
                self.locomotion.forward(self._config.physics_dt * self.locomotion.decimation)
        self.arm_controller.update()
        if not self.arm_controller.is_tracking:
            self.executor.update()
        self.navigation_executor.update()

    def reinitialize(self) -> None:
        """Re-run physics and locomotion init after a GUI stop → Play cycle."""
        self._world.reset()
        self._apply_all_physics(self._config.robot.prim_path)
        self.locomotion.initialize()
        x, y, yaw = self._config.robot.initial_goal_pose
        self.locomotion.set_target_pose(x, y, yaw)
        self._step_count = 0
        self._loco_counter = 0


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
    import_config.set_self_collision(True)
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
