"""
SceneBuilder: data-driven scene construction from scene_cfg.yaml.

Usage:
    scene = SceneBuilder(world, config.robot)
    robot = scene.objects["robot"]

Custom YAML:
    scene = SceneBuilder(world, config.robot, yaml_path="path/to/cfg.yaml")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from scripts.spot_isaacsim.spot_config.robot import RobotConfig

DEFAULT_CFG_PATH = Path(__file__).parent / "scene_cfg.yaml"


@dataclass
class SceneConfig:
    """Simulation configuration: robot + physics settings."""

    robot: RobotConfig = field(default_factory=RobotConfig)
    physics_dt: float = 1.0 / 500.0  # locomotion requires 500 Hz
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)


class SceneBuilder:
    """Builds an Isaac Sim scene from a YAML configuration file.

    Constructs and populates the scene immediately. Results are in ``self.objects``,
    a dict mapping object names to Isaac Sim objects (always contains "robot").
    """

    def __init__(self, world, robot_cfg: RobotConfig, yaml_path: Optional[str | Path] = None):
        import yaml
        import omni.usd

        cfg_path = Path(yaml_path) if yaml_path else DEFAULT_CFG_PATH
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        print(f"[SceneBuilder] Loaded config from: {cfg_path}")

        robot_yaml = cfg.get("robot", {})
        if "spawn_position" in robot_yaml:
            robot_cfg.position = tuple(robot_yaml["spawn_position"])
        if "initial_goal_pose" in robot_yaml:
            robot_cfg.initial_goal_pose = tuple(robot_yaml["initial_goal_pose"])

        stage = omni.usd.get_context().get_stage()
        self.objects: Dict[str, Any] = {}

        stage_loaded = _build_stage(world, cfg, self.objects)
        if not stage_loaded:
            _build_ground_plane(world, cfg)
        _build_dome_light(stage, cfg)
        self.objects["robot"] = _build_robot(world, robot_cfg)

        for obj in cfg.get("objects", []):
            if not obj.get("enabled", True):
                continue
            name = obj["name"]
            obj_type = obj.get("type")
            if obj_type == "static_cube":
                self.objects[name] = _build_static_cube(world, obj)
            elif obj_type == "dynamic_cube":
                self.objects[name] = _build_dynamic_cube(world, obj)
            else:
                print(f"[SceneBuilder] Unknown object type '{obj_type}' for '{name}', skipping.")

        for asset in cfg.get("assets", []):
            if not asset.get("enabled", False):
                continue
            self.objects[asset["name"]] = _build_usd_asset(world, asset)

        print(f"[SceneBuilder] Scene built with {len(self.objects)} objects: {list(self.objects.keys())}")


# ---------------------------------------------------------------------------
# Private helpers — unpack YAML dicts and delegate to loaders
# ---------------------------------------------------------------------------

def _build_stage(world, cfg: dict, objects: dict) -> bool:
    stage_cfg = cfg.get("stage", {})
    if not stage_cfg.get("enabled", False):
        return False
    usd_path = stage_cfg.get("usd_path")
    stage_type = stage_cfg.get("type")
    prim_path = stage_cfg.get("prim_path", "/World/Warehouse")
    position = tuple(stage_cfg.get("position", [0.0, 0.0, 0.0]))
    scale = tuple(stage_cfg.get("scale", [1.0, 1.0, 1.0]))
    if usd_path and not stage_type:
        from .loaders import load_scene_usd
        objects["stage"] = load_scene_usd(world, prim_path, usd_path, position=position, scale=scale)
    else:
        from .loaders import load_warehouse_stage
        objects["stage"] = load_warehouse_stage(
            world, prim_path=prim_path, stage_type=stage_type,
            usd_path=usd_path, position=position, scale=scale,
        )
    return True


def _build_ground_plane(world, cfg: dict) -> None:
    ground_cfg = cfg.get("ground", {})
    from .loaders import create_ground_plane
    create_ground_plane(
        world,
        prim_path=ground_cfg.get("prim_path", "/World/GroundPlane"),
        size=ground_cfg.get("size", 10.0),
        z_position=ground_cfg.get("z_position", 0.0),
        color=tuple(ground_cfg.get("color", [0.5, 0.5, 0.5])),
    )


def _build_dome_light(stage, cfg: dict) -> None:
    light_cfg = cfg.get("light", {})
    from .loaders import create_dome_light
    create_dome_light(
        stage,
        prim_path=light_cfg.get("prim_path", "/World/DomeLight"),
        intensity=light_cfg.get("intensity", 4000.0),
        color=tuple(light_cfg.get("color", [0.9, 0.9, 0.9])),
    )


def _build_robot(world, robot_cfg: RobotConfig):
    from scripts.spot_isaacsim.spot_config.robot import load_robot_from_urdf
    return load_robot_from_urdf(
        world,
        urdf_path=robot_cfg.urdf_path,
        prim_path=robot_cfg.prim_path,
        position=robot_cfg.position,
        orientation=robot_cfg.orientation,
    )


def _build_static_cube(world, obj: dict):
    from .loaders import create_static_cube
    return create_static_cube(
        world,
        prim_path=obj["prim_path"],
        size=tuple(obj["scale"]),
        position=tuple(obj["position"]),
        color=tuple(obj["color"]),
        orientation_rpy=tuple(obj.get("orientation_rpy", [0.0, 0.0, 0.0])),
        static_friction=obj.get("static_friction", 0.7),
        dynamic_friction=obj.get("dynamic_friction", 0.7),
    )


def _build_dynamic_cube(world, obj: dict):
    from .loaders import create_dynamic_cube
    return create_dynamic_cube(
        world,
        prim_path=obj["prim_path"],
        size=tuple(obj["scale"]),
        position=tuple(obj["position"]),
        color=tuple(obj["color"]),
        mass=obj.get("mass", 0.1),
        orientation_rpy=tuple(obj.get("orientation_rpy", [0.0, 0.0, 0.0])),
    )


def _build_usd_asset(world, asset: dict):
    from .loaders import load_usd_asset
    return load_usd_asset(
        world,
        prim_path=asset["prim_path"],
        usd_path=asset["usd_path"],
        position=tuple(asset.get("position", [0.0, 0.0, 0.0])),
        orientation_rpy=tuple(asset.get("orientation_rpy", [0.0, 0.0, 0.0])),
        scale=tuple(asset.get("scale", [1.0, 1.0, 1.0])),
        physics=asset.get("physics"),
    )
