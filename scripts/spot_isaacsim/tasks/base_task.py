"""
BaseTask — base class for all Isaac Sim scenarios.

Subclasses set config attributes in __init__ and implement update().
play.py creates the World, then calls task.build() to create the scene and
robot, then drives the lifecycle hooks (on_setup, update, on_reinitialize,
on_shutdown).
"""

import argparse
import inspect
from abc import ABC, abstractmethod
from pathlib import Path

from scripts.spot_isaacsim.spot_config.robot import RobotConfig
from scripts.spot_isaacsim.scene.builder import SceneConfig
from scripts.spot_isaacsim.omnigraph.config import BridgeConfig
from scripts.spot_isaacsim.spot_config.cameras.mounted_zed import ZedConfig


class BaseTask(ABC):
    def __init__(self, args: argparse.Namespace):
        self.args = args

        # Scene config path — resolved by SceneBuilder; None = SceneBuilder uses its default
        self.scene_cfg_path: str | None = None

        # Subsystem configs — override fields in subclass __init__ as needed
        self.robot_cfg: RobotConfig = RobotConfig()
        self.scene_cfg: SceneConfig = SceneConfig(robot=self.robot_cfg)
        self.bridge_cfg: BridgeConfig = BridgeConfig()
        self.zed_cfg: ZedConfig | None = ZedConfig()  # None = ZED disabled

    @property
    def task_dir(self) -> Path:
        """Directory containing the concrete task's task.py file."""
        return Path(inspect.getfile(type(self))).parent

    def build(self, world):
        """
        Create SceneBuilder and SpotRobot from this task's config.
        Called by play.py after World is created. Returns the SpotRobot instance.
        Override to swap out the build sequence.
        """
        # Deferred imports — must happen after SimulationApp + extensions are ready
        from scripts.spot_isaacsim.scene import SceneBuilder
        from scripts.spot_isaacsim.spot_config.robot import SpotRobot

        args = self.args
        self.bridge_cfg.camera_step = max(
            1, round(1.0 / (self.scene_cfg.physics_dt * args.render_interval * self.bridge_cfg.camera_hz))
        )
        self.gate_step = self.bridge_cfg.camera_step  # kept for print_sim_summary

        scene = SceneBuilder(world, self.scene_cfg.robot, yaml_path=self.scene_cfg_path)
        self._scene_objects = scene.objects
        robot = scene.objects["robot"]
        self.spot = SpotRobot(
            robot, world,
            scene_cfg=self.scene_cfg,
            bridge_cfg=self.bridge_cfg,
            zed_cfg=self.zed_cfg,
            device=args.device,
            headless=args.headless,
        )
        return self.spot

    def on_setup(self, _world, _robot) -> None:
        """Called once after Isaac Sim is ready and robot is spawned.
        Override for one-time scene setup."""

    @abstractmethod
    def update(self, world, robot, step: int) -> None:
        """Called every step of the main simulation loop."""

    def on_reinitialize(self, _world, _robot) -> None:
        """Called when the simulation is restarted (GUI stop → play).
        Override to add custom reinit logic."""

    def on_shutdown(self, _world, _robot) -> None:
        """Called on clean exit. Override for cleanup."""
