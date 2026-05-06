"""
Agnostic Grasp Pipeline — scene and task oriented for agn_grasp testing.

"""

import argparse
from pathlib import Path

from scripts.spot_isaacsim.tasks.base_task import BaseTask

_TASK_DIR = Path(__file__).parent


class Task(BaseTask):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.scene_cfg_path = str(_TASK_DIR / "scene.yaml")
        self.robot_cfg.enable_keyboard_controller = False
        self.bridge_cfg.publishing_cameras = ["hand", "front_left", "front_right"]

    def update(self, world, robot, step: int) -> None:
        robot.update()
