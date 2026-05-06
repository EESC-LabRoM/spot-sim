"""
DefaultTask — replicates the original play.py behavior.

All subsystems enabled. Scene loaded from tasks/default/scene.yaml.
"""

import argparse
from pathlib import Path

from scripts.spot_isaacsim.tasks.base_task import BaseTask

_TASK_DIR = Path(__file__).parent


class Task(BaseTask):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.scene_cfg_path = str(_TASK_DIR / "scene.yaml")
        # All RobotConfig flags stay at their defaults (all True)

    def update(self, world, robot, step: int) -> None:
        robot.update()
