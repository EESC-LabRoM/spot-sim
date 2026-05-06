import argparse
import math
from pathlib import Path

import numpy as np
import yaml

from scripts.spot_isaacsim.tasks.base_task import BaseTask

_TASK_DIR = Path(__file__).parent
_RANDOMIZE_INTERVAL = 40   # sim steps between randomizations
_ROBOT_RADIUS       = 1.4  # m — circle radius around table center


class Task(BaseTask):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.scene_cfg_path = str(_TASK_DIR / "scene.yaml")
        self.robot_cfg.enable_locomotion = False
        self.robot_cfg.enable_keyboard_controller = False
        self.robot_cfg.enable_ros_controllers = False
        self.robot_cfg.enable_collision_avoidance = False
        self.robot_cfg.cameras = []
        self.bridge_cfg.publishing_cameras = []

        with open(self.scene_cfg_path) as f:
            cfg = yaml.safe_load(f)

        table = next(o for o in cfg["objects"] if o["name"] == "table")
        t_pos   = table["position"]
        t_scale = table["scale"]

        self._table_center = np.array([t_pos[0], t_pos[1]])
        self._table_half_x = t_scale[0] / 2.0
        self._table_half_y = t_scale[1] / 2.0

        self._asset_infos = [
            {
                "name": a["name"],
                "z":    a["position"][2],
                "rpy":  a.get("orientation_rpy", [0.0, 0.0, 0.0]),
            }
            for a in cfg.get("assets", [])
            if a.get("enabled", False)
        ]

    def build(self, world):
        from scripts.spot_isaacsim.scene import SceneBuilder
        from scripts.spot_isaacsim.spot_config.robot import SpotRobot
        import omni.usd
        from pxr import Usd, UsdPhysics, PhysxSchema

        args = self.args
        self.bridge_cfg.camera_step = max(
            1, round(1.0 / (self.scene_cfg.physics_dt * args.render_interval * self.bridge_cfg.camera_hz))
        )
        self.gate_step = self.bridge_cfg.camera_step

        scene = SceneBuilder(world, self.scene_cfg.robot, yaml_path=self.scene_cfg_path)
        self._scene_objects = scene.objects

        # Disable gravity before world.reset() — SpotRobot.__init__ calls world.reset(),
        # and modifying PhysxRigidBodyAPI after reset invalidates the simulation view.
        stage = omni.usd.get_context().get_stage()
        for prim in Usd.PrimRange(stage.GetPrimAtPath(self.scene_cfg.robot.prim_path)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                PhysxSchema.PhysxRigidBodyAPI.Get(stage, prim.GetPath()).CreateDisableGravityAttr().Set(True)
                print("[Task] Gravity disabled on robot root body")
                break

        self.spot = SpotRobot(
            scene.objects["robot"], world,
            scene_cfg=self.scene_cfg,
            bridge_cfg=self.bridge_cfg,
            zed_cfg=self.zed_cfg,
            device=args.device,
            headless=args.headless,
        )
        return self.spot

    def on_setup(self, _world, robot) -> None:
        from scripts.spot_isaacsim.scene.loaders.utils import rpy_deg_to_quat
        self._rpy_to_quat = rpy_deg_to_quat
        self._asset_prims = {ai["name"]: self._scene_objects[ai["name"]] for ai in self._asset_infos}
        self._loco = robot.controller.locomotion

    def update(self, world, robot, step: int) -> None:
        if step % _RANDOMIZE_INTERVAL == 0:
            self._randomize()
        robot.update()

    def _randomize(self) -> None:
        theta = np.random.uniform(0.0, 2.0 * math.pi)
        cx, cy = self._table_center
        self._loco.set_target_pose(
            cx + _ROBOT_RADIUS * math.cos(theta),
            cy + _ROBOT_RADIUS * math.sin(theta),
            theta + math.pi,
        )

        for ai in self._asset_infos:
            x = np.random.uniform(cx - self._table_half_x, cx + self._table_half_x)
            y = np.random.uniform(cy - self._table_half_y, cy + self._table_half_y)
            yaw_deg = np.random.uniform(0.0, 360.0)
            quat = self._rpy_to_quat([ai["rpy"][0], ai["rpy"][1], yaw_deg])
            prim = self._asset_prims[ai["name"]]
            prim.set_world_pose(position=np.array([x, y, ai["z"]]), orientation=quat)
            if hasattr(prim, "set_linear_velocity"):
                prim.set_linear_velocity(np.zeros(3))
                prim.set_angular_velocity(np.zeros(3))
