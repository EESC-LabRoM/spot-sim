#!/usr/bin/env python3
"""
Spot Robot Isaac Sim Spawner

Spawns Spot with cameras, ROS2 bridge, Lula IK, locomotion policy, and collision avoidance.

ML inference nodes (segmentation, grasp) run inside the agn_grasp container:
    just ros-dl-nodes          # full pipeline
    just ros-segmentation      # segmentation only

Via justfile:
    just run scripts/spot_isaacsim/play.py
"""

import argparse
import sys
from pathlib import Path
import importlib
import os

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ZED_EXT_PATH = PROJECT_ROOT / "external" / "zed-isaac-sim" / "exts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spawn Spot robot in Isaac Sim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless mode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for camera data tensors (default: cuda:0)",
    )
    parser.add_argument(
        "--render-interval",
        type=int,
        default=5,
        help="Render every N steps for performance (default: 5)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="_default",
        help="Task name to load from tasks/<name>/task.py (default: default)",
    )
    return parser.parse_args()


def _detect_exec_mode() -> bool:
    """Check if Kit is already running (isaacsim --exec mode)."""
    try:
        import omni.kit.app

        app = omni.kit.app.get_app()
        return app is not None and app.is_running()
    except (ImportError, RuntimeError, AttributeError):
        return False


class _KitAppAdapter:
    """Wraps an already-running Kit app with SimulationApp-like interface."""

    def __init__(self):
        import omni.kit.app

        self._app = omni.kit.app.get_app()
        self._exiting = False

    def is_running(self) -> bool:
        if self._exiting:
            return False
        return self._app.is_running()

    def close(self) -> None:
        self._exiting = True
        self._app.post_quit()


def main():
    args = parse_args()

    exec_mode = _detect_exec_mode()

    # Build environment with Isaac Sim's runtime paths
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        os.pathsep.join(sys.path) + os.pathsep + env.get("PYTHONPATH", "")
    )

    # Force FastDDS Large Data mode to prevent UDP buffer overflows
    env["FASTDDS_BUILTIN_TRANSPORTS"] = "LARGE_DATA"

    if exec_mode:
        print("[INFO] Detected --exec mode (Kit already running)")
        simulation_app = _KitAppAdapter()
        import builtins

        builtins.ISAAC_LAUNCHED_FROM_TERMINAL = False
        if not hasattr(builtins, "ISAACSIM_APP_LAUNCHED"):
            builtins.ISAACSIM_APP_LAUNCHED = True
    else:
        print("[INFO] Starting Isaac Sim (standalone mode)...")
        from isaacsim import SimulationApp

        app_config = {
            "headless": args.headless,
            "width": 1920,
            "height": 1080,
            "window_title": "Spot Robot - Isaac Sim",
        }
        if args.headless:
            app_config["disable_viewport_updates"] = True
        simulation_app = SimulationApp(app_config)

    print("[INFO] SimulationApp ready")

    # Enable required extensions
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
    manager.set_extension_enabled_immediate("omni.graph.scriptnode", True)

    # ZED camera extension (for USD asset and ROS2 integration)
    if ZED_EXT_PATH.exists():
        manager.add_path(str(ZED_EXT_PATH))
        manager.set_extension_enabled_immediate("sl.sensor.camera", True)
        print(f"[INFO] ZED extension loaded from: {ZED_EXT_PATH}")
    else:
        print(f"[WARN] ZED extension not found at {ZED_EXT_PATH}")
        print(
            f"[WARN] ZED camera will not be available. To enable, clone the zed-isaac-sim repo and ensure the ext path exists: {ZED_EXT_PATH}"
        )

    # Isaac Sim modules — must import after SimulationApp and extensions are ready
    from isaacsim.core.api import World

    # Load task
    task_module = importlib.import_module(
        f"scripts.spot_isaacsim.tasks.{args.task}.task"
    )
    task = task_module.Task(args)

    # Main objects building
    world = World(physics_dt=task.scene_cfg.physics_dt)
    print("[INFO] World created")

    spot = task.build(world)

    task.on_setup(world, spot)

    print_sim_summary(task, args)

    # Main simulation loop
    step_count = 0
    reset_needed = False
    try:
        while simulation_app.is_running():
            if world.is_stopped():
                world.step(render=True)
                if not reset_needed:
                    reset_needed = True
                    step_count = 0
                    print("[INFO] Simulation stopped — waiting for restart...")
                continue

            if reset_needed:
                world.reset()
                spot.reinitialize()
                task.on_reinitialize(world, spot)
                reset_needed = False
                step_count = 0
                print("[INFO] Simulation restarted — controllers reinitialized")
                continue

            should_render = step_count % args.render_interval == 0
            world.step(render=should_render)
            step_count += 1

            task.update(world, spot, step_count)

            if step_count % (1000 * args.render_interval) == 0:
                print(f"[INFO] Step {step_count}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    print(f"[INFO] Simulation ended after {step_count} steps")
    task.on_shutdown(world, spot)
    world.stop()
    simulation_app.close()
    print("[INFO] Cleanup complete")


def print_sim_summary(task, args):
    print(f"[INFO] Robot joint count: {task.spot._robot.num_dof}")
    print("")
    print("=" * 60)
    print("  Spot Robot Spawned Successfully!")
    print("=" * 60)
    print(f"  Robot:  {task.robot_cfg.prim_path}")
    print(f"  Task:   {args.task}")
    print(f"  Cameras: {task.bridge_cfg.publishing_cameras} @ {task.bridge_cfg.camera_hz} Hz (gate_step={task.gate_step})")
    print(f"  Device: {args.device}")
    print(f"  Render Interval: Every {args.render_interval} step(s)")
    if args.headless:
        print(f"  Viewport: Disabled (headless mode)")
    if task.spot.ros2_bridge is not None and task.spot.ros2_bridge.success:
        print(f"  ROS2: Enabled")
        print(f"  ML nodes: run 'just ros-dl-nodes' in a separate terminal")
    if task.robot_cfg.enable_arm_ik:
        print(f"  IK: Enabled")
    if task.robot_cfg.enable_locomotion:
        print(f"  Locomotion: Enabled")
        print(
            f"  Collision Avoidance: {'Enabled' if task.robot_cfg.enable_collision_avoidance else 'Disabled'}"
        )
    else:
        print(f"  Locomotion: Disabled — base locked")
    print("=" * 60)
    print("")
    print("[INFO] Running simulation... (Press Ctrl+C or close window to exit)")


if __name__ == "__main__":
    main()
