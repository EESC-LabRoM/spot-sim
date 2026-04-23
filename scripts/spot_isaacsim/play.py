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
import time
import os

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
        "--camera-hz",
        type=float,
        default=10.0,
        help="Camera publish frequency in Hz (default: 10)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=["hand", "frontleft", "frontright"], # Also accept "left", "right", "rear"
        metavar="CAM",
        help="Cameras to enable (default: hand and front cams). E.g. --cameras hand frontleft",
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
    env['PYTHONPATH'] = os.pathsep.join(sys.path) + os.pathsep + env.get('PYTHONPATH', '')

    # Force FastDDS Large Data mode to prevent UDP buffer overflows
    env['FASTDDS_BUILTIN_TRANSPORTS'] = 'LARGE_DATA'
    env['ROS_DOMAIN_ID'] = os.environ.get('ROS_DOMAIN_ID', '77')

    # ROS2 runtime environment
    os.environ.setdefault('ROS_DISTRO', 'jazzy')
    os.environ.setdefault('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    _venv_dir = Path(sys.executable).parent.parent
    _ros_bridge_lib = (
        _venv_dir / 'lib' / 'python3.11' / 'site-packages'
        / 'isaacsim' / 'exts' / 'isaacsim.ros2.bridge'
        / os.environ['ROS_DISTRO'] / 'lib'
    )
    os.environ['LD_LIBRARY_PATH'] = (
        str(_ros_bridge_lib) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    )

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
    zed_ext_path = PROJECT_ROOT / "external" / "zed-isaac-sim" / "exts"
    if zed_ext_path.exists():
        manager.add_path(str(zed_ext_path))
        manager.set_extension_enabled_immediate("sl.sensor.camera", True)
        print(f"[INFO] ZED extension loaded from: {zed_ext_path}")
    else:
        print(f"[WARN] ZED extension not found at {zed_ext_path}")
        print(f"[WARN] ZED camera will not be available. To enable, clone the zed-isaac-sim repo and ensure the ext path exists: {zed_ext_path}")

    # Isaac Sim modules — must import after SimulationApp and extensions are ready
    from isaacsim.core.api import World

    from scripts.spot_isaacsim.spot_config.robot import SpotRobot
    from scripts.spot_isaacsim.scene import SceneBuilder, SceneConfig

    config = SceneConfig()

    # How many render frames between camera publishes
    gate_step = max(1, round(1.0 / (config.physics_dt * args.render_interval * args.camera_hz)))

    # Create World
    world = World(physics_dt=config.physics_dt)
    print(f"[INFO] World created")

    scene = SceneBuilder(world, config.robot)
    robot = scene.objects["robot"]

    spot = SpotRobot(robot, world, config,
                     device=args.device,
                     enabled_cameras=set(args.cameras),
                     gate_step=gate_step,
                     headless=args.headless)

    print(f"[INFO] Robot joint count: {robot.num_dof}")

    print("")
    print("=" * 60)
    print("  Spot Robot Spawned Successfully!")
    print("=" * 60)
    print(f"  Robot:  {config.robot.prim_path}")
    print(f"  Table:  /World/Table")
    print(f"  Target: /World/Target")
    print(f"  Cameras: {args.cameras} @ {args.camera_hz} Hz (gate_step={gate_step})")
    print(f"  Device: {args.device}")
    print(f"  Render Interval: Every {args.render_interval} step(s)")
    if args.headless:
        print(f"  Viewport: Disabled (headless mode)")
    if spot.ros2_bridge.success:
        print(f"  ROS2: Enabled")
        print(f"  ML nodes: run 'just ros-dl-nodes' in a separate terminal")
    print(f"  IK: Enabled — /grasp/set_target → /grasp/execute")
    print(f"  Locomotion: Enabled (500 Hz physics, 50 Hz policy + control rate)")
    print(f"  Controls: W/S=fwd/back, A/D=strafe, Q/E=turn")
    print(f"  Collision Avoidance: Enabled (threshold=0.30m, 50 Hz)")
    print("=" * 60)
    print("")
    print("[INFO] Running simulation... (Press Ctrl+C or close window to exit)")

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
                spot.reinitialize()
                reset_needed = False
                step_count = 0
                print("[INFO] Simulation restarted — controllers reinitialized")
                continue

            should_render = (step_count % args.render_interval == 0)
            world.step(render=should_render)
            step_count += 1

            spot.update()

            if step_count % (1000 * args.render_interval) == 0:
                print(f"[INFO] Step {step_count}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    print(f"[INFO] Simulation ended after {step_count} steps")
    world.stop()
    simulation_app.close()
    print("[INFO] Cleanup complete")


if __name__ == "__main__":
    main()
