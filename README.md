# Spot IsaacSim — Simulation Environment

> Isaac Sim environment for Boston Dynamics Spot with 6 RGB-D cameras and ROS2 bridge

<table>
  <tr>
    <td style="padding:2px"><img src=".midia/output2.gif"/></td>
    <td style="padding:2px"><img src=".midia/output3.gif"/></td>
  </tr>
  <tr>
    <td colspan="2" style="padding:2px"><img src=".midia/output1.gif" width="100%"/></td>
  </tr>
</table>

## Requirements

- Ubuntu 22.04.5 LTS
- CUDA 12.8+ (Recommended: RTX 3000+)
- ~30 GB of free disk

## Installation

### 0. Clone the repo

```bash
git clone --recurse-submodules <repo-url>
cd spot-sim
```

### 1. Install `just`

```bash
sudo snap install just --classic
```

### 2. Install the simulation environment

```bash
just install-sim
```

> [!IMPORTANT]
> Isaac Sim already has internal ROS2 libraries. **Do not source your local ROS2 installation** (`source /opt/ros/<distro>/setup.bash`) in `.bashrc` — this causes conflicts when running Isaac Sim.

## Usage

Run `just` to see all available commands:

```bash
just
```

### Run the simulation

```bash
just run-spot-sim          # with viewport
just run-spot-sim-h        # headless mode
```

The `--task` flag selects a named task configuration (default: `_default`):

```bash
just run-spot-sim -- --task my_task
```

Each task lives in `scripts/spot_isaacsim/tasks/<name>/task.py` and owns its scene build, camera/bridge config, and lifecycle hooks.

## Project Structure

```
spot-isaacsim/
├── assets/                     # Robot + object assets (Spot, drill, jar, etc.)
├── external/                   # Git submodules
│   ├── relic/                    # Spot URDF/USD assets
│   ├── curobo/                   # Motion planning & IK
│   ├── IsaacRobotics/            # Isaac Robotics utilities
│   └── zed-isaac-sim/            # ZED camera extension for Isaac Sim
├── scripts/
│   ├── spot_isaacsim/          # Main: Isaac Sim + ROS2 bridge
│   │   ├── play.py               # Entry point (just run-spot-sim)
│   │   ├── tasks/                # Named task configs (--task flag)
│   │   ├── scene/                # Scene setup (builder.py, loaders/, scene_cfg.yaml)
│   │   ├── omnigraph/            # ROS2 bridge (OmniGraph)
│   │   ├── control/
│   │   │   ├── components/       # Locomotion, arm IK, collision avoidance
│   │   │   └── interfaces/       # Keyboard and ROS controllers
│   │   └── spot_config/          # Robot configuration and constants
│   └── tools/                  # Utility scripts (install, patch)
├── justfile                    # Task runner (run 'just' for all commands)
└── pyproject.toml              # uv dependency management
```

## ROS2 Interface

The simulation node is named `grasp_executor` and communicates over `ROS_DOMAIN_ID=77`.

### Published topics

| Topic | Type | Description |
|---|---|---|
| `/spot/camera/<cam>/image` | `sensor_msgs/Image` | RGB image per camera |
| `/spot/camera/<cam>/camera_info` | `sensor_msgs/CameraInfo` | RGB camera intrinsics |
| `/spot/depth_registered/<cam>/image` | `sensor_msgs/Image` | Depth image per camera |
| `/spot/depth_registered/<cam>/camera_info` | `sensor_msgs/CameraInfo` | Depth camera intrinsics |
| `/joint_states` | `sensor_msgs/JointState` | Robot joint states |
| `/tf` / `/tf_static` | `tf2_msgs/TFMessage` | Robot + camera transforms |

Cameras: `hand`, `frontleft`, `frontright`, `left`, `right`, `rear`. Which cameras are published is configured per-task via `publishing_cameras` in the task's `BridgeConfig`.

### Subscribed topics

| Topic | Type | Description |
|---|---|---|
| `/graspgen/arm_joint_command` | `trajectory_msgs/JointTrajectory` | Arm joint trajectory commands |

### Services (std_srvs/Trigger)

| Service | Description |
|---|---|
| `/grasp_executor/pick` | Full sequence: approach → grasp → retrieve |
| `/grasp_executor/approach` | Move arm to pre-grasp pose |
| `/grasp_executor/grasp` | Close gripper |
| `/grasp_executor/retrieve` | Lift arm after grasp |
| `/grasp_executor/approach_and_grasp` | Approach + close gripper |
| `/grasp_executor/move_to` | Walk robot to face target |
| `/grasp_executor/home` | Return arm to home position |
| `/grasp_executor/trajectory` | Execute a joint trajectory |

## Documentation

| Doc | Description |
|---|---|
| [docs/tasks.md](docs/tasks.md) | Task plugin system — how to create and configure tasks |
| [docs/robot_control.md](docs/robot_control.md) | Keyboard and ROS2 control interfaces |

## Configuration

- **Tasks**: Create `scripts/spot_isaacsim/tasks/<name>/task.py` with a `Task` class implementing `build()`, `on_setup()`, `update()`, `on_reinitialize()`, `on_shutdown()`. Select it with `--task <name>`.
- **Scene**: Edit `scripts/spot_isaacsim/scene/scene_cfg.yaml` to change objects, stage, robot spawn, and lighting. Asset physics uses a nested `physics:` block (`rigid_body`, `material`).
- **Environment**: Use `just run <script>` to run scripts inside the simulation Python environment (via `uv run`).
- **ROS2**: `ROS_DOMAIN_ID=77` and `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` are written into the venv activate script at install time. Make sure the ROS2 consumer uses the same domain ID.

## License

MIT License
