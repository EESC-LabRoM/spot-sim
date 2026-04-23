# Spot IsaacSim — Simulation Environment

> Isaac Sim environment for Boston Dynamics Spot with 6 RGB-D cameras and ROS2 bridge

## Requirements

- Ubuntu 22.04.5 LTS
- CUDA 12.8+ (Recommended: RTX 3000+)
- ~30 GB of free disk

## Installation

### 0. Clone the repo

```bash
git clone --recurse-submodules <repo-url>
cd spot-isaacsim
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

### Run an arbitrary script

```bash
just run scripts/spot_isaacsim/play.py --headless
```

### Asset conversion (one-time)

```bash
just convert-spot-urdf                              # URDF → USD
just convert-obj assets/drill/drill.obj             # OBJ/GLB → USD
```

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
│   │   ├── scene/                # Scene setup (builder.py, scene_cfg.yaml)
│   │   ├── omnigraph/            # ROS2 bridge (OmniGraph)
│   │   ├── control/              # Grasp execution state machine
│   │   └── spot_config/          # Robot configuration
│   ├── spot_isaaclab/          # Isaac Lab environments (deprecated)
│   ├── spot_vlm/               # VLM tracking (deprecated)
│   ├── tools/                  # Utility scripts (install, convert, patch)
│   └── dataset_generation/     # Synthetic data generation
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

Cameras: `hand`, `frontleft`, `frontright`, `left`, `right`, `rear` (enabled via `--cameras` flag).

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

## Configuration

- **Scene**: Edit `scripts/spot_isaacsim/scene/scene_cfg.yaml` to change objects, stage type, robot spawn position, and lighting.
- **Environment**: Use `just run <script>` to run scripts inside the simulation Python environment (via `uv run`).
- **ROS2**: `ROS_DOMAIN_ID=77` is set automatically by all `just` recipes. Make sure the ROS2 consumer (NVBlox, ML nodes) uses the same domain ID.

## License

MIT License
