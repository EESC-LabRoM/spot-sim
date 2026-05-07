# Robot Control Interfaces

This simulation provides two primary interfaces to control the Spot robot: Keyboard Control and ROS 2 Control.

## Keyboard Control

The `StdinKeyboardController` (`simulation/scripts/spot_isaacsim/control/interfaces/keyboard_controller.py`) allows controlling the robot directly from the terminal or GUI window. It works seamlessly in both headless and headed modes.

> [!NOTE]
> In GUI (headed) mode, the terminal window must have focus for standard keys to register. However, a special listener ensures that pressing `NUMPAD_5` toggles the arm tracking regardless of window focus.

### Locomotion
The locomotion is mapped to terminal keys for real-time control over the robot base:

| Key | Action |
| --- | --- |
| `W` / `S` | Move forward / Move backward |
| `A` / `D` | Strafe left / Strafe right |
| `Q` / `E` | Rotate (Yaw) left / Rotate (Yaw) right |

### Arm & Gripper
Arm control involves toggling between preset positions and actuating the gripper:

| Key | Action |
| --- | --- |
| `5` | Toggle Arm Tracking |
| `0` | Stow Arm (Resting Position) |
| `1` | Extend Arm (Standing Position) |
| `Space` | Toggle Gripper (Open/Close) |

---

## ROS 2 Control

For programmatic and remote control, the simulation exposes a set of ROS 2 services and subscriptions via `ros_controller.py`. This mirrors the real robot's service interface, enabling the same ML inference scripts to work both in simulation and on the physical hardware.

### Grasp Executor Services

The `GraspExecutor` manages the arm's state machine to perform complex grasping sequences. All commands are exposed under the `~/` namespace (e.g., `/grasp_executor/approach`).

| Service Name | Description |
| --- | --- |
| `~/approach` | Opens the gripper and moves the arm to a pre-grasp pose based on the latest target from `/grasp/selected_grasp`. |
| `~/grasp` | Advances the arm along the approach axis and closes the gripper. Must be called after `~/approach`. |
| `~/retrieve` | Lifts the arm 20 cm above the grasp location and then stows it (gripper remains closed). |
| `~/approach_and_grasp` | Chains `~/approach` and `~/grasp` automatically into a single fluid action. |
| `~/pick` | Performs the full pick sequence: `~/approach` → `~/grasp` → `~/retrieve`. |
| `~/home` | Stows the arm to the resting joint position and opens the gripper. |
| `~/trajectory` | Replays a custom `trajectory_msgs/JointTrajectory` sent to `/graspgen/arm_joint_command`. |

> [!IMPORTANT]
> The target grasping pose must be published as a `geometry_msgs/PoseStamped` message to the `/grasp/selected_grasp` topic before calling `~/approach`, `~/approach_and_grasp`, or `~/pick`.

### Navigation Executor

The `NavigationExecutor` drives the Spot base to a specific 2D goal coordinate.

To navigate the robot, publish a `geometry_msgs/PoseStamped` message to the `/goal_pose` topic. This is compatible with the standard **RViz 2D Nav Goal** tool.

Once a goal is received, the robot transitions from `IDLE` to `NAVIGATING` and will automatically stop once it arrives at the target position (`ARRIVED`), before returning to the `IDLE` state.
