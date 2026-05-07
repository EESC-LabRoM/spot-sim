# Simulation Tasks

The Isaac Sim simulation uses a modular task system to define different environments, scenarios, and robot configurations. Tasks encapsulate the setup logic and the simulation step logic.

## Task Architecture

Tasks are dynamically loaded by `play.py` based on the `--task` argument. By default, running `just run scripts/spot_isaacsim/play.py` will load the `default` task.

To run a specific task, use:
```bash
python scripts/spot_isaacsim/play.py --task <task_name>
```

When a task is specified, `play.py` attempts to import `Task` from `scripts.spot_isaacsim.tasks.<task_name>.task`.

## The `BaseTask` Class

All tasks must inherit from `BaseTask` (`simulation/scripts/spot_isaacsim/tasks/base_task.py`). This base class defines the lifecycle of a scenario.

### Lifecycle Hooks

When a simulation is run, `play.py` calls the following methods in order:

1. **`build(world)`**: Instantiates the `SceneBuilder` and `SpotRobot`. It uses the configuration objects defined in the task's `__init__`.
2. **`on_setup(world, robot)`**: Called once right after the simulation is ready and the robot is spawned. Useful for spawning custom scene objects or setting initial states.
3. **`update(world, robot, step)`**: Called at every step of the physics simulation. This is where active controllers, state machines, and task logic should be executed.
4. **`on_reinitialize(world, robot)`**: Triggered when the simulation is restarted (e.g., stopping and pressing play in the GUI).
5. **`on_shutdown(world, robot)`**: Called upon a clean exit to allow for cleanup.

### Subsystem Configuration

A task configures the simulation by overriding the config objects in its `__init__` method:

- `self.scene_cfg_path`: Path to a YAML file defining the static scene and rigid bodies (resolved by `SceneBuilder`).
- `self.robot_cfg`: An instance of `RobotConfig` to toggle features like IK, Locomotion, and Collision Avoidance.
- `self.bridge_cfg`: Configuration for the ROS 2 Bridge.
- `self.zed_cfg`: Configuration for the ZED Camera extension.

## Creating a New Task

To create a new task named `my_scenario`:

1. Create a directory: `simulation/scripts/spot_isaacsim/tasks/my_scenario/`
2. Create a `task.py` file inside this directory.
3. Define a `Task` class inheriting from `BaseTask`.

```python
import argparse
from pathlib import Path
from scripts.spot_isaacsim.tasks.base_task import BaseTask

class Task(BaseTask):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        
        # 1. Define custom configurations
        self.scene_cfg_path = str(Path(__file__).parent / "scene.yaml")
        
        # Disable locomotion for this specific task
        self.robot_cfg.enable_locomotion = False

    def on_setup(self, world, robot):
        # 2. Add custom setup logic
        print("My Scenario Initialized!")

    def update(self, world, robot, step: int) -> None:
        # 3. Add per-step update logic
        robot.update()
```

You can then run your new task with:
```bash
python scripts/spot_isaacsim/play.py --task my_scenario
```
