"""
Factory functions for creating primitive scene objects in Isaac Sim.

Functions:
    create_ground_plane  — static ground plane
    create_dome_light    — ambient dome light
    create_static_cube   — non-moving FixedCuboid
    create_dynamic_cube  — physics-enabled DynamicCuboid
"""

from typing import Tuple
import numpy as np

from .utils import rpy_deg_to_quat, _set_friction


def create_ground_plane(
    world,
    prim_path: str = "/World/GroundPlane",
    size: float = 10.0,
    z_position: float = 0.0,
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> None:
    """Create a static ground plane."""
    from isaacsim.core.api.objects import GroundPlane

    ground = GroundPlane(
        prim_path=prim_path,
        size=size,
        z_position=z_position,
        color=np.array(color),
    )
    world.scene.add(ground)
    print(f"[Scene] Ground plane created at {prim_path}")


def create_dome_light(
    stage,
    prim_path: str = "/World/DomeLight",
    intensity: float = 4000.0,
    color: Tuple[float, float, float] = (0.9, 0.9, 0.9),
) -> None:
    """Create a dome light for ambient illumination."""
    from pxr import Gf, UsdLux

    light = UsdLux.DomeLight.Define(stage, prim_path)
    light.GetIntensityAttr().Set(intensity)
    light.GetColorAttr().Set(Gf.Vec3f(*color))
    print(f"[Scene] Dome light created at {prim_path}")


def create_static_cube(
    world,
    prim_path: str,
    size: Tuple[float, float, float],
    position: Tuple[float, float, float],
    color: Tuple[float, float, float],
    orientation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    static_friction: float = 0.7,
    dynamic_friction: float = 0.7,
):
    """Create a static (non-moving) cube.

    Args:
        size: Cube dimensions (x, y, z) applied as scale.
        position: Position (x, y, z).
        color: RGB color.
        orientation_rpy: Rotation (roll, pitch, yaw) in degrees — ZYX extrinsic.
        static_friction: Static friction coefficient.
        dynamic_friction: Dynamic friction coefficient.

    Returns:
        FixedCuboid object.
    """
    from isaacsim.core.api.objects import FixedCuboid

    cube = FixedCuboid(
        prim_path=prim_path,
        name=prim_path.split("/")[-1],
        size=1.0,
        scale=np.array(size),
        position=np.array(position),
        orientation=rpy_deg_to_quat(orientation_rpy),
        color=np.array(color),
    )
    world.scene.add(cube)
    _set_friction(cube.prim, static_friction, dynamic_friction)
    print(f"[Scene] Static cube '{prim_path.split('/')[-1]}' created at {prim_path}")
    return cube


def create_dynamic_cube(
    world,
    prim_path: str,
    size: Tuple[float, float, float],
    position: Tuple[float, float, float],
    color: Tuple[float, float, float],
    mass: float = 0.1,
    orientation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Create a dynamic (physics-enabled) cube.

    Args:
        size: Cube dimensions (x, y, z) applied as scale.
        position: Position (x, y, z).
        color: RGB color.
        mass: Object mass in kg.
        orientation_rpy: Rotation (roll, pitch, yaw) in degrees — ZYX extrinsic.

    Returns:
        DynamicCuboid object.
    """
    from isaacsim.core.api.objects import DynamicCuboid

    cube = DynamicCuboid(
        prim_path=prim_path,
        name=prim_path.split("/")[-1],
        size=1.0,
        scale=np.array(size),
        position=np.array(position),
        orientation=rpy_deg_to_quat(orientation_rpy),
        color=np.array(color),
        mass=mass,
    )
    world.scene.add(cube)
    print(f"[Scene] Dynamic cube '{prim_path.split('/')[-1]}' created at {prim_path}")
    return cube
