"""
Safely convert Meshes to .usd and add Physics for Isaac Lab.
    1. Generates an .usd from Mesh file
    2. Add Physics to the .usd file generated
"""

import argparse
import asyncio
import os
import sys

# Add the parent directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up to dataset_generation directory: objectsCfg -> assetsCfg -> dataset_generation
dataset_generation_dir = os.path.dirname(os.path.dirname(script_dir))
# Go up one more to scripts directory
scripts_dir = os.path.dirname(dataset_generation_dir)
# Add scripts directory to path so we can import dataset_generation
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from isaaclab.app import AppLauncher
from dataset_generation import ASSET_DIR

# paths to mesh files
INPUT_MESH_PATH = f"{ASSET_DIR}/downloaded_meshes/drill.glb"
OUTPUT_USD_PATH = f"{ASSET_DIR}/converted_usd/drill.usd"

# physics parameters
DENSITY = 100.0  # kg/m^3
COLLISION_OFFSET = 0.001
# ---------------------------------------------------------

# Launch App
parser = argparse.ArgumentParser(description="Convert Mesh to USD")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# imports
import omni.kit.asset_converter
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
import isaaclab.sim.schemas as schemas
import isaaclab.sim.schemas.schemas_cfg as schemas_cfg


async def convert_fbx_to_usd(input_path, output_path):
    """Run the Omniverse Asset Converter."""
    print(f"[INFO] Converting '{input_path}' to USD...")

    # Create converter context
    context = omni.kit.asset_converter.AssetConverterContext()
    context.ignore_materials = False
    context.merge_all_meshes = True
    context.use_meter_as_world_unit = True
    context.baking_scales = True
    context.use_double_precision_to_usd_transform_op = True

    # Create and run task
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(input_path, output_path, None, context)
    success = await task.wait_until_finished()

    if not success:
        raise RuntimeError(
            f"Failed to convert {input_path} to USD: {task.get_error_message()}"
        )
    print(f"[INFO] Raw conversion successful: {output_path}")


def add_physics_properties(usd_path):
    """Add Rigid Body and Collision properties to the converted USD."""
    print(f"[INFO] Adding physics properties to {usd_path}...")

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Could not open stage: {usd_path}")

    # find the root Mesh prim -> converted assets often put the mesh under a root Xform
    # we traverse to find the first mesh type prim to apply physics to
    target_prim = None
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            target_prim = prim
            break

    if not target_prim:
        print("[WARNING] No Mesh prim found!! Applying physics to Default Prim.")
        target_prim = stage.GetDefaultPrim()

    prim_path = target_prim.GetPath().pathString
    print(f"[INFO] Applying physics to prim: {prim_path}")

    # define Rigid Body Properties
    rb_cfg = schemas_cfg.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        disable_gravity=False,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=10.0,
    )
    schemas.define_rigid_body_properties(prim_path, rb_cfg, stage)

    # define Collision Properties
    col_cfg = schemas_cfg.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.001,
        rest_offset=COLLISION_OFFSET,
    )
    schemas.define_collision_properties(prim_path, col_cfg, stage)

    # define Collision Approximation -> Convex Hull
    mesh_col_cfg = schemas_cfg.ConvexHullPropertiesCfg()
    schemas.define_mesh_collision_properties(prim_path, mesh_col_cfg, stage)

    # define mass -> based on density
    mass_cfg = schemas_cfg.MassPropertiesCfg(
        density=DENSITY,
    )
    schemas.define_mass_properties(prim_path, mass_cfg, stage)

    # save changes
    stage.GetRootLayer().Save()
    print(f"[INFO] Physics added!! Asset ready at: {usd_path}")


def main():
    # ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_USD_PATH), exist_ok=True)

    # run async conversion
    asyncio.get_event_loop().run_until_complete(
        convert_fbx_to_usd(INPUT_MESH_PATH, OUTPUT_USD_PATH)
    )

    # run physics setup
    add_physics_properties(OUTPUT_USD_PATH)


if __name__ == "__main__":
    main()
    simulation_app.close()
