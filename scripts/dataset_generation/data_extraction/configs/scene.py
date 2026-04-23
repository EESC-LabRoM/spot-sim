import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from dataset_generation.assetsCfg.objectsCfg.assets import DRILL_CFG


@configclass
class RawDataExtractionSceneCfg(InteractiveSceneCfg):
    """Configuration for textraction raw data: depth, point Cloud etc"""

    # asset
    target_object: RigidObjectCfg = DRILL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/TargetObject"
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0),
    )

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=[
            "distance_to_image_plane",
            "semantic_segmentation",
            "rgb",
        ],  # -> required for point cloud
        spawn=sim_utils.PinholeCameraCfg(),
        colorize_semantic_segmentation=False,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0, 0.35, 0.575),  # best front view: (1.0, 0.35, 0.575)
            rot=(0.5, 0.5, -0.5, -0.5),
            convention="world",
        ),
        semantic_filter=["class:*"],
    )
    # visualize camera in the scene
    camera_visual: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Camera/visual",
        spawn=sim_utils.CylinderCfg(
            radius=0.05,
            height=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        # same location as camera
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),
    )
