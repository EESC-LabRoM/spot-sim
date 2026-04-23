import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from dataset_generation.assetsCfg.spot_gripperCfg.spot import SPOT_ARM_CFG
from dataset_generation.assetsCfg.objectsCfg.assets import DRILL_CFG

from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
)


@configclass
class SpotGripperSceneCfg(InteractiveSceneCfg):
    """Configuration for the cabinet scene with a robot and a cabinet.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    # robot
    robot: ArticulationCfg = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    target_object: RigidObjectCfg = DRILL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/TargetObject"
    )
    # End-effector
    # Listens to the required transforms
    # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
    # the other frames are the fingers
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_link_wr0",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(
            prim_path="/Visuals/EndEffectorFrameTransformer"
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/arm_link_wr1",
                name="ee_tcp",
                offset=OffsetCfg(
                    pos=(0.21, 0.0, -0.01),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/arm_link_fngr",
                name="tool_leftfinger",
                offset=OffsetCfg(
                    pos=(0.095456954, 0.0, -0.04),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/arm_link_wr1",
                name="tool_rightfinger",
                offset=OffsetCfg(
                    pos=(0.21, 0.0, -0.04),
                ),
            ),
        ],
    )
    # plane
    # plane = AssetBaseCfg(
    #     prim_path="/World/GroundPlane",
    #     init_state=AssetBaseCfg.InitialStateCfg(),
    #     spawn=sim_utils.GroundPlaneCfg(),
    #     collision_group=-1,
    # )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_link_fngr",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/TargetObject"],
    )

    # camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Camera",
    #     update_period=0.0,
    #     height=480,
    #     width=640,
    #     data_types=[
    #         "distance_to_image_plane",
    #         "semantic_segmentation",
    #         "rgb",
    #     ],  # -> required for point cloud
    #     spawn=sim_utils.PinholeCameraCfg(),
    #     colorize_semantic_segmentation=False,
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(1.0, 0.5, 0.5),
    #         rot=(0.5, 0.5, -0.5, -0.5),
    #         convention="world",
    #     ),
    #     semantic_filter=["class:*"],
    # )
    # visualize camera in the scene
    # camera_visual: AssetBaseCfg = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Camera/visual",
    #     spawn=sim_utils.CylinderCfg(
    #         radius=0.05,
    #         height=0.1,
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    #     ),
    #     # same location as camera
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.5, 0.5)),
    # )
