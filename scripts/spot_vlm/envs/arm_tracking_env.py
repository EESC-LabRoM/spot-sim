"""
Spot Arm Tracking Environment
CORREÇÃO: 6 DOF (sem gripper) - Atualizado para compatibilidade com nova estrutura de prompts
"""
import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg, ObservationGroupCfg

import omni.graph.core as og
import usdrt.Sdf

from scripts.spot_vlm.configs import global_config as config
from scripts.spot_vlm.configs.scene_config import SpotManipulationSceneCfg, SpotManipulationWithObjectsSceneCfg
from scripts.spot_vlm.controllers.robot_controller import RobotController, setup_robot_controller
from scripts.spot_vlm.utils.robot_initialization import initialize_robot_sequence, get_arm_joint_info
from scripts.spot_vlm.inference.depth_processing import process_depth_map, calculate_camera_adjustments

# Tentar importar VLM
try:
    from scripts.spot_vlm.inference.vlm_inference import VLMInference
except ImportError:
    VLMInference = None

# Tentar importar Curobo
try:
    from scripts.spot_vlm.controllers.motion_planner import CuroboMotionPlanner
    CUROBO_AVAILABLE = True
except (ImportError, Exception, RuntimeError) as e:
    print(f"[WARN] Curobo indisponível: {e}")
    CUROBO_AVAILABLE = False


def dummy_observation(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, 1, device=env.device)

@configclass
class DummyObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        dummy = ObservationTermCfg(func=dummy_observation)
    policy: PolicyCfg = PolicyCfg()

@configclass
class DummyActionsCfg:
    pass

@configclass
class SpotArmTrackingEnvCfg(ManagerBasedEnvCfg):
    """Configuração do environment"""
    if config.USE_EXTENDED_SCENE:
        scene: SpotManipulationWithObjectsSceneCfg = SpotManipulationWithObjectsSceneCfg(num_envs=1, env_spacing=2.0)
    else:
        scene: SpotManipulationSceneCfg = SpotManipulationSceneCfg(num_envs=1, env_spacing=2.0)
    observations: DummyObservationsCfg = DummyObservationsCfg()
    actions: DummyActionsCfg = DummyActionsCfg()

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 100.0
        self.viewer.eye = (2.0, 2.0, 1.0)
        self.viewer.lookat = (0.3, 0.0, 0.0)
        self.sim.dt = 0.01
        self.sim.render_interval = config.RENDER_EVERY_N_STEPS
        self.sim.device = "cuda:0"
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_max_rigid_contact_count = 2**20
        self.sim.physx.enable_stabilization = True
        self.sim.physx.solver_type = 1
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**20
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**20
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005


class SpotArmTrackingEnv(ManagerBasedEnv, gym.Env):
    """Environment para tracking com braço (6 DOF)"""
    cfg: SpotArmTrackingEnvCfg

    def __init__(self, cfg: SpotArmTrackingEnvCfg, **kwargs):
        super().__init__(cfg=cfg)
        # CORREÇÃO: Action space agora é 6 DOF
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self._setup_robot()
        self._setup_vlm()
        self._setup_ros_bridge()

        self.step_count = 0
        self.current_command = "search"
        self.tracking_mode = False
        self.last_inference_step = 0
        self.consecutive_failures = 0
        self.first_reset_done = False

        # Control rate limiting
        self.last_control_step = 0
        self.control_interval = config.CONTROL_EVERY_N_STEPS

        # Fallback system
        self.fallback_active = False
        self.fallback_steps = 0
        self.max_fallback_steps = 100
        self.fallback_init_step = 0
        self.fallback_target = None
        self.max_consecutive_failures = config.MAX_CONSECUTIVE_FAILURES

        # CORREÇÃO: Usar apenas 6 DOF
        self.controller.apply_joint_targets(self.controller.initial_arm_positions, lock_base=True)
        self._pending_arm_target = self.controller.initial_arm_positions.clone()

        print("[ENV] Spot Arm Tracking inicializado (6 DOF) - MODO: BUSCA")

    def _setup_robot(self):
        self.robot = self.scene["robot"]
        # Pega TODOS os índices (incluindo gripper)
        arm_joint_indices_all, _, _ = get_arm_joint_info(self.robot)

        if config.APPLY_SPOT_COLORS:
            try:
                from scripts.spot_vlm.utils.visualization import apply_spot_colors
                apply_spot_colors(self.sim)
            except Exception: pass

        # CORREÇÃO: Passar todos os índices, o controller separa internamente
        initialize_robot_sequence(self.robot, self.scene, self.sim, arm_joint_indices_all)

        # Controlador unificado (separa 6 DOF + gripper internamente)
        self.controller = setup_robot_controller(self.robot)

        # Curobo (opcional)
        self.motion_planner = None
        if CUROBO_AVAILABLE and config.USE_CUROBO:
            try:
                self.motion_planner = CuroboMotionPlanner(self.robot)
                print("[ENV] ✓ Curobo inicializado (6 DOF)")
            except Exception as e:
                print(f"[ENV] Curobo indisponível: {e}")

    def _setup_vlm(self):
        if config.VLM_ENABLED and VLMInference is not None:
            try:
                self.vlm = VLMInference(config.VLM_MODEL_PATH)
                self.vlm.start_async_loop()
                print(f"[ENV] VLM iniciado")
            except Exception as e:
                print(f"[ENV] VLM falhou: {e}")
        else:
            self.vlm = None

    def _setup_ros_bridge(self):
        """
        Creates the Action Graph to publish ROS 2 data for nvblox.
        Publishes: /tf, /joint_states, /image, /depth, /camera_info
        """
        # Only setup for the first environment to avoid duplicate topics
        if self.scene.num_envs > 1:
            print("[WARN] ROS Bridge only configured for env 0")

        # Get actual spawned prim paths (not config paths)
        # In Isaac Lab:
        #   - Articulations use: root_physx_view.prim_paths[0]
        #   - Sensors use: _view.prim_paths[0]
        robot_prim_path = self.scene["robot"].root_physx_view.prim_paths[0]
        camera_prim_path = self.scene["camera"]._view.prim_paths[0]

        # Verify prims exist in the stage
        import omni.usd
        stage = omni.usd.get_context().get_stage()

        print(f"[DEBUG] Attempting to find robot at: {robot_prim_path}")
        print(f"[DEBUG] Attempting to find camera at: {camera_prim_path}")

        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        camera_prim = stage.GetPrimAtPath(camera_prim_path)

        if not robot_prim.IsValid():
            print(f"[ERROR] Robot prim does not exist at: {robot_prim_path}")
            return
        if not camera_prim.IsValid():
            print(f"[ERROR] Camera prim does not exist at: {camera_prim_path}")
            return

        try:
            render_product_path = self.scene["camera"].render_product_paths[0]
        except Exception as e:
            print(f"[ERROR] Could not retrieve render product from camera sensor: {e}")
            return

        # try:
        #     joint_states_path = self.scene["robot"].cfg.joint_states_path[0]
        # except Exception as e:
        #     print(f"[ERROR] Could not retrieve joint states path from robot: {e}")
        #     return

        print(f"[ENV] ROS Bridge Config:")
        print(f"  > Robot Prim: {robot_prim_path}")
        print(f"  > Robot Prim Valid: {robot_prim.IsValid()}")
        print(f"  > Robot Prim Type: {robot_prim.GetTypeName()}")
        print(f"  > Camera Prim: {camera_prim_path}")
        print(f"  > Camera Prim Valid: {camera_prim.IsValid()}")
        print(f"  > Render Product: {render_product_path}")
        # -------------------------------------


        graph_path = "/ActionGraph_ROS"

        # 2. Define the Graph
        try:
            keys = og.Controller.Keys
            (graph, nodes, _, _) = og.Controller.edit(
                {"graph_path": graph_path, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("PubJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                        ("PubTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                        ("PubCamera", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ],
                    keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "PubJointState.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "PubTF.inputs:execIn"),
                        ("OnPlaybackTick.outputs:tick", "PubCamera.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PubJointState.inputs:timeStamp"),
                        ("ReadSimTime.outputs:simulationTime", "PubTF.inputs:timeStamp"),
                    ],
                    keys.SET_VALUES: [
                        # Joint State - MUST be a list of usdrt.Sdf.Path objects
                        ("PubJointState.inputs:targetPrim", [usdrt.Sdf.Path(robot_prim_path)]),
                        ("PubJointState.inputs:topicName", "/joint_states"),

                        # TF Tree - targetPrims MUST be a list of usdrt.Sdf.Path objects
                        ("PubTF.inputs:targetPrims", [
                            usdrt.Sdf.Path(robot_prim_path),
                            usdrt.Sdf.Path(camera_prim_path),
                        ]),

                        # Camera (Uses Render Product Path)
                        ("PubCamera.inputs:frameId", "spot_camera_frame"),
                        ("PubCamera.inputs:topicName", "/spot/camera/hand/image"),
                        ("PubCamera.inputs:renderProductPath", render_product_path),
                        ("PubCamera.inputs:type", "rgb"),
                    ],
                },
            )
            print("[ENV] ROS Bridge Graph successfully created!")

        except Exception as e:
            print(f"[ERROR] Failed to create ROS Bridge: {e}")

            # NOTE: For nvblox you need RGB AND Depth.
            # You typically create multiple CameraHelper nodes or configure one to do both.
            # To add Depth, you would repeat the CameraHelper setup changing type to 'depth'
            # and topic to '/front_left_spot/depth'.

        print("[ENV] ROS Bridge Graph successfully created!")

    def step(self, action):
        self.step_count += 1
        self._pre_physics_step(action)

        # CRÍTICO: Apply targets APENAS UMA VEZ por frame
        self._apply_control_once()

        self.scene.write_data_to_sim()
        self.sim.step(render=(self.step_count % self.cfg.sim.render_interval == 0))
        self.scene.update(dt=self.sim.get_physics_dt())
        return self._get_observations(), 0.0, False, False, {}

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        self._reset_idx(None)
        return self._get_observations(), {}

    def _pre_physics_step(self, actions):
        """
        Lógica de decisão (NÃO aplica comandos aqui).
        Apenas calcula próximo target (6 DOF).
        """
        # VLM Assíncrono
        if self.vlm and self.vlm.available:
            result = self.vlm.get_async_result()
            if result:
                self._process_vlm_result(result)
                self.last_inference_step = self.step_count

            if (self.step_count - self.last_inference_step) >= config.VLM_INFERENCE_INTERVAL:
                self._submit_to_vlm()

        # Decidir próxima ação (rate limited)
        if (self.step_count - self.last_control_step) >= self.control_interval:
            if self.fallback_active:
                self._prepare_fallback_target()
            elif self.tracking_mode:
                self._prepare_arm_target()

            self.last_control_step = self.step_count

    def _apply_control_once(self):
        """
        ÚNICA função que chama set_joint_position_target.
        CORREÇÃO: Trabalha com 6 DOF.
        """
        if self.fallback_active and self.fallback_target is not None:
            self.controller.apply_joint_targets(self.fallback_target, lock_base=True, gripper_scaling_factor=1.0)
        elif hasattr(self, '_pending_arm_target') and self._pending_arm_target is not None:
            #print("[CONTROL] Aplicando novo target do braço")
            self.controller.apply_joint_targets(self._pending_arm_target, lock_base=True, gripper_scaling_factor=1.0)
            #self._pending_arm_target = None
        else:
            #print("[CONTROL] Manter posição atual")
            self.controller.apply_joint_targets(None, lock_base=True, gripper_scaling_factor=1.0)

    def _submit_to_vlm(self):
        """Envia imagem para VLM - Compatível com novo VLM Inference"""
        try:
            rgb_data = self.scene["camera"].data.output["rgb"][0]
            depth_data = self.scene["camera"].data.output["distance_to_image_plane"][0]

            if rgb_data is None or depth_data is None: return

            rgb_pil = self.vlm.process_image(rgb_data[:, :, :3])

            depth_np = depth_data.cpu().numpy()
            if len(depth_np.shape) == 3: depth_np = depth_np[:, :, 0]

            depth_colored = process_depth_map(depth_data)
            from PIL import Image
            depth_pil = Image.fromarray(depth_colored).resize((336, 336), Image.Resampling.LANCZOS)

            if config.SAVE_IMAGES and (self.step_count % config.SAVE_IMAGES_EVERY_N_STEPS == 0):
                rgb_pil.save("arm_vision_rgb.jpg")
                depth_pil.save("arm_vision_depth.jpg")

            adjustments = calculate_camera_adjustments(depth_np, target_distance=config.TARGET_DISTANCE)

            # ATUALIZADO: Passar explicitamente os modos suportados pelo novo inference
            # "tracking" mapeia para o prompt de posição original (sem atitude)
            current_mode = "search" if not self.tracking_mode else "tracking"

            self.vlm.submit_async_request(
                rgb_pil, depth_pil, adjustments['distance'],
                mode=current_mode
            )

        except Exception as e:
            print(f"[ENV SUBMIT ERROR] {e}")

    def _process_vlm_result(self, response):
        """Processa resposta do VLM"""
        if not response:
            return

        # print(f"[VLM RESPONSE] {response}")

        # MODO BUSCA
        if not self.tracking_mode:
            if "status" in response:
                if response["status"] == "target_found":
                    print(f"[VLM] ✓ Objeto encontrado! Iniciando tracking...")
                    self.tracking_mode = True
                    self.current_command = "hold_position"
                else:
                    print(f"[VLM] Buscando objeto...")

        # MODO TRACKING
        else:
            if "command" in response:
                new_command = response["command"].lower().strip()
                valid_commands = ["move_closer", "move_away", "adjust_left", "adjust_right",
                                "adjust_up", "adjust_down", "hold_position"]

                if new_command in valid_commands:
                    if self.current_command != new_command:
                        print("\n" + 5*"-" + "VLM INFERENCE" + 5*"-")
                        print(f"[VLM] {self.current_command} -> {new_command.upper()}")
                        if "reason" in response:
                            print(f"[VLM REASON] {response['reason']}")
                    self.current_command = new_command
                    # Reseta falhas em comando válido
                    if new_command != "hold_position":
                         self.consecutive_failures = 0
                else:
                    self.consecutive_failures += 1

    def _prepare_arm_target(self):
        """
        Calcula próximo target do braço (6 DOF).
        Armazena em self._pending_arm_target.
        """
        try:
            if self.current_command in ["hold_position"]:
                #self._pending_arm_target = None
                return

            # CORREÇÃO: Pega apenas 6 DOF
            current_arm_pos = self.robot.data.joint_pos[0, self.controller.arm_joint_indices].clone()
            curr_ee_pos, curr_ee_quat = self.controller.get_current_ee_pose()

            # Usar apenas comando de posição (tracking original)
            tgt_pos, tgt_quat = self.controller.compute_target_pose_from_command(
                curr_ee_pos, curr_ee_quat, self.current_command, step=0.04) # step um pouco maior 4cm

            # Debug target
            # print(f"[TARGET POS] {tgt_pos.cpu().numpy()}")

            if self.motion_planner is not None:
                # CORREÇÃO: Motion planner agora retorna 6 DOF
                target_config = self.motion_planner.plan_to_pose(tgt_pos, tgt_quat, current_arm_pos)

                if target_config is not None:
                    # Validar que retornou 6 valores
                    if target_config.numel() != 6:
                        print(f"[CONTROL] ✗ Planner retornou {target_config.numel()} valores (esperado 6)")
                        self._pending_arm_target = None
                        return

                    # Interpolação suave
                    self._pending_arm_target = self.controller.interpolate_arm_config(
                        current_arm_pos, target_config, alpha=0.3)
                    self.consecutive_failures = 0
                else:
                    self._pending_arm_target = None
                    self.consecutive_failures += 1
                    print(f"[CONTROL] Falha {self.consecutive_failures}/{self.max_consecutive_failures} no planejamento")

                    if self.consecutive_failures >= self.max_consecutive_failures:
                        print("[FALLBACK] Iniciando retorno à posição inicial")
                        self.fallback_active = True
                        self.fallback_steps = 0
                        self.fallback_init_step = self.step_count
                        self.tracking_mode = False
            else:
                self._pending_arm_target = None

            #print(f"[CONTROL] Aplicando novo target do braço {self._pending_arm_target.cpu().numpy()}")

        except Exception as e:
            print(f"[CONTROL ERROR] {e}")
            self._pending_arm_target = None

    def _prepare_fallback_target(self):
        """
        Calcula target de fallback (6 DOF).
        """
        current_arm_pos = self.robot.data.joint_pos[0, self.controller.arm_joint_indices].clone().to(self.device)
        initial_config = self.controller.initial_arm_positions.to(self.device)

        # Interpolação
        self.fallback_target = self.controller.interpolate_arm_config(
            current_arm_pos, initial_config, alpha=0.4).to(self.device)

        self.fallback_steps = self.step_count - self.fallback_init_step

        # Verificar conclusão
        distance = torch.norm(current_arm_pos - initial_config).item()

        if distance < 0.05 or self.fallback_steps >= self.max_fallback_steps:
            print("[FALLBACK] ✓ Retorno completo - reativando tracking")
            self.fallback_active = False
            self.fallback_steps = 0
            self.fallback_target = None
            self.consecutive_failures = 0
            self._reactivate_tracking_next_frame = True

        if hasattr(self, '_reactivate_tracking_next_frame') and self._reactivate_tracking_next_frame:
            self.tracking_mode = True
            self._reactivate_tracking_next_frame = False
            self.controller.ee_cur_pos, self.controller.ee_cur_quat = self.controller.get_current_ee_pose()

    def _get_observations(self):
        # CORREÇÃO: Retorna 6 DOF
        return {"policy": {"command": self.robot.data.joint_pos[0, self.controller.arm_joint_indices].clone()}}

    def _reset_idx(self, env_ids):
        if env_ids is None: env_ids = torch.arange(self.num_envs, device=self.device)
        if not self.first_reset_done:
            self.first_reset_done = True
            return
        super()._reset_idx(env_ids)
        self.step_count = 0
        self.current_command = "search"
        self.tracking_mode = False
        self.fallback_active = False
        self.consecutive_failures = 0
        self.last_control_step = 0
        self._pending_arm_target = None
        self.fallback_target = None

    def close(self):
        if self.vlm:
            self.vlm.stop_async_loop()
        super().close()
