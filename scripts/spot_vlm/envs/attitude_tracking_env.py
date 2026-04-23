"""
Spot Attitude Tracking Environment
ATUALIZADO: Movimento relativo ao EE + Status IK no prompt
"""
import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from scripts.spot_vlm.envs.arm_tracking_env import SpotArmTrackingEnv, SpotArmTrackingEnvCfg
from scripts.spot_vlm.configs import global_config as config
from scripts.spot_vlm.inference.prompts import get_attitude_tracking_prompt
from scripts.spot_vlm.inference.depth_processing import process_depth_map, calculate_camera_adjustments


class SpotAttitudeTrackingEnv(SpotArmTrackingEnv):
    """
    Environment com controle 6D (Posição + Orientação).
    NOVO: Movimento relativo ao frame do EE.
    """
    
    def __init__(self, cfg: SpotArmTrackingEnvCfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        
        self.current_pos_command = "hold_position"
        self.current_att_command = "look_at_center"
        
        # Tracking de métricas
        self.last_ee_distance = 0.0
        
        print("[ENV] Spot ATTITUDE Tracking Environment inicializado!")
        print("[ENV] Movimento RELATIVO ao frame do EE")

    def _submit_to_vlm(self):
        """Envia imagem para VLM com status IK e distância EE."""
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
            
            # NOVO: Passar informações extras no modo attitude
            if self.tracking_mode:
                # Obter status IK e distância
                ik_status = self.controller.get_ik_status_string()
                ee_distance = self.controller.get_ee_to_base_distance()
                self.last_ee_distance = ee_distance
                
                # Criar prompt com informações
                prompt_with_status = get_attitude_tracking_prompt(
                    current_distance=adjustments['distance'],
                    ik_status=ik_status,
                    ee_distance=ee_distance
                )
                
                # Submeter com prompt customizado (não usa o mode padrão)
                from PIL import Image as PILImage
                self.vlm.submit_async_request(
                    rgb_pil, depth_pil, adjustments['distance'], 
                    mode="attitude_tracking"
                )
            else:
                # Modo busca
                self.vlm.submit_async_request(
                    rgb_pil, depth_pil, adjustments['distance'], 
                    mode="search"
                )
                 
        except Exception as e:
            print(f"[ENV SUBMIT ERROR] {e}")

    def _process_vlm_result(self, response):
        """Processa JSON com pos_command E att_command."""
        if not response: return

        # Modo Busca (herdado)
        if not self.tracking_mode:
            super()._process_vlm_result(response)
            return
            
        # Modo Tracking
        if "pos_command" in response:
            self.current_pos_command = response["pos_command"].lower().strip()
        elif "command" in response:
            self.current_pos_command = response["command"].lower().strip()
            
        if "att_command" in response:
            self.current_att_command = response["att_command"].lower().strip()
        else:
            self.current_att_command = "look_at_center"
        
        print(f"[DECISION] POS: {self.current_pos_command.upper()} | ATT: {self.current_att_command.upper()}")
        
        if self.current_pos_command != "search_environment":
            self.consecutive_failures = 0

    def _prepare_arm_target(self):
        """
        Calcula próximo target usando MOVIMENTO RELATIVO ao EE.
        """
        try:
            if self.current_pos_command == "hold_position" and self.current_att_command == "look_at_center":
                return
            
            current_arm_pos = self.robot.data.joint_pos[0, self.controller.arm_joint_indices].clone()
            curr_ee_pos, curr_ee_quat = self.controller.get_current_ee_pose()

            # USAR NOVA FUNÇÃO DE MOVIMENTO RELATIVO
            tgt_pos, tgt_quat = self.controller.compute_target_pose_relative_to_ee(
                curr_ee_pos, 
                curr_ee_quat, 
                self.current_pos_command,
                self.current_att_command,
                pos_step=0.04,      # 4cm por comando
                rot_step_deg=5.0    # 5 graus por comando
            )
            
            # Planejamento IK
            if self.motion_planner is not None:
                target_config = self.motion_planner.plan_to_pose(tgt_pos, tgt_quat, current_arm_pos)
                
                if target_config is not None:
                    if target_config.numel() != 6:
                        print(f"[CONTROL] ✗ Planner retornou {target_config.numel()} valores (esperado 6)")
                        self.controller.update_ik_status(False)
                        return
                    
                    self._pending_arm_target = self.controller.interpolate_arm_config(
                        current_arm_pos, target_config, alpha=0.3)
                    
                    # Sucesso IK
                    self.consecutive_failures = 0
                    self.controller.update_ik_status(True)
                else:
                    self.consecutive_failures += 1
                    self.controller.update_ik_status(False)
                    print(f"[CONTROL] Falha IK {self.consecutive_failures}/{self.max_consecutive_failures}")
                    
                    # Fallback se muitas falhas
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        print("[FALLBACK] Limite de falhas atingido. Retornando à inicial.")
                        self.fallback_active = True
                        self.fallback_steps = 0
                        self.fallback_init_step = self.step_count
                        self.tracking_mode = False
                        self.current_pos_command = "hold_position"
                        self.current_att_command = "look_at_center"
                        self._pending_arm_target = None
            else:
                pass

        except Exception as e:
            print(f"[CONTROL ERROR] {e}")
            self.controller.update_ik_status(False)
            self._pending_arm_target = None
    
    def _reset_idx(self, env_ids):
        """Reseta environment."""
        super()._reset_idx(env_ids)
        
        # Resetar estado
        if hasattr(self, 'controller'):
            self.controller.last_applied_quat = None
            self.controller.target_line_pos = None
            self.controller.movement_axis = None
            self.controller.ik_failure_count = 0
            self.controller.last_ik_success = True
        
        self.current_pos_command = "hold_position"
        self.current_att_command = "look_at_center"
        self.last_ee_distance = 0.0
