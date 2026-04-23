"""
Controlador unificado do Spot - Base + Braço
ATUALIZADO: Movimento relativo ao EE + Atitude cumulativa
"""
import torch
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms, quat_from_euler_xyz, quat_mul, quat_rotate, axis_angle_from_quat
from ..configs import global_config as config


class RobotController:
    """
    Controlador unificado que gerencia base E braço.
    NOVO: Suporte a movimento relativo ao frame do EE
    """
    
    def __init__(self, robot, arm_joint_indices, leg_joint_indices, device="cuda:0"):
        self.robot = robot
        self.device = device
        
        # Separar gripper dos índices do braço
        if len(arm_joint_indices) == 7:
            self.arm_joint_indices = arm_joint_indices[:6]
            self.gripper_index = arm_joint_indices[-1]
            print(f"[ROBOT_CTRL] Separado: 6 DOF braço + 1 gripper")
        else:
            self.arm_joint_indices = arm_joint_indices
            self.gripper_index = None
            print(f"[ROBOT_CTRL] Usando {len(arm_joint_indices)} DOF")
        
        self.leg_joint_indices = leg_joint_indices
        
        # Estado inicial (referência)
        self.standing_leg_positions = robot.data.joint_pos[0, leg_joint_indices].clone().to(self.device)
        self.initial_arm_positions = robot.data.joint_pos[0, self.arm_joint_indices].clone().to(self.device)
        
        # Gripper
        self.gripper_open_position = -1.0
        
        # Posição do end-effector 
        self.ee_cur_pos, self.ee_cur_quat = self.get_current_ee_pose()
        
        # NOVO: Controle de atitude (para tracking opcional)
        self.last_applied_quat = None  # Salvar último quaternion aplicado
        
        # NOVO: Sistema de trajetória em linha reta
        self.target_line_pos = None  # Posição alvo para movimento em linha reta
        self.movement_axis = None     # Eixo de movimento atual ('x', 'y', 'z' no mundo)
        
        # Limites do braço (6 DOF)
        self.arm_joint_limits = {
            0: (-2.618, 3.142),   # sh0
            1: (-3.142, 0.524),   # sh1
            2: (0.0, 3.142),      # el0
            3: (-2.793, 2.793),   # el1
            4: (-1.833, 1.833),   # wr0
            5: (-2.880, 2.880),   # wr1
        }
        
        # Limites de workspace (frame da base)
        self.workspace_limits = {
            'x': (0.25, 0.75),
            'y': (-0.35, 0.35),
            'z': (0.15, 0.95),
        }
        
        # Ganhos PD para pernas
        self.kp_legs = config.BASE_KP
        self.kd_legs = config.BASE_KD
        
        # NOVO: Tracking de status IK
        self.last_ik_success = True
        self.ik_failure_count = 0
        
        print(f"[ROBOT_CTRL] Controlador unificado inicializado (6 DOF + movimento relativo)")
    
    def get_current_ee_pose(self):
        """Obtém pose atual do end-effector NO MUNDO."""
        kin_state = self.robot.root_physx_view.get_link_transforms()
        body_names = self.robot.body_names
        
        if "arm_link_wr1" in body_names:
            ee_idx = body_names.index("arm_link_wr1")
        else:
            print("[WARN] arm_link_wr1 não encontrado!")
            return torch.tensor([0.5, 0., 0.3], device=self.device), \
                   torch.tensor([1., 0., 0., 0.], device=self.device)
        
        ee_pos_w = kin_state[0, ee_idx, :3].clone()
        ee_quat_w = kin_state[0, ee_idx, 3:].clone()
        
        return ee_pos_w, ee_quat_w
    
    def get_base_pose(self):
        """Obtém pose da base do robô."""
        kin_state = self.robot.root_physx_view.get_link_transforms()
        body_names = self.robot.body_names
        
        base_idx = body_names.index("body") if "body" in body_names else 0
        base_pos = kin_state[0, base_idx, :3].clone()
        base_quat = kin_state[0, base_idx, 3:].clone()
        
        return base_pos, base_quat
    
    def get_ee_to_base_distance(self):
        """Calcula distância euclidiana do EE até a base."""
        ee_pos, _ = self.get_current_ee_pose()
        base_pos, _ = self.get_base_pose()
        distance = torch.norm(ee_pos - base_pos).item()
        return distance

    def compute_target_pose_relative_to_ee(self, current_pos, current_quat, pos_command, att_command, 
                                           pos_step=0.01, rot_step_deg=5.0):
        """
        NOVO: Calcula pose alvo com movimentos RELATIVOS ao frame do EE.
        
        IMPORTANTE: Compensa a rotação da câmera em relação ao link do EE.
        
        Movimentação:
        - "move_closer" = +X no frame da CÂMERA (para frente na imagem)
        - "move_away" = -X no frame da CÂMERA (para trás na imagem)
        - "adjust_right" = +Y no frame da CÂMERA (direita na imagem)
        - "adjust_left" = -Y no frame da CÂMERA (esquerda na imagem)
        - "adjust_up" = +Z no frame da CÂMERA (cima na imagem)
        - "adjust_down" = -Z no frame da CÂMERA (baixo na imagem)
        
        Atitude (incremental sobre orientação atual):
        - "look_up" / "look_down" → Pitch
        - "look_left" / "look_right" → Yaw
        """
        
        # Garantir device correto
        current_pos = current_pos.to(self.device)
        current_quat = current_quat.to(self.device)
        
        # COMPENSAÇÃO DA ROTAÇÃO DA CÂMERA
        # A câmera está rotacionada: rot=(pi/2, pi, pi/2) em relação ao link wr1
        # Isso significa: Roll=90°, Pitch=180°, Yaw=90°
        # 
        # Transformação dos comandos da câmera para o frame do EE:
        # - Câmera +X (forward) → EE -Z (devido ao pitch=180° e yaw=90°)
        # - Câmera +Y (right) → EE -X
        # - Câmera +Z (up) → EE +Y
        
        # 1. MOVIMENTO RELATIVO À CÂMERA (Local Frame da câmera)
        camera_displacement = torch.zeros(3, device=self.device)
        
        if pos_command == "move_closer":
            camera_displacement[0] = pos_step  # +X câmera
        elif pos_command == "move_away":
            camera_displacement[0] = -pos_step  # -X câmera
        elif pos_command == "adjust_right":
            camera_displacement[1] = pos_step  # +Y câmera
        elif pos_command == "adjust_left":
            camera_displacement[1] = -pos_step  # -Y câmera
        elif pos_command == "adjust_up":
            camera_displacement[2] = pos_step  # +Z câmera
        elif pos_command == "adjust_down":
            camera_displacement[2] = -pos_step  # -Z câmera
        
        # Transformar do frame da câmera para o frame do EE
        # Rotação inversa: camera_to_ee
        import numpy as np
        
        # Rotação da câmera em relação ao EE: (pi/2, pi, pi/2)
        camera_rot_quat = quat_from_euler_xyz(
            torch.tensor([np.pi/2], device=self.device),
            torch.tensor([np.pi], device=self.device),
            torch.tensor([np.pi/2], device=self.device)
        ).squeeze(0)
        
        # Quaternion inverso para transformar câmera → EE
        from isaaclab.utils.math import quat_inv
        ee_to_camera_quat = quat_inv(camera_rot_quat.unsqueeze(0)).squeeze(0)
        
        # Aplicar rotação inversa ao deslocamento
        from isaaclab.utils.math import quat_apply
        ee_displacement = quat_apply(ee_to_camera_quat.unsqueeze(0), camera_displacement.unsqueeze(0)).squeeze(0)
        
        # Converter deslocamento do frame do EE para mundo
        if current_quat.dim() == 1:
            current_quat_batch = current_quat.unsqueeze(0)
        else:
            current_quat_batch = current_quat
            
        world_displacement = quat_apply(current_quat_batch, ee_displacement.unsqueeze(0)).squeeze(0)
        target_pos = current_pos + world_displacement
        
        # 2. ATITUDE INCREMENTAL (aplicada sobre orientação atual)
        rot_step = rot_step_deg * (3.14159 / 180.0)
        roll, pitch, yaw = 0.0, 0.0, 0.0
        
        if att_command == "look_up":
            pitch = -rot_step  # Pitch negativo = olhar para cima
        elif att_command == "look_down":
            pitch = rot_step   # Pitch positivo = olhar para baixo
        elif att_command == "look_left":
            yaw = rot_step     # Yaw positivo = virar esquerda
        elif att_command == "look_right":
            yaw = -rot_step    # Yaw negativo = virar direita
        elif att_command == "reset_attitude":
            # Voltar para orientação padrão (identidade ou inicial)
            target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            return target_pos, target_quat
        
        # Criar quaternion de rotação incremental (Local Frame)
        if att_command != "look_at_center":
            rot_quat = quat_from_euler_xyz(
                torch.tensor([roll], device=self.device), 
                torch.tensor([pitch], device=self.device), 
                torch.tensor([yaw], device=self.device)
            )
            
            # Garantir shapes compatíveis
            if current_quat.dim() == 1:
                current_quat_batch = current_quat.unsqueeze(0)
            else:
                current_quat_batch = current_quat
                
            if rot_quat.dim() == 1:
                rot_quat = rot_quat.unsqueeze(0)
            
            # Aplicar rotação: target = current * incremental
            target_quat = quat_mul(current_quat_batch, rot_quat).squeeze(0)
            
            # VALIDAÇÃO: Verificar se rotação total não é muito extrema
            from isaaclab.utils.math import quat_error_magnitude
            
            neutral_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            
            if target_quat.dim() == 1:
                error = quat_error_magnitude(target_quat.unsqueeze(0), neutral_quat.unsqueeze(0))
            else:
                error = quat_error_magnitude(target_quat, neutral_quat.unsqueeze(0))
            
            max_rotation = 1.57  # ~90 graus
            
            if error.item() > max_rotation:
                target_quat = current_quat.clone()
        else:
            target_quat = current_quat.clone()
        
        return target_pos, target_quat
    
    def compute_target_pose_from_command(self, current_pos, current_quat, command, step=0.01):
        """Mantido para compatibilidade (modo legado - movimento absoluto)."""
        target_pos = current_pos.clone()
        
        if command == "move_closer":
            target_pos[0] += step
        elif command == "move_away":
            target_pos[0] -= step
        elif command == "adjust_left":
            target_pos[1] += step
        elif command == "adjust_right":
            target_pos[1] -= step
        elif command == "adjust_up":
            target_pos[2] += step
        elif command == "adjust_down":
            target_pos[2] -= step
        
        target_quat = current_quat.clone()

        if command not in ["adjust_up", "adjust_down"]:
            target_pos[2] = self.ee_cur_pos[2]
        else:
            self.ee_cur_pos[2] = target_pos[2]
        
        if command not in ["adjust_left", "adjust_right"]:
            target_pos[1] = self.ee_cur_pos[1]
        else:
            self.ee_cur_pos[1] = target_pos[1]

        if command not in ["move_closer", "move_away"]:
            target_pos[0] = self.ee_cur_pos[0]
        else:
            self.ee_cur_pos[0] = target_pos[0]
        
        return target_pos, target_quat

    def compute_target_pose_with_attitude(self, current_pos, current_quat, pos_command, att_command, 
                                         pos_step=0.01, rot_step_deg=5.0):
        """MODO LEGADO: Movimento absoluto + atitude incremental."""
        target_pos, _ = self.compute_target_pose_from_command(current_pos, current_quat, pos_command, step=pos_step)
        
        rot_step = rot_step_deg * (3.14159 / 180.0)
        roll, pitch, yaw = 0.0, 0.0, 0.0
        
        if att_command == "look_up":
            pitch = -rot_step 
        elif att_command == "look_down":
            pitch = rot_step
        elif att_command == "look_left":
            yaw = rot_step
        elif att_command == "look_right":
            yaw = -rot_step
        
        rot_quat = quat_from_euler_xyz(
            torch.tensor([roll], device=self.device), 
            torch.tensor([pitch], device=self.device), 
            torch.tensor([yaw], device=self.device)
        )
        
        if current_quat.dim() == 1:
            current_quat = current_quat.unsqueeze(0)
        
        current_quat = current_quat.to(self.device)

        if rot_quat.dim() == 1:
            rot_quat = rot_quat.unsqueeze(0)
            
        rot_quat = rot_quat.to(self.device)

        target_quat = quat_mul(current_quat, rot_quat)
        
        return target_pos, target_quat.squeeze(0)
    
    def apply_joint_targets(self, arm_positions, lock_base=True, gripper_scaling_factor=1.0):
        """ÚNICA função que seta targets de juntas."""
        if arm_positions is not None:
            arm_positions = arm_positions.to(self.device)

        full_target = self.robot.data.joint_pos.clone().to(self.device)
        
        # 1. Atualizar braço
        if arm_positions is not None:
            if arm_positions.dim() > 1:
                arm_positions = arm_positions.squeeze()
            
            if arm_positions.numel() != 6:
                print(f"[ROBOT_CTRL] ✗ Esperado 6 DOF, recebido {arm_positions.numel()}")
                return
            
            clamped_arm = arm_positions.clone()
            for i, (min_val, max_val) in self.arm_joint_limits.items():
                clamped_arm[i] = torch.clamp(clamped_arm[i], min_val, max_val)
            
            full_target[0, self.arm_joint_indices] = clamped_arm
        
        # 2. Gripper
        if self.gripper_index is not None:
            full_target[0, self.gripper_index] = self.gripper_open_position * gripper_scaling_factor
        
        # 3. Travar pernas
        if lock_base:
            full_target[0, self.leg_joint_indices] = self.standing_leg_positions
        
        self.robot.set_joint_position_target(full_target)
    
    def interpolate_arm_config(self, current, target, alpha=0.5):
        """Interpolação suave entre configurações."""
        current = current.to(self.device)
        target = target.to(self.device)

        if target.dim() > 1:
            target = target.squeeze()
        
        if current.numel() != 6 or target.numel() != 6:
            print(f"[ROBOT_CTRL] ✗ Interpolação requer 6 DOF")
            return current
        
        return current + alpha * (target - current)
    
    def reset_to_initial(self):
        """Retorna braço à posição inicial."""
        self.last_applied_quat = None
        self.target_line_pos = None
        self.movement_axis = None
        self.apply_joint_targets(self.initial_arm_positions, lock_base=True)
    
    def update_ik_status(self, success: bool):
        """Atualiza status do IK para enviar ao VLM."""
        self.last_ik_success = success
        if not success:
            self.ik_failure_count += 1
        else:
            self.ik_failure_count = 0
    
    def get_ik_status_string(self):
        """Retorna string de status do IK para o prompt."""
        if self.last_ik_success:
            return "IK: OK"
        else:
            return f"IK: FAILED ({self.ik_failure_count} consecutive)"


def setup_robot_controller(robot):
    """Função helper para inicializar o controlador."""
    arm_joint_names = [name for name in robot.data.joint_names if name.startswith("arm")]
    arm_joint_indices = [robot.data.joint_names.index(name) for name in arm_joint_names]
    
    leg_joint_names = [name for name in robot.data.joint_names 
                       if any(x in name for x in ['_hx', '_hy', '_kn'])]
    leg_joint_indices = [robot.data.joint_names.index(name) for name in leg_joint_names]
    
    controller = RobotController(robot, arm_joint_indices, leg_joint_indices)
    
    print(f"[SETUP] Braço: {len(controller.arm_joint_indices)} DOF + gripper")
    print(f"[SETUP] Pernas: {len(leg_joint_indices)} juntas")
    
    return controller