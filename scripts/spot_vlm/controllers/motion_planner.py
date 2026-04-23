"""
Planejador de movimento usando Curobo IK Solver
CORREÇÃO: Retorna apenas 6 juntas (sem gripper)
"""
import torch
from pathlib import Path
import traceback
from ..configs.global_config import USE_IK

if USE_IK:
    print("[INFO] Modo: IK Solver")
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

else:
    print("[INFO] Modo: MotionGen")
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    from curobo.geom.sdf.world import CollisionCheckerType

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState as CuroboJointState
CUROBO_AVAILABLE = True
print("[INFO] Curobo disponível.")

class CuroboMotionPlanner:
    """
    Planejador baseado em exemplos oficiais do Isaac Lab
    CORREÇÃO: Retorna apenas 6 DOF (sem gripper)
    """
    
    def __init__(self, robot, device="cuda:0"):
        if not CUROBO_AVAILABLE:
            raise ImportError("Curobo não disponível!")
        
        self.robot = robot
        self.device = device
        self.use_ik = USE_IK
        
        self.ik_solver = None
        self.motion_gen = None
        
        if self.use_ik:
            self._initialize_ik_solver()
        else:
            self._initialize_motion_gen()
    
    def _load_robot_config(self, with_collision=False):
        """Carrega configuração do robô COM 6 DOF"""
        import yaml
        from ..configs.robot_config import arm_prefix
        
        urdf_candidates = [
            Path("assets/spot/spot_with_arm.urdf").resolve(),
            Path("external/relic/source/relic/relic/assets/spot/spot_with_arm.urdf").resolve(),
        ]
        
        urdf_path = None
        for candidate in urdf_candidates:
            if candidate.exists():
                urdf_path = candidate
                break
        
        if urdf_path is None:
            print("[ERRO] URDF não encontrado!")
            return None
        
        try:
            ee_link_name = f"{arm_prefix}_link_wr1"
            
            robot_config_dict = {
                "kinematics": {
                    "urdf_path": str(urdf_path),
                    "base_link": "body",
                    "ee_link": ee_link_name,  # Usa prefixo correto
                    # NÃO travar gripper aqui - será controlado separadamente
                }
            }
            
            print(f"[INFO] Curobo: EE={ee_link_name} (6 DOF do braço)")
            
            if with_collision:
                spheres_path = Path("assets/spot/spot_spheres.yml").resolve()
                
                # CORREÇÃO: Usar prefixo correto nos nomes dos links
                collision_links = [
                    f"{arm_prefix}_link_sh0", f"{arm_prefix}_link_sh1", 
                    f"{arm_prefix}_link_el0", f"{arm_prefix}_link_el1", 
                    f"{arm_prefix}_link_wr0", f"{arm_prefix}_link_wr1",
                ]
                
                robot_config_dict["kinematics"].update({
                    "collision_link_names": collision_links,
                    "collision_sphere_buffer": 0.002,
                    "self_collision_buffer": {},
                    "self_collision_ignore": {
                        f"{arm_prefix}_link_sh0": [f"{arm_prefix}_link_sh1"],
                        f"{arm_prefix}_link_sh1": [f"{arm_prefix}_link_el0"],
                        f"{arm_prefix}_link_el0": [f"{arm_prefix}_link_el1"],
                        f"{arm_prefix}_link_el1": [f"{arm_prefix}_link_wr0"],
                        f"{arm_prefix}_link_wr0": [f"{arm_prefix}_link_wr1"],
                    }
                })
                
                if spheres_path.exists():
                    with open(spheres_path, 'r') as f:
                        spheres_data = yaml.safe_load(f)
                    robot_config_dict["kinematics"]["collision_spheres"] = spheres_data.get("collision_spheres", {})
            
            tensor_args = TensorDeviceType(device=self.device, dtype=torch.float32)
            robot_cfg = RobotConfig.from_dict(robot_config_dict, tensor_args)
            
            print(f"[INFO] ✓ Config Curobo carregada (6 DOF - sem gripper)")
            return robot_cfg
            
        except Exception as e:
            print(f"[ERRO] Falha ao carregar config: {e}")
            traceback.print_exc()
            return None
    
    def _initialize_ik_solver(self):
        """Inicializa IK Solver com 6 DOF"""
        print("[INFO] Inicializando IK Solver (6 DOF)...")
        
        try:
            robot_cfg = self._load_robot_config(with_collision=False)
            if robot_cfg is None:
                return
            
            tensor_args = TensorDeviceType(device=torch.device(self.device), dtype=torch.float32)
            
            ik_config = IKSolverConfig.load_from_robot_config(
                robot_cfg=robot_cfg,
                world_model=None,
                tensor_args=tensor_args,
                rotation_threshold=0.314159,  # ~18 graus
                position_threshold=0.03,
                num_seeds=20,
                self_collision_check=True,
                self_collision_opt=True,
                use_cuda_graph=True,
                collision_checker_type=None,
            )
            
            self.ik_solver = IKSolver(ik_config)
            print("[INFO] ✓ IK Solver pronto (6 DOF)")
            
        except Exception as e:
            print(f"[ERRO] Falha IK Solver: {e}")
            traceback.print_exc()
    
    def _initialize_motion_gen(self):
        """Inicializa MotionGen (não usado atualmente)"""
        print("[INFO] MotionGen não implementado - use IK Solver")
    
    def plan_to_pose(self, target_pos, target_quat, current_joint_pos):
        """
        Planeja movimento para pose alvo.
        
        Args:
            target_pos: [3] posição NO MUNDO
            target_quat: [4] quaternion NO MUNDO (w,x,y,z)
            current_joint_pos: [6] configuração atual (SEM gripper)
        
        Returns:
            [6] configuração alvo ou None (SEM gripper)
        """
        if not self.use_ik or self.ik_solver is None:
            return None
        
        try:
            # Passa para o dispositivo correto
            target_pos = target_pos.to(self.device)
            target_quat = target_quat.to(self.device)
            current_joint_pos = current_joint_pos.to(self.device)

            # Validar entrada
            if torch.any(torch.isnan(target_pos)) or torch.any(torch.isnan(target_quat)):
                print("[CUROBO] ✗ NaN detectado")
                return None
            
            # Converter para frame da base
            kin_state = self.robot.root_physx_view.get_link_transforms()
            body_names = self.robot.body_names
            base_idx = body_names.index("body") if "body" in body_names else 0
            base_pos_w = kin_state[0, base_idx, :3].to(self.device)
            base_quat_w = kin_state[0, base_idx, 3:].to(self.device)
            
            from isaaclab.utils.math import subtract_frame_transforms
            
            rel_pos, rel_quat = subtract_frame_transforms(
                target_pos, target_quat,
                base_pos_w, base_quat_w
                #,target_pos, target_quat
            )

            rel_pos[2] *= -1
            
            # Validar workspace
            distance = torch.norm(rel_pos)
            # if distance > 0.80:
            #     print(f"[CUROBO] ✗ Fora do alcance: {distance:.2f}m")
            #     return None
            
            #print(f"[DEBUG] Target rel: pos={rel_pos.cpu().numpy()} dist={distance:.2f}m")
            
            # Criar pose alvo (formato correto para batch)
            goal_pose = Pose(
                position=rel_pos.unsqueeze(0),
                quaternion=rel_quat.unsqueeze(0)
            )
            
            # CORREÇÃO: Validar que recebemos 6 DOF
            if current_joint_pos.numel() != 6:
                print(f"[CUROBO] ✗ Esperado 6 juntas, recebido {current_joint_pos.numel()}")
                return None
            
            # Formato seed: [batch=1, num_seeds, dof]
            q_seed = current_joint_pos.unsqueeze(0).unsqueeze(0)
            num_seeds = 3
            q_seed_batch = q_seed.repeat(1, num_seeds, 1)

            try:
                # Adicionar ruído gaussiano às seeds extras (mantém primeira igual)
                if num_seeds > 1:
                    noise = torch.randn(1, num_seeds - 1, 6, device=self.device) * 0.1
                    q_seed_batch[:, 1:, :] += noise
            except:
                pass
    

            result = self.ik_solver.solve_batch(
                goal_pose = goal_pose,
                seed_config = q_seed_batch,
                return_seeds = num_seeds)
            
            if result.success.any():
                success_mask = result.success#.squeeze()
                
                if success_mask.dim() == 0:
                    solution = result.js_solution.position.squeeze()
                else:
                    successful_solutions = result.js_solution.position[success_mask]
                    ref_pos = current_joint_pos.unsqueeze(0)
                    distances = torch.norm(successful_solutions - ref_pos, dim=1)
                    best_idx = torch.argmin(distances).item()
                    solution = successful_solutions[best_idx]
                
                pos_err = result.position_error.min().item() * 1000
                rot_err = result.rotation_error.min().item() * 57.3
                delta = torch.norm(solution - current_joint_pos).item()
                
                #print(f"[CUROBO] ✓ IK OK (err: {pos_err:.1f}mm/{rot_err:.1f}°, Δ: {delta:.2f}rad)")
                
                # CORREÇÃO: Garantir que retorna apenas 6 valores
                if solution.numel() > 6:
                    solution = solution[:6]
                
                return solution #torch.multiply(solution, torch.tensor([-1.0], device=solution.device))
            else:
                pos_err = result.position_error.min().item() * 1000
                rot_err = result.rotation_error.min().item() * 57.3
                print(f"[CUROBO] ✗ IK falhou (err: {pos_err:.1f}mm/{rot_err:.1f}°)")
                return None
                
        except Exception as e:
            print(f"[ERRO CUROBO] {e}")
            traceback.print_exc()
            return None
    
    def is_available(self):
        """Verifica disponibilidade"""
        if self.use_ik:
            return self.ik_solver is not None
        else:
            return self.motion_gen is not None