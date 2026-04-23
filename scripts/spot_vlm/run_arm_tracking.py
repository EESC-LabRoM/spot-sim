"""
Spot Arm VLM Control - Tracking com Profundidade
Script principal refatorado - ORDEM DE IMPORTS CORRIGIDA

NOTA SOBRE WARNINGS:
- "Warp CUDA error: Failed to get driver entry point 'cuDeviceGetUuid'"
  → Este é um WARNING do Isaac Sim, NÃO um erro. O simulador funciona normalmente.
  → Causado por incompatibilidade de versão CUDA (Isaac Sim usa CUDA 11.8, driver é mais novo)
  → Pode ser ignorado com segurança.

- "Modules: ['omni.kit_app'] were loaded before SimulationApp"
  → Também pode ser ignorado - é do Isaac Sim internamente, não afeta funcionamento.
"""
import argparse
import sys
from pathlib import Path

# ============================================================
# CRÍTICO: Isaac Lab imports DEVEM vir ANTES de qualquer coisa
# ============================================================
from isaaclab.app import AppLauncher

# 1. Configuração de argumentos
parser = argparse.ArgumentParser(description="Spot Arm VLM Control")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# 2. Criar AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================
# Imports que dependem do Isaac Sim
# ============================================================
import torch
import time

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveScene

# ============================================================
# Imports do projeto (FAZER APÓS ISAAC LAB)
# ============================================================
# Adicionar diretório do projeto ao path se necessário
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Importar configurações
try:
    from scripts.spot_vlm_deprecated.configs import global_config as config
except ImportError:
    print("[WARN] config.py não encontrado, usando valores padrão")
    class config:
        VLM_MODEL_PATH = "./models/Qwen2-VL-2B-Instruct"
        VLM_INFERENCE_INTERVAL = 50
        VLM_ENABLED = True
        USE_CUROBO = False
        RENDER_EVERY_N_STEPS = 2
        USE_EXTENDED_SCENE = True
        APPLY_SPOT_COLORS = True
        TARGET_DISTANCE = 0.10
        SAVE_IMAGES = True

try:
    # Imports que NÃO dependem de Isaac Lab (seguros)
    from scripts.spot_vlm_deprecated.vision.depth_processing import process_depth_map, calculate_camera_adjustments

    # Imports que DEPENDEM de Isaac Lab (fazer por último)
    from scripts.spot_vlm_deprecated.configs.scene_config import SpotManipulationWithObjectsSceneCfg
    from scripts.spot_vlm_deprecated.controllers.arm_controller import ArmController
    from scripts.spot_vlm_deprecated.controllers.base_controller import BaseController, setup_leg_lock
    from scripts.spot_vlm_deprecated.utils.collision_utils import create_primitive_arm_colliders
    from scripts.spot_vlm_deprecated.utils.robot_initialization import initialize_robot_sequence, get_arm_joint_info
except ImportError as e:
    print(f"[ERRO] Não foi possível importar módulos do projeto: {e}")
    print(f"[INFO] Certifique-se de estar executando de: {project_root}")
    print(f"[INFO] Estrutura esperada: scripts/spot_vlm/configs/, controllers/, etc.")
    import traceback
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)

# VLM inference (importar separado para evitar carregar transformers cedo)
VLMInference = None
try:
    from scripts.spot_vlm_deprecated.vision.vlm_inference import VLMInference
except ImportError as e:
    print(f"[WARN] VLM não disponível: {e}")

# Curobo (opcional)
CuroboMotionPlanner = None
CUROBO_AVAILABLE = False
try:
    from scripts.spot_vlm_deprecated.controllers.motion_planner import CuroboMotionPlanner
    CUROBO_AVAILABLE = True
except (ImportError, Exception, RuntimeError) as e:
    print(f"[WARN] Curobo não disponível. Usando controle direto. Erros: {e}")


def main():
    # ==================== SETUP ====================

    # 1. Carregar VLM
    print(f"[INFO] Inicializando VLM de: {config.VLM_MODEL_PATH}")

    vlm = None
    if config.VLM_ENABLED and VLMInference is not None:
        vlm = VLMInference(config.VLM_MODEL_PATH)
    else:
        print("[WARN] VLM não disponível - continuando sem inferência")

    # 2. Inicializar simulação
    print("[INFO] Inicializando simulação...")
    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=0.01,
            device="cuda:0",
            physx=sim_utils.PhysxCfg(
                bounce_threshold_velocity=0.2,
                gpu_max_rigid_contact_count=2**20,
                enable_stabilization=True,
                solver_type=1,
                gpu_found_lost_pairs_capacity=2**20,
                gpu_total_aggregate_pairs_capacity=2**20,
                friction_offset_threshold=0.01,
                friction_correlation_distance=0.0005,
            )
        )
    )

    # 3. Criar cena COM TODOS OS OBJETOS
    print("[INFO] Criando cena...")
    if config.USE_EXTENDED_SCENE:
        scene_cfg = SpotManipulationWithObjectsSceneCfg(num_envs=1, env_spacing=2.0)
        print("[INFO] Usando cena estendida (cubo vermelho + azul + honeycomb)")
    else:
        from scripts.spot_vlm_deprecated.configs.scene_config import SpotManipulationSceneCfg
        scene_cfg = SpotManipulationSceneCfg(num_envs=1, env_spacing=2.0)
        print("[INFO] Usando cena básica (apenas cubo vermelho)")

    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # 4. Configurar colisões
    #print("[INFO] Configurando colisões...")
    #create_primitive_arm_colliders(scene, sim)

    # 5. Inicializar robô
    print("[INFO] Inicializando robô...")
    robot = scene["robot"]
    arm_joint_indices, arm_joint_names, _ = get_arm_joint_info(robot)

    print(f"[INFO] Robô: {robot.num_bodies} corpos, {robot.num_joints} juntas")
    print(f"[INFO] Juntas do braço: {arm_joint_names}")

    # 6. Aplicar cores do Spot
    if config.APPLY_SPOT_COLORS:
        try:
            from scripts.spot_vlm_deprecated.utils.visualization import apply_spot_colors
            apply_spot_colors(sim)
            print("[INFO] Braço do Spot colorido! Ele está lindo 😎")
        except Exception as e:
            print(f"[INFO] Cores não aplicadas: {e}")

    # 7. Inicialização suave
    print("[INFO] Executando inicialização suave...")
    initialize_robot_sequence(robot, scene, sim, arm_joint_indices)

    # 8. Configurar controladores
    print("[INFO] Configurando controladores...")
    arm_controller = ArmController(robot, arm_joint_indices, device="cuda:0")

    leg_joint_indices, standing_positions = setup_leg_lock(robot)
    base_controller = BaseController(robot, leg_joint_indices, standing_positions, device="cuda:0")

    # 9. Inicializar Curobo (opcional)
    motion_planner = None

    if CUROBO_AVAILABLE and config.USE_CUROBO:
        try:
            print("[INFO] Tentando inicializar Curobo...")
            motion_planner = CuroboMotionPlanner(robot)
            print("[INFO] ✓ Curobo inicializado!")
        except Exception as e:
            print(f"[WARN] Curobo falhou: {e}")
            print("[INFO] Continuando com controle direto...")
    else:
        if not config.USE_CUROBO:
            print("[INFO] Curobo desabilitado no global_config.py - usando controle direto")
        else:
            print("[INFO] Curobo não disponível - usando controle direto")

    # ==================== LOOP PRINCIPAL ====================

    step_count = 0
    current_command = "observe"
    last_inference_step = 0
    inference_interval = config.VLM_INFERENCE_INTERVAL
    consecutive_failures = 0

    # Performance monitoring
    last_fps_time = time.time()
    fps_counter = 0

    print("\n" + "="*70)
    print("🤖 INICIANDO LOOP DE CONTROLE DO BRAÇO".center(70))
    print("="*70 + "\n")

    try:
        while simulation_app.is_running():

            # FPS monitoring
            fps_counter += 1
            if fps_counter % 100 == 0:
                current_time = time.time()
                fps = 100 / (current_time - last_fps_time)
                last_fps_time = current_time

                current_el0 = robot.data.joint_pos[0, arm_joint_indices[2]].item() if len(arm_joint_indices) > 2 else 0.0
                print(f"\r[PERF] FPS: {fps:.1f} | Steps: {step_count} | Cmd: {current_command:15s} | Elbow: {current_el0:.2f}rad", end="")

            # Travar base
            base_controller.lock_base()

            # ==================== VLM INFERENCE ====================

            if vlm and vlm.available and (step_count - last_inference_step) >= inference_interval:
                print("\n--- [VLM ARM CONTROL] ---")
                last_inference_step = step_count

                try:
                    # Capturar imagens
                    rgb_data = scene["camera"].data.output["rgb"][0]
                    depth_data = scene["camera"].data.output["distance_to_image_plane"][0]

                    if rgb_data is not None and depth_data is not None:
                        # Processar RGB
                        rgb_pil = vlm.process_image(rgb_data[:, :, :3])

                        # Processar Depth
                        depth_colored = process_depth_map(depth_data)
                        from PIL import Image
                        depth_pil = Image.fromarray(depth_colored).resize((336, 336), Image.Resampling.LANCZOS)

                        # Calcular ajustes
                        depth_np = depth_data.cpu().numpy()
                        if len(depth_np.shape) == 3:
                            depth_np = depth_np[:, :, 0]

                        adjustments = calculate_camera_adjustments(depth_np, target_distance=config.TARGET_DISTANCE)

                        # Salvar para debug
                        if config.SAVE_IMAGES and step_count % config.SAVE_IMAGES_EVERY_N_STEPS == 0:
                            rgb_pil.save("arm_vision_rgb.jpg")
                            depth_pil.save("arm_vision_depth.jpg")

                        # Inferência VLM
                        response = vlm.infer_arm_tracking(rgb_pil, depth_pil, adjustments['distance'])

                        if response and "command" in response:
                            new_command = response["command"].lower().strip()

                            valid_commands = ["move_closer", "move_away", "adjust_left", "adjust_right",
                                            "adjust_up", "adjust_down", "hold_position", "observe"]

                            if new_command in valid_commands:
                                current_command = new_command
                                consecutive_failures = 0
                                print(f"[VLM] ✓ Comando: {current_command.upper()}")
                                if "reason" in response:
                                    print(f"[VLM] Razão: {response['reason'][:100]}")
                            else:
                                consecutive_failures += 1
                        else:
                            consecutive_failures += 1

                        # Fallback para controle baseado em regras
                        if consecutive_failures >= 3:
                            print(f"[VLM] ⚠ Muitas falhas, usando controle baseado em regras...")
                            if adjustments['distance'] > 0.15:
                                current_command = "move_closer"
                            elif adjustments['distance'] < 0.08:
                                current_command = "move_away"
                            elif not adjustments['is_centered']:
                                current_command = "adjust_right"
                            else:
                                current_command = "hold_position"
                            consecutive_failures = 0

                except Exception as e:
                    print(f"[ERRO VLM] {e}")
                    import traceback
                    traceback.print_exc()

            # ==================== ARM CONTROL ====================

            current_arm_pos = robot.data.joint_pos[0, arm_joint_indices].clone()

            # Tentar usar Curobo se disponível
            if motion_planner and current_command not in ["hold_position", "observe"]:
                try:
                    current_ee_pos, current_ee_quat = arm_controller.get_current_ee_pose()
                    target_ee_pos, target_ee_quat = arm_controller.compute_target_pose_from_command(
                        current_ee_pos, current_ee_quat, current_command, step=0.01
                    )

                    target_config = motion_planner.plan_to_pose(
                        target_ee_pos, target_ee_quat, current_arm_pos
                    )

                    if target_config is not None:
                        #print(f"[Curobo] Plano bem-sucedido, aplicando posições...")
                        arm_controller.apply_joint_positions(target_config)
                    else:
                        # Fallback
                        print("[ERRO Curobo] ⚠ Plano falhou...")
                        #arm_controller.apply_direct_control(current_command)
                except Exception as e:
                    print(f"[ERRO Curobo] {e}...")
                    #arm_controller.apply_direct_control(current_command)
            else:
                # Controle direto
                if current_command not in ["hold_position", "observe"]:
                    arm_controller.apply_direct_control(current_command)

            #print(f"Estado das juntas das pernas: {robot.data.joint_pos[0, leg_joint_indices]}")


            # ==================== SIMULATION STEP ====================

            scene.write_data_to_sim()
            sim.step(render=(step_count % config.RENDER_EVERY_N_STEPS == 0))
            scene.update(dt=sim.get_physics_dt())

            step_count += 1

    except KeyboardInterrupt:
        print("\n[INFO] Encerrando por interrupção do usuário...")
    except Exception as e:
        print(f"\n[ERRO FATAL] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
        print("[INFO] Simulação encerrada.")


if __name__ == "__main__":
    main()
