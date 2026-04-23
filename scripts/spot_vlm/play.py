"""
Script principal para rodar as tarefas de Tracking do Spot VLM.
Suporta:
1. arm-tracking: Apenas posição (XYZ)
2. arm-attitude-tracking: Posição + Atitude (6D Pose)

Atualizado com suporte a gravação de vídeo e Gym Wrappers.
"""
import argparse
#--------------------------------PATCH INSPECT--------------------------------#
import inspect

# Salva a função original
_old_getfile = inspect.getfile

def _safe_getfile(object):
    """Patch para evitar crash com IsaacLab/Torch Package"""
    try:
        return _old_getfile(object)
    except TypeError:
        # Se falhar dizendo que é built-in, retorna um nome falso seguro
        return "<isaaclab_module>"

# Aplica o patch
inspect.getfile = _safe_getfile
#-----------------------------END PATCH INSPECT------------------------------#

# Isaac Lab imports DEVEM vir ANTES
from isaaclab.app import AppLauncher

# 1. Configurar Argumentos (Antes de lançar o App)
parser = argparse.ArgumentParser(description="Spot VLM Task Runner")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument(
    "--task",
    type=str,
    default="arm-attitude-tracking",
    choices=["arm-tracking", "arm-attitude-tracking"],
    help="Task name: 'arm-tracking' (Position only) or 'arm-attitude-tracking' (Pos + Orientation)."
)
parser.add_argument("--video", action="store_true", default=False, help="Record video of the episode.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")

# Adiciona argumentos padrão do Isaac Lab (headless, livestreams, etc)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Forçar enable_cameras para True, pois o environment precisa de câmeras
args_cli.enable_cameras = True

# 2. Iniciar a Simulação (Deve ser feito antes de importar torch/isaaclab)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- FORCE LOAD EXTENSIONS (FIX) ---
# Use the Extension Manager directly to avoid ModuleNotFoundError
import omni.kit.app
manager = omni.kit.app.get_app().get_extension_manager()

# 1. Enable Core Nodes (Fixes: unrecognized type 'IsaacReadSimulationTime')
manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

# 2. Enable ROS 2 Bridge (Fixes: ROS topics not appearing)
manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
# -----------------------------------

# 3. Imports do Sistema (Pós-Launch)
import torch
import traceback
import gymnasium as gym
from datetime import datetime
import os
import sys
from pathlib import Path

# --- CORREÇÃO DE PATH ---
# Adiciona o diretório raiz do projeto ao sys.path para permitir imports de 'scripts.spot_vlm'
# Assume que play.py está em .../scripts/spot_vlm/play.py
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2] # Sobe 2 níveis: spot_vlm -> scripts -> raiz
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
# ------------------------

from scripts.spot_vlm.configs import global_config as config

# Imports dos Environments
# Nota: Importamos dentro de try/except para facilitar debug de caminhos
try:
    from scripts.spot_vlm.envs.arm_tracking_env import SpotArmTrackingEnv, SpotArmTrackingEnvCfg
    from scripts.spot_vlm.envs.attitude_tracking_env import SpotAttitudeTrackingEnv
except ImportError as e:
    print(f"\n[ERRO IMPORT] Não foi possível importar os environments.")
    print(f"Certifique-se de que 'scripts.spot_vlm' está no PYTHONPATH.")
    print(f"Erro original: {e}\n")
    # Tentar import relativo como fallback caso scripts.spot_vlm falhe
    try:
        print("[INFO] Tentando import relativo...")
        from envs.arm_tracking_env import SpotArmTrackingEnv, SpotArmTrackingEnvCfg
        from envs.attitude_tracking_env import SpotAttitudeTrackingEnv
        print("[INFO] Import relativo funcionou!")
    except ImportError as e2:
        print(f"[ERRO FATAL] Falha também no import relativo: {e2}")
        raise e

def main():
    print(f"\n[INFO] Inicializando tarefa: {args_cli.task.upper()}")
    print(f"[INFO] VLM Enabled: {config.VLM_ENABLED}")

    # Seleção de Ambiente
    if args_cli.task == "arm-attitude-tracking":
        # Usa a configuração base (ArmTrackingCfg) mas instancia a classe nova
        env_cfg = SpotArmTrackingEnvCfg()
        env_cfg = config_env(env_cfg)
        env = SpotAttitudeTrackingEnv(cfg=env_cfg)
        print("[INFO] Ambiente carregado: SpotAttitudeTrackingEnv (6D Control)")

    elif args_cli.task == "arm-tracking":
        # Ambiente original
        env_cfg = SpotArmTrackingEnvCfg()
        env_cfg = config_env(env_cfg)
        env = SpotArmTrackingEnv(cfg=env_cfg)
        print("[INFO] Ambiente carregado: SpotArmTrackingEnv (Position Only)")

    else:
        raise ValueError(f"Tarefa desconhecida: {args_cli.task}")

    # --- Configuração de Gravação de Vídeo ---
    if args_cli.video:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_folder = os.path.join("videos", f"{args_cli.task}_{timestamp}")

        # Wrapper padrão do Gymnasium para gravação
        # Nota: IsaacLab environments geralmente precisam de wrapper específico ou render() customizado
        # Aqui usamos o RecordVideo padrão, assumindo que env.render() retorna array RGB
        print(f"[INFO] Gravando vídeo em: {video_folder}")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0, # Grava desde o início
            video_length=args_cli.video_length,
            name_prefix=f"spot_{args_cli.task}"
        )

    # Loop de Simulação
    try:
        obs, _ = env.reset()

        while simulation_app.is_running():
            # A ação enviada ao step aqui é 'dummy' pois o controle real
            # é calculado internamente pelo env baseado no VLM.
            # No entanto, mantemos o formato correto (num_envs, action_dim).

            # Action space é 6 (DOF) para ambos os envs agora
            actions = torch.zeros((env.num_envs, 6), device=env.device)

            # Step
            obs, rew, terminated, truncated, info = env.step(actions)

            # Reset se necessário (geralmente tratado internamente pelo ManagerBasedEnv, mas bom ter aqui)
            if terminated or truncated:
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupção pelo usuário.")
    except Exception as e:
        print(f"\n[ERRO FATAL] {e}")
        traceback.print_exc()
    finally:
        # Limpeza
        print("[INFO] Fechando ambiente...")
        env.close()
        simulation_app.close()

def config_env(env_cfg_):
    # Configurar ambiente
    env_cfg_.scene.num_envs = args_cli.num_envs
    env_cfg_.sim.device = args_cli.device
    return env_cfg_

if __name__ == "__main__":
    main()
