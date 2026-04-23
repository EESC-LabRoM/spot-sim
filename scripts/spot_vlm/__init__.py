"""
Sistema Spot VLM - Controle Visual do Boston Dynamics Spot

AVISO: Este pacote contém módulos que dependem do Isaac Lab.
Certifique-se de inicializar o AppLauncher ANTES de importar:

    from isaaclab.app import AppLauncher
    app = AppLauncher(args)
    simulation_app = app.app
    
    # Agora é seguro importar
    from scripts.spot_vlm.configs import SpotManipulationSceneCfg

Estrutura do pacote:
scripts/spot_vlm/
├── configs/
│   ├── __init__.py              ✓ Exportações organizadas
│   ├── global_config.py         ✓ Configurações globais + validação
│   ├── robot_config.py          ✓ Configuração do Spot
│   └── scene_config.py          ✓ Cenas (básica + com objetos)
│
├── controllers/
│   ├── __init__.py              ✓ Com import condicional do Curobo
│   ├── robot_controller.py      ✓ Classe RobotController + helpers -> Controle do braço + pernas
│   ├── arm_controller.py        x Deprecated - Classe ArmController completa
│   ├── base_controller.py       x Deprecated - Classe BaseController + helpers
│   └── motion_planner.py        ✓ Integração Curobo (opcional)
│
├── vision/
│   ├── __init__.py              ✓
│   ├── vlm_inference.py         ✓ Classe VLMInference robusta
│   └── depth_processing.py      ✓ Processamento + visualização
│
├── utils/
│   ├── __init__.py              ✓
│   ├── collision_utils.py       ✓ Colisores primitivos + mesh
│   ├── robot_initialization.py  ✓ Sequência de setup suave
│   └── visualization.py         ✓ Cores, debug spheres, trajetórias
│
├── tasks/                       ✓ Sistema de tasks (NOVO)
│   ├── __init__.py              ✓ Exportações de environments
│   └── arm_tracking_env.py      ✓ Environment de tracking com VLM
│
├── __init__.py                  ✓ Versão + autor (este arquivo)
├── play.py                      ✓ Script runner de tasks
└── run_arm_tracking.py          ✓ Script standalone (legado)

Uso via play.py (recomendado):
    python scripts/spot_vlm/play.py --task Spot-Arm-Tracking --device cuda:0

Uso via script direto (legado):
    python scripts/spot_vlm/run_arm_tracking.py

Tasks disponíveis:
- Spot-Arm-Tracking: Controle visual do braço usando VLM com depth
"""

__version__ = "0.1.0"
__author__ = "Diler"

# NÃO fazer imports automáticos aqui para evitar carregar Isaac Lab
# Usuários devem importar explicitamente o que precisam

__all__ = [
    "__version__",
    "__author__"
]