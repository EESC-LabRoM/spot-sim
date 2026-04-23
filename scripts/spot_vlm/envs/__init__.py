"""
Spot VLM Tasks

Coleção de tasks/environments para o robô Spot.

AVISO: Este módulo contém configurações que dependem do Isaac Lab.
Certifique-se de inicializar o AppLauncher ANTES de importar.

Tasks disponíveis:
- SpotArmTrackingEnvCfg: Tracking visual com braço usando VLM
"""

from scripts.spot_vlm.envs.arm_tracking_env import (
    SpotArmTrackingEnv,
    SpotArmTrackingEnvCfg,
)

__all__ = [
    "SpotArmTrackingEnv",
    "SpotArmTrackingEnvCfg",
]