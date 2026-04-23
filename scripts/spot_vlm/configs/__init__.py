"""
Configurações do sistema Spot VLM

IMPORTANTE: Este módulo só deve ser importado APÓS o AppLauncher
ter sido inicializado, pois as configurações dependem do Isaac Lab.
"""

# NÃO importar nada aqui que dependa do Isaac Lab
# Imports devem ser feitos explicitamente pelos usuários

__all__ = [
    "create_spot_with_arm_config",
    "ARM_JOINT_NAMES",
    "LEG_JOINT_SUFFIXES", 
    "ARM_JOINT_LIMITS",
    "SpotManipulationSceneCfg",
    "SpotManipulationWithObjectsSceneCfg"
]