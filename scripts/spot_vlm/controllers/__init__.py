"""
Controladores para o robô Spot

IMPORTANTE: Imports lazy para evitar carregar Isaac Lab antes da hora
"""

__all__ = [
    "ArmController",
    "BaseController",
    "setup_leg_lock",
    "CuroboMotionPlanner",
    "CUROBO_AVAILABLE"
]

# Variável global para Curobo
CUROBO_AVAILABLE = False

def _check_curobo():
    """Verifica se Curobo está disponível (lazy)"""
    global CUROBO_AVAILABLE
    try:
        import curobo
        CUROBO_AVAILABLE = True
    except ImportError:
        CUROBO_AVAILABLE = False
    return CUROBO_AVAILABLE