"""
Configurações globais do sistema Spot VLM
Edite aqui para ajustar comportamento sem modificar código
"""
# ==================== SPOT ====================
USE_URDF = True  # Usar URDF ao invés de USD (True = URDF, False = USD)

# ==================== VLM ====================
VLM_MODEL_PATH = "./models/Qwen3-VL-4B-Instruct"
VLM_INFERENCE_INTERVAL = 10  # Steps entre inferências (↑ = mais rápido, ↓ = mais preciso)
VLM_ENABLED = True  # Set False para desabilitar VLM completamente

# ==================== MOTION PLANNING ====================
USE_CUROBO = True  # Usar CuRobo para planejamento de movimento (True = CuRobo, False = Simples)
USE_IK = True      # Usar Cinemática Inversa para controle do braço (True = IK, False = Motion Generation)

# ==================== ARM CONTROL ====================
# Velocidades de movimento
ARM_DELTA_FAST = 0.06     # Movimento rápido (el0)
ARM_DELTA_NORMAL = 0.03   # Movimento normal (sh1, sh0)
ARM_DELTA_SLOW = 0.015    # Movimento lento (ajustes finos)

# Multiplicador para movimento "closer/away"
ARM_EXTEND_MULTIPLIER = 1  # Quantas vezes mais rápido que normal

# Rate limiting do controle do braço
CONTROL_EVERY_N_STEPS = 2  # Steps entre comandos de braço (100 steps = 1 segundo @ 100Hz)
# Valores recomendados:
#   - 50 steps = 0.5s (rápido, menos estável)
#   - 100 steps = 1.0s (balanceado)
#   - 150 steps = 1.5s (lento, mais estável)

MAX_CONSECUTIVE_FAILURES = 20

# ==================== BASE CONTROL ====================
# Ganhos PD para travamento das pernas
BASE_KP = 400.0  # Proporcional (↑ = mais rígido)
BASE_KD = 5.0    # Derivativo (↑ = menos oscilação)

# ==================== RENDERING ====================
# Controle de performance
RENDER_EVERY_N_STEPS = 2  # Render a cada N steps (↑ = mais rápido)
RENDER_DURING_INIT = 10   # Render a cada N steps durante inicialização

# ==================== DEPTH PROCESSING ====================
TARGET_DISTANCE = 0.10  # Distância alvo do objeto (metros)
DISTANCE_TOLERANCE = 0.02  # Tolerância (±)

# ==================== DEBUGGING ====================
SAVE_IMAGES = True  # Salvar imagens de debug (rgb/depth)
SAVE_IMAGES_EVERY_N_STEPS = 5  # Frequência de salvamento

VERBOSE = False  # Prints extras de debug
FPS_MONITORING = True  # Mostrar FPS no terminal

# ==================== SCENE ====================
# Qual cena usar
USE_EXTENDED_SCENE = False  # True = com cubo azul e multilamp, False = só cubo vermelho

# ==================== COLORS ====================
APPLY_SPOT_COLORS = True  # Aplicar cores amarelo/preto no braço

# ==================== VALIDATION ====================
def validate_config():
    """Valida configurações e mostra warnings se necessário"""
    warnings = []
    
    if VLM_INFERENCE_INTERVAL < 20:
        warnings.append("⚠️  VLM_INFERENCE_INTERVAL muito baixo pode causar lentidão")
        warnings.append(f"   Atual: {VLM_INFERENCE_INTERVAL}, Recomendado: 50+")
    
    if RENDER_EVERY_N_STEPS < 2:
        warnings.append("⚠️  RENDER_EVERY_N_STEPS=1 pode causar lentidão")
        warnings.append("   Recomendado: 2 ou mais")
    
    if CONTROL_EVERY_N_STEPS < 50:
        warnings.append("⚠️  CONTROL_EVERY_N_STEPS muito baixo pode causar instabilidade")
        warnings.append(f"   Atual: {CONTROL_EVERY_N_STEPS}, Recomendado: 100+")
    
    if warnings:
        print("\n" + "="*60)
        print("AVISOS DE CONFIGURAÇÃO:")
        for w in warnings:
            print(w)
        print("="*60 + "\n")
    
    return len(warnings) == 0

# Validar ao importar
if __name__ != "__main__":
    validate_config()