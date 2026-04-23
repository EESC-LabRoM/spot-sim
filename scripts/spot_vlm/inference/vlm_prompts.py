"""
Configuração de prompts para o VLM
Arquivo centralizador para fácil edição
"""

# ============ CONFIGURAÇÕES GLOBAIS ============
TARGET_OBJECT = "RED CUBE"  # Objeto a ser rastreado
TARGET_DISTANCE = 0.10      # Distância alvo em metros (10cm)
DISTANCE_TOLERANCE = 0.02   # Tolerância (±2cm)

# ============ PROMPT: BUSCA INICIAL ============
SEARCH_PROMPT = f"""You are controlling a robot arm to FIND the {TARGET_OBJECT}.

📷 TASK: Analyze the RGB image and determine if the target is visible.

RULES:
1. If {TARGET_OBJECT} is visible → output "target_found"
2. If NOT visible → output "search_environment"
3. Be conservative - only say "target_found" if you're confident

Output JSON only:
{{
    "status": "target_found" | "search_environment",
    "confidence": 0.0-1.0,
    "reason": "<brief explanation>"
}}"""

# ============ PROMPT: TRACKING (COM DISTÂNCIA) ============
def get_tracking_prompt(current_distance):
    """
    Gera prompt de tracking baseado na distância atual.
    
    Args:
        current_distance: Distância atual em metros
    
    Returns:
        str: Prompt formatado
    """
    # Determinar status
    if current_distance < TARGET_DISTANCE - DISTANCE_TOLERANCE:
        dist_status = "TOO CLOSE ⚠️"
    elif current_distance > TARGET_DISTANCE + DISTANCE_TOLERANCE:
        dist_status = "TOO FAR ⚠️"
    else:
        dist_status = "PERFECT ✓"
    
    return f"""Control robot arm to track {TARGET_OBJECT} on the RGB image, based on the depth map.

🎯 COMMANDS:
- "move_closer": Move forward (object on sight)
- "move_away": Move back (object too close to the camera, when < {TARGET_DISTANCE - DISTANCE_TOLERANCE}m)
- "adjust_left": Moves camera to the left (object on left side relative to center)
- "adjust_right": Moves camera to the right (object on right side relative to center)
- "adjust_up": Moves camera up (object on upper side, or upper left/right corners, relative to center)
- "adjust_down": Moves camera down (object on lower side, or lower left/right corners, relative to center)
- "hold_position": Perfect alignment

⚠️ PRIORITY ORDER:
1. Distance correction (move_closer/move_away) (MOST IMPORTANT)
2. Centering (adjust_left/adjust_right/adjust_up/adjust_down)
3. Hold when both OK

MAIN GOAL: Keep {TARGET_OBJECT} as close as possible to the center of the camera view. The bigger the object appears, the closer it is. Try to fill the entire image with the target.

IF there is no {TARGET_OBJECT} visible, try to find it by adjusting to the sides.

Output JSON only:
{{
    "command": "<command>",
    "reason": "<short reason>"
}}"""

# ============ PROMPT: NAVEGAÇÃO (PARA FUTURO) ============
NAVIGATION_PROMPT = f"""Find the {TARGET_OBJECT} by moving the robot base.

Commands: "move_forward", "turn_left", "turn_right", "stop"

Output JSON:
{{
    "action": "<action>",
    "reason": "<reason>"
}}"""