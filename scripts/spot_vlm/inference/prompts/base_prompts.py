"""
Configurações globais e prompts de busca.
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