"""
Prompt para controle de atitude e posição simultâneos.
ATUALIZADO: Inclui status IK e distância EE-base
"""
from .base_prompts import TARGET_OBJECT, TARGET_DISTANCE, DISTANCE_TOLERANCE

def get_attitude_tracking_prompt(current_distance, ik_status="OK", ee_distance=0.0):
    """
    Gera prompt com informações de IK e distância do EE.
    
    Args:
        current_distance: Distância do objeto (metros)
        ik_status: String de status do IK ("OK" ou "FAILED (N consecutive)")
        ee_distance: Distância do EE até a base (metros)
    """
    if current_distance < TARGET_DISTANCE - DISTANCE_TOLERANCE:
        dist_status = "TOO CLOSE ⚠️"
    elif current_distance > TARGET_DISTANCE + DISTANCE_TOLERANCE:
        dist_status = "TOO FAR ⚠️"
    else:
        dist_status = "PERFECT ✓"

    return f"""Control robot arm to track {TARGET_OBJECT}. Adjust POSITION and ATTITUDE (orientation).

📊 SYSTEM STATUS:
- Object Distance: {dist_status} ({current_distance:.3f}m)
- IK Solver: {ik_status}
- EE-Base Distance: {ee_distance:.3f}m (Limit: ~0.70m)

🎯 POSITION COMMANDS (Move Camera Base):
- "move_closer": Move FORWARD relative to where camera is pointing
- "move_away": Move BACKWARD relative to where camera is pointing
- "adjust_left": Move LEFT relative to camera orientation
- "adjust_right": Move RIGHT relative to camera orientation
- "adjust_up": Move UP (vertical)
- "adjust_down": Move DOWN (vertical)
- "hold_position": Position is good

👀 ATTITUDE COMMANDS (Rotate Camera - CUMULATIVE):
- "look_at_center": Keep current orientation
- "look_up": Pitch camera UP (object too low in frame)
- "look_down": Pitch camera DOWN (object too high in frame)
- "look_left": Yaw camera LEFT (object too far right in frame)
- "look_right": Yaw camera RIGHT (object too far left in frame)
- "reset_attitude": Reset cumulative pitch/yaw to zero

⚠️ IMPORTANT NOTES:
1. POSITION commands are RELATIVE to camera orientation (not world frame)
2. ATTITUDE changes are CUMULATIVE (each command adds rotation)
3. If IK fails repeatedly, try smaller adjustments or "reset_attitude"
4. Keep EE-Base distance < 0.70m to avoid workspace limits

🎯 PRIORITY:
1. Keep IK solver working (avoid extreme poses)
2. Center the object (Use attitude for fine alignment)
3. Maintain target distance

Output JSON only:
{{
    "pos_command": "<move_closer|move_away|adjust_left|adjust_right|adjust_up|adjust_down|hold_position>",
    "att_command": "<look_up|look_down|look_left|look_right|reset_attitude|look_at_center>",
    "reason": "<brief reasoning>"
}}"""
