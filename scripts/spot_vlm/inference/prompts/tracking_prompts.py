"""
Prompt original de tracking (apenas posição).
"""
from .base_prompts import TARGET_OBJECT, TARGET_DISTANCE, DISTANCE_TOLERANCE

def get_tracking_prompt(current_distance):
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
1. Centering (adjust_left/adjust_right/adjust_up/adjust_down)
2. Distance correction (move_closer/move_away) (MOST IMPORTANT)
3. Hold when both OK

MAIN GOAL: Keep {TARGET_OBJECT} as close as possible to the center of the camera view.

Output JSON only:
{{
    "command": "<command>",
    "reason": "<short reason>"
}}"""