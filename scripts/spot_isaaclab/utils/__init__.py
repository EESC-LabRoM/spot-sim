"""
Utilitários gerais do sistema

IMPORTANTE: Imports lazy para collision_utils e robot_initialization
pois dependem do Isaac Lab
"""

__all__ = [
    "create_primitive_arm_colliders",
    "enable_mesh_collisions",
    "initialize_robot_sequence",
    "get_arm_joint_info",
    "apply_spot_colors"
]