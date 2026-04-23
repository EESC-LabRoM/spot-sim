"""
Dataset Generation Package for grasping
experiments in Isaaclab.
"""

import os
import toml

# path to the package directory
DATAGEN_DIR = os.path.dirname(os.path.abspath(__file__))

# path to the projecto root: isaac_dl_grasp
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(DATAGEN_DIR))

# path to mesh files (.fbx, .obj. usd etc): isaac_dl_grasp/assets -> this folder is in the .gitignore
ASSET_DIR = os.path.join(PROJECT_ROOT_DIR, "assets")


class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"
