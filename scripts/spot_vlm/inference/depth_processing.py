"""
Processamento de mapas de profundidade
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def process_depth_map(depth_data):
    """
    Processa mapa de profundidade para visualização com escala de cores Viridis.
    
    Args:
        depth_data: Tensor de profundidade [H, W] ou [H, W, C]
    
    Returns:
        np.ndarray: Mapa colorido [H, W, 3] (RGB) em uint8
    """
    # 1. Converter para numpy e garantir 2D
    # Verifica se é tensor do PyTorch antes de chamar .cpu()
    if hasattr(depth_data, 'cpu'):
        depth_np = depth_data.cpu().numpy()
    else:
        depth_np = depth_data
        
    if len(depth_np.shape) == 3:
        depth_np = depth_np[:, :, 0]
    
    # 2. Normalizar para 0-1 (limitar a 5 metros)
    # Valores > 5m serão cortados, valores < 0 serão 0
    depth_normalized = np.clip(depth_np, 0, 5.0) / 5.0
    
    # 3. Aplicar a escala de cores 'viridis'
    # O matplotlib retorna um array [H, W, 4] (RGBA) com floats entre 0.0 e 1.0
    # Onde: 0.0 (perto) = Roxo, 1.0 (longe) = Amarelo
    viridis_map = plt.cm.viridis(depth_normalized)
    
    # 4. Converter para formato de imagem RGB uint8 (0-255)
    # Descartamos o canal Alpha (transparência) com slicing [:, :, :3]
    depth_colored = (viridis_map[:, :, :3] * 255).astype(np.uint8)
    
    return depth_colored


def calculate_camera_adjustments(depth_map, target_distance=0.10):
    """
    Calcula ajustes necessários baseados no mapa de profundidade.
    
    Args:
        depth_map: Mapa de profundidade [H, W]
        target_distance: Distância alvo (metros)
    
    Returns:
        dict: {
            "move_closer": bool,
            "move_away": bool,
            "adjust_left": bool,
            "adjust_right": bool,
            "distance": float,
            "is_centered": bool
        }
    """
    # Garantir 2D
    if len(depth_map.shape) == 3:
        depth_np = depth_map[:, :, 0]
    else:
        depth_np = depth_map
    
    h, w = depth_np.shape
    
    # Região central
    center_region = depth_np[h//3:2*h//3, w//3:2*w//3]
    center_region_valid = center_region[np.isfinite(center_region) & (center_region < 10.0)]
    
    if center_region_valid.size > 0:
        center_depth = np.median(center_region_valid)
    else:
        center_depth = 5.0
    
    # Verificar centralização
    left_region = depth_np[h//3:2*h//3, :w//3]
    right_region = depth_np[h//3:2*h//3, 2*w//3:]
    
    left_region_valid = left_region[np.isfinite(left_region) & (left_region < 10.0)]
    right_region_valid = right_region[np.isfinite(right_region) & (right_region < 10.0)]
    
    left_depth = np.min(left_region_valid) if left_region_valid.size > 0 else float('inf')
    right_depth = np.min(right_region_valid) if right_region_valid.size > 0 else float('inf')
    
    adjustments = {
        "move_closer": center_depth > target_distance + 0.02,
        "move_away": center_depth < target_distance - 0.02,
        "adjust_left": right_depth < left_depth - 0.05,
        "adjust_right": left_depth < right_depth - 0.05,
        "distance": float(center_depth),
        "is_centered": abs(left_depth - right_depth) < 0.05 if np.isfinite(left_depth) and np.isfinite(right_depth) else False
    }
    
    return adjustments


class DepthVisualizer:
    """Helper para visualização de profundidade"""
    
    def __init__(self, output_dir="./debug_vision"):
        """
        Args:
            output_dir: Diretório para salvar imagens
        """
        from pathlib import Path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_depth_visualization(self, depth_colored, step_count):
        """Salva visualização de profundidade"""
        filename = self.output_dir / f"depth_{step_count:06d}.jpg"
        Image.fromarray(depth_colored).save(filename)
    
    def save_rgb_with_depth_overlay(self, rgb_np, depth_colored, step_count):
        """Salva RGB com overlay de profundidade"""
        # Blend 70% RGB + 30% Depth
        blended = (0.7 * rgb_np + 0.3 * depth_colored).astype(np.uint8)
        filename = self.output_dir / f"blended_{step_count:06d}.jpg"
        Image.fromarray(blended).save(filename)