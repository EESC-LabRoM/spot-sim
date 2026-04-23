import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List, Tuple
import requests
from io import BytesIO

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("[ERRO] Instale: pip install segment-anything")
    exit(1)


class SAMSegmenter:
    """Segmentador de objetos usando Segment Anything Model (SAM)"""
    
    def __init__(self,
                 checkpoint_path: str = "./models/sam/sam_vit_b_01ec64.pth",
                 model_type: str = "vit_b",
                 device: str = "cuda:0"):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device
        self.predictor = None
        self.loaded = False

    def load(self):
        """Carrega o modelo SAM"""
        if self.loaded:
            return
            
        if not Path(self.checkpoint_path).exists():
            print(f"[AVISO] Checkpoint não encontrado: {self.checkpoint_path}")
            return
            
        print(f"[INFO] Carregando SAM ({self.model_type})")
        sam = sam_model_registry[self.model_type](
            checkpoint=self.checkpoint_path
        ).to(device=self.device)
        self.predictor = SamPredictor(sam)
        self.loaded = True
        print("[INFO] ✓ SAM pronto")

    def unload(self):
        """Libera memória da GPU"""
        if self.predictor:
            if hasattr(self.predictor, 'model'):
                del self.predictor.model
            del self.predictor
        self.predictor = None
        self.loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def segment_from_boxes(self,
                          image: Union[str, Path, np.ndarray, Image.Image],
                          boxes: np.ndarray) -> List[np.ndarray]:
        """
        Segmenta objetos a partir de bounding boxes.
        
        Args:
            image: Imagem de entrada
            boxes: Array Nx4 com boxes no formato xyxy
            
        Returns:
            Lista de máscaras binárias (H, W)
        """
        if not self.loaded:
            self.load()

        image_np = self._prepare_image(image)
        if image_np is None:
            return []

        self.predictor.set_image(image_np)
        
        masks = []
        for box in boxes:
            mask, _, _ = self.predictor.predict(
                box=box,
                multimask_output=False
            )
            masks.append(mask[0])
            
        return masks

    def segment_from_points(self,
                           image: Union[str, Path, np.ndarray, Image.Image],
                           points: np.ndarray,
                           labels: np.ndarray) -> np.ndarray:
        """
        Segmenta a partir de pontos.
        
        Args:
            image: Imagem de entrada
            points: Array Nx2 com coordenadas (x, y)
            labels: Array N com 1 (foreground) ou 0 (background)
            
        Returns:
            Máscara binária (H, W)
        """
        if not self.loaded:
            self.load()

        image_np = self._prepare_image(image)
        if image_np is None:
            return None

        self.predictor.set_image(image_np)
        
        mask, _, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False
        )
        
        return mask[0]

    def crop_with_mask(self,
                       image: Union[np.ndarray, Image.Image],
                       mask: np.ndarray) -> Image.Image:
        """
        Cria recorte RGBA com fundo transparente.
        
        Args:
            image: Imagem original (RGB)
            mask: Máscara binária (H, W)
            
        Returns:
            Imagem RGBA com transparência
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # Encontra bounding box da máscara
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        # Recorta
        crop_rgb = image_np[y1:y2+1, x1:x2+1]
        crop_mask = mask[y1:y2+1, x1:x2+1]

        # RGBA
        h, w = crop_rgb.shape[:2]
        crop_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        crop_rgba[..., :3] = crop_rgb
        crop_rgba[..., 3] = crop_mask * 255

        return Image.fromarray(crop_rgba)

    def _prepare_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Converte entrada para numpy array RGB"""
        try:
            if isinstance(image, np.ndarray):
                return image
            elif isinstance(image, Image.Image):
                return np.array(image.convert("RGB"))
            elif isinstance(image, str):
                if image.startswith("http"):
                    response = requests.get(image, stream=True)
                    response.raise_for_status()
                    pil_img = Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    pil_img = Image.open(image).convert("RGB")
                return np.array(pil_img)
            elif isinstance(image, Path):
                return np.array(Image.open(image).convert("RGB"))
            else:
                raise ValueError(f"Tipo não suportado: {type(image)}")
        except Exception as e:
            print(f"[ERRO] Falha ao processar imagem: {e}")
            return None


if __name__ == "__main__":
    segmenter = SAMSegmenter(device="cuda:0")
    
    # Teste com box
    test_image = "./test.jpg"
    boxes = np.array([[100, 100, 200, 200]])  # x1, y1, x2, y2
    masks = segmenter.segment_from_boxes(test_image, boxes)
    print(f"Geradas {len(masks)} máscaras")
    
    segmenter.unload()
    