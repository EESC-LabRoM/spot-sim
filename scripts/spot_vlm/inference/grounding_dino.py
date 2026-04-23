import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, List
import requests
from io import BytesIO

try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
except ImportError as e:
    print(f"[ERRO] Instale: pip install transformers")
    exit(1)


class GroundingDINODetector:
    """Detector de objetos baseado em texto usando GroundingDINO"""
    
    def __init__(self, 
                 model_path: str = "IDEA-Research/grounding-dino-base",
                 device: str = "cuda:0"):
        self.model_path = model_path
        self.device = device
        self.processor = None
        self.model = None
        self.loaded = False

    def load(self):
        """Carrega o modelo na memória"""
        if self.loaded:
            return
            
        print(f"[INFO] Carregando GroundingDINO: {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_path
        ).to(self.device)
        self.loaded = True
        print("[INFO] ✓ GroundingDINO pronto")

    def unload(self):
        """Libera memória da GPU"""
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        self.model = None
        self.processor = None
        self.loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def detect(self, 
               image: Union[str, Path, np.ndarray, Image.Image],
               prompt: str,
               box_threshold: float = 0.35) -> dict:
        """
        Detecta objetos na imagem baseado no prompt.
        
        Returns:
            dict com 'boxes' (xyxy), 'scores' e 'labels'
        """
        if not self.loaded:
            self.load()

        image_pil = self._prepare_image(image)
        if image_pil is None:
            return {"boxes": np.array([]), "scores": np.array([]), "labels": []}

        # Processa prompt
        clean_prompt = prompt.lower().strip()
        if not clean_prompt.endswith("."):
            clean_prompt += "."

        # Inferência
        inputs = self.processor(
            images=image_pil, 
            text=clean_prompt, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[image_pil.size[::-1]]
        )[0]

        # Filtragem
        scores = results["scores"]
        boxes = results["boxes"]
        keep = scores > box_threshold
        
        filtered_boxes = boxes[keep].cpu().numpy()
        filtered_scores = scores[keep].cpu().numpy()
        labels = [clean_prompt.replace(".", "")] * len(filtered_boxes)

        return {
            "boxes": filtered_boxes,
            "scores": filtered_scores,
            "labels": labels
        }

    def _prepare_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
        """Converte entrada para PIL Image"""
        try:
            if isinstance(image, np.ndarray):
                return Image.fromarray(image).convert("RGB")
            elif isinstance(image, Image.Image):
                return image.convert("RGB")
            elif isinstance(image, str):
                if image.startswith("http"):
                    response = requests.get(image, stream=True)
                    response.raise_for_status()
                    return Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    return Image.open(image).convert("RGB")
            elif isinstance(image, Path):
                return Image.open(image).convert("RGB")
            else:
                raise ValueError(f"Tipo não suportado: {type(image)}")
        except Exception as e:
            print(f"[ERRO] Falha ao processar imagem: {e}")
            return None


if __name__ == "__main__":
    detector = GroundingDINODetector(device="cuda:0")
    
    # Teste
    test_image = "./test.jpg"
    results = detector.detect(test_image, "red cube")
    print(f"Detectados {len(results['boxes'])} objetos")
    
    detector.unload()