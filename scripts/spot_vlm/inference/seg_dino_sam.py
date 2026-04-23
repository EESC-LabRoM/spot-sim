import torch
import numpy as np
import supervision as sv
from PIL import Image
from pathlib import Path
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import gc 
from typing import List, Union, Tuple, Optional

# Tenta importar bibliotecas específicas. 
# Se falhar, avisa o usuário (útil no ambiente Isaac Sim)
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from segment_anything import sam_model_registry, SamPredictor
except ImportError as e:
    print(f"[ERRO CRÍTICO] Dependências faltando: {e}")
    print("Instale: pip install transformers segment-anything supervision")
    exit(1)

class SemanticSegmenter:
    """
    Classe para gerenciar segmentação semântica usando GroundingDINO (Detecção) + SAM (Segmentação).
    Otimizada para carregar modelos apenas quando necessário e liberar VRAM.
    """
    
    def __init__(self, 
                 device: str = "cuda:0", 
                 dino_model_path: str = "IDEA-Research/grounding-dino-base",
                 sam_checkpoint_path: str = "./models/sam/sam_vit_b_01ec64.pth",
                 sam_type: str = "vit_b"):
        
        self.device = device
        self.dino_path = dino_model_path
        self.sam_path = sam_checkpoint_path
        self.sam_type = sam_type
        
        # Modelos
        self.dino_processor = None
        self.dino_model = None
        self.sam_predictor = None
        self.loaded = False

    def get_memory_usage(self) -> Tuple[float, float]:
        """
        Retorna o uso de memória da GPU.
        Returns:
            (allocated_gb, reserved_gb)
        """
        if torch.cuda.is_available():
            # Memória usada por tensores
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            # Memória total reservada pelo PyTorch (cache)
            reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
            return allocated, reserved
        return 0.0, 0.0

    def load_models(self):
        """Carrega os modelos na memória (VRAM)"""
        if self.loaded:
            return

        print(f"[INFO] Carregando modelos de segmentação no dispositivo: {self.device}...")
        
        # Medição inicial
        mem_start, _ = self.get_memory_usage()

        try:
            # 1. GroundingDINO
            print(f"   -> Carregando GroundingDINO de: {self.dino_path}")
            self.dino_processor = AutoProcessor.from_pretrained(self.dino_path)
            self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.dino_path).to(self.device)

            # 2. SAM (Segment Anything)
            print(f"   -> Carregando SAM ({self.sam_type}) de: {self.sam_path}")
            if not Path(self.sam_path).exists():
                 # Fallback se o caminho local não existir, tenta o padrão ou avisa
                 print(f"[AVISO] Checkpoint SAM não encontrado em {self.sam_path}. Verifique o caminho.")
            
            sam = sam_model_registry[self.sam_type](checkpoint=self.sam_path).to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            
            self.loaded = True
            
            # Medição final
            mem_end, mem_reserved = self.get_memory_usage()
            used_by_models = mem_end - mem_start
            
            print(f"[INFO] ✓ Modelos prontos! (+{used_by_models:.2f} GB usados)")
            print(f"       Total VRAM Alocada: {mem_end:.2f} GB (Reservada: {mem_reserved:.2f} GB)")
            
        except Exception as e:
            print(f"[ERRO] Falha ao carregar modelos de segmentação: {e}")
            self.unload_models() # Limpa se falhar no meio

    def unload_models(self):
        """Libera memória da GPU explicitamente"""
        print("[INFO] Liberando modelos de segmentação da memória...")
        
        if self.dino_model: del self.dino_model
        if self.dino_processor: del self.dino_processor
        if self.sam_predictor: 
            if hasattr(self.sam_predictor, 'model'): del self.sam_predictor.model
            del self.sam_predictor
            
        self.dino_model = None
        self.dino_processor = None
        self.sam_predictor = None
        self.loaded = False
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _prepare_image(self, image: Union[Path, str, np.ndarray, Image.Image]) -> Tuple[Image.Image, np.ndarray]:
        """Helper para converter qualquer entrada em PIL e Numpy"""
        try:
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image).convert("RGB")
                image_np = image
            elif isinstance(image, Image.Image):
                image_pil = image.convert("RGB")
                image_np = np.array(image_pil)
            elif isinstance(image, str):  # URL ou Path string
                if image.startswith("http"):
                    response = requests.get(image, stream=True)
                    response.raise_for_status()
                    image_pil = Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    image_pil = Image.open(image).convert("RGB")
                image_np = np.array(image_pil)
            elif isinstance(image, Path):
                image_pil = Image.open(image).convert("RGB")
                image_np = np.array(image_pil)
            else:
                raise ValueError(f"Tipo de imagem não suportado: {type(image)}")
            
            return image_pil, image_np
        except Exception as e:
            print(f"[ERRO] Falha ao processar imagem de entrada: {e}")
            return None, None

    def detect_and_segment(self, 
                          image_source: Union[Path, str, np.ndarray, Image.Image], 
                          prompt: str, 
                          box_threshold: float = 0.35, 
                          text_threshold: float = 0.25) -> dict:
        """
        Executa o pipeline completo: Texto -> Box (DINO) -> Máscara (SAM).
        """
        if not self.loaded:
            self.load_models()

        image_pil, image_np = self._prepare_image(image_source)
        if image_pil is None: return {}

        # 1. GroundingDINO (Detecção)
        # O processador do DINO espera o texto em lowercase e geralmente terminado em .
        clean_prompt = prompt.lower().strip()
        if not clean_prompt.endswith("."): clean_prompt += "."

        inputs = self.dino_processor(images=image_pil, text=clean_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        
        # --- CORREÇÃO DO ERRO ---
        # Chamamos sem os thresholds, pois as versões novas do transformers não aceitam mais
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs, 
            inputs.input_ids, 
            target_sizes=[image_pil.size[::-1]]
        )[0]

        # Extração manual e filtragem
        scores = results["scores"]
        boxes = results["boxes"]
        
        # Filtra manualmente pelo box_threshold
        keep = scores > box_threshold
        
        filtered_boxes = boxes[keep]
        filtered_scores = scores[keep]
        
        # Se não achou nada, retorna vazio
        if len(filtered_boxes) == 0:
            print(f"   [INFO] Nenhum objeto '{prompt}' encontrado com confiança > {box_threshold}.")
            empty_detections = sv.Detections.empty()
            return {
                "detections": empty_detections,
                "labels": [],
                "masks": None,
                "annotated_image": image_pil,
                "cropped_objects": []
            }

        # Converte para Supervision Detections
        detections = sv.Detections(
            xyxy=filtered_boxes.cpu().numpy(),
            confidence=filtered_scores.cpu().numpy(),
            class_id=np.array(range(len(filtered_boxes))) # ID temporário
        )
        
        # Labels (opcional, pode vir do DINO ou usar o prompt)
        labels_text = [clean_prompt.replace(".", "")] * len(detections)

        # 2. SAM (Segmentação)
        self.sam_predictor.set_image(image_np)
        
        generated_masks = []
        for box in detections.xyxy:
            masks, _, _ = self.sam_predictor.predict(
                box=box,
                multimask_output=False
            )
            generated_masks.append(masks[0]) # Pega a melhor máscara

        if generated_masks:
            detections.mask = np.array(generated_masks)

        # 3. Pós-processamento (Anotação e Recortes)
        annotated_image, montage, description, crops = self._visualize_and_crop(image_np, detections, labels_text)

        return {
            "detections": detections,
            "labels": labels_text,
            "masks": detections.mask,
            "annotated_image": annotated_image,
            "montage": montage,
            "description": description,
            "cropped_objects": crops # Lista de imagens RGBA recortadas
        }

    def _visualize_and_crop(self, image_np, detections, labels):
        """Gera visualizações e recortes transparentes"""
        box_annotator = sv.BoxAnnotator(thickness=2)
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)

        # Anotação
        annotated = mask_annotator.annotate(scene=image_np.copy(), detections=detections)
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        annotated_pil = Image.fromarray(annotated)

        # Recortes
        crops = []
        desc = "Detectados:\n----------------"
        
        # Verifica se há máscaras e detecções
        if len(detections) > 0 and detections.mask is not None:
            for i, (box, mask, label, conf) in enumerate(zip(detections.xyxy, detections.mask, labels, detections.confidence)):
                x1, y1, x2, y2 = box.astype(int)
                
                # Garante limites da imagem
                h, w, _ = image_np.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1: continue

                # Recorte RGB
                crop_rgb = image_np[y1:y2, x1:x2]
                # Recorte Máscara
                crop_mask = mask[y1:y2, x1:x2]

                # Cria RGBA (Fundo transparente)
                crop_rgba = np.zeros((crop_rgb.shape[0], crop_rgb.shape[1], 4), dtype=np.uint8)
                crop_rgba[..., :3] = crop_rgb
                crop_rgba[..., 3] = crop_mask * 255
                
                crops.append(Image.fromarray(crop_rgba))
                desc += f"\n{label} ({conf:.2f})"

        # Montagem simples para debug
        if crops:
            total_w = sum(c.width for c in crops)
            max_h = max(c.height for c in crops)
            montage = Image.new('RGBA', (total_w, max_h))
            x_off = 0
            for c in crops:
                montage.paste(c, (x_off, 0))
                x_off += c.width
        else:
            montage = Image.new('RGB', (100, 100))

        return annotated_pil, montage, desc, crops

# Exemplo de uso rápido (Main)
if __name__ == "__main__":
    # Configuração Padrão
    DINO_PATH = "./models/grounding-dino-base" # Caminho local se baixado pelo script
    SAM_PATH = "./models/sam/sam_vit_b_01ec64.pth"
    
    # Se não existir local, usa o do Hub
    if not Path(DINO_PATH).exists():
        DINO_PATH = "IDEA-Research/grounding-dino-base"

    segmenter = SemanticSegmenter(
        device="cuda:0",
        dino_model_path=DINO_PATH,
        sam_checkpoint_path=SAM_PATH
    )