"""
Sistema de inferência VLM com suporte a BUSCA, TRACKING e ATITUDE
Versão otimizada para GPU (Qwen2-VL-2B)
"""
import json
import re
import threading
import queue
import gc
from pathlib import Path
from PIL import Image
import warnings

# Suprimir aviso de depreciação do transformers
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Importar novos prompts
try:
    from scripts.spot_vlm.inference.prompts import SEARCH_PROMPT, get_tracking_prompt, get_attitude_tracking_prompt
except ImportError:
    # Fallback caso a pasta não esteja configurada corretamente (para testes locais)
    SEARCH_PROMPT = "Find RED CUBE"
    def get_tracking_prompt(dist): return "Track object"
    def get_attitude_tracking_prompt(dist): return "Track object with attitude"


class VLMInference:
    """Gerenciador de inferência VLM"""
    
    def __init__(self, model_path, device="cuda:0"):
        import torch
        self.torch = torch
        
        self.device = device
        self.model = None
        self.processor = None
        self.available = False
        
        # Configuração assíncrona
        self.async_mode = False
        self._input_queue = queue.Queue(maxsize=1)
        self._output_queue = queue.Queue(maxsize=1)
        self._thread = None
        self._stop_event = threading.Event()
        self._is_thinking = False
        
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Carrega modelo VLM na GPU"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"[WARN] Modelo não encontrado: {model_path}")
            return
        
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            
            print(f"[INFO] Carregando modelo de: {model_path}")
            print(f"[INFO] Dispositivo: {self.device}")

            # Configuração para GPU (Float16)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path, 
                torch_dtype=self.torch.float16, 
                device_map=self.device, 
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                _attn_implementation="flash_attention_2" if self.torch.cuda.is_available() and self.torch.cuda.get_device_capability()[0] >= 8 else "eager"
            )
            
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.available = True
            print("[INFO] ✓ Modelo VLM carregado com sucesso!")
            
        except Exception as e:
            print(f"[ERRO] Falha ao carregar VLM: {e}")

    def unload_model(self):
        """Libera memória da GPU explicitamente"""
        if self.async_mode:
            self.stop_async_loop()
            
        print("\n[INFO] Liberando recursos da GPU...")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
            
        self.available = False
        
        # Limpeza profunda
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch.cuda.ipc_collect()
            
        print("[INFO] ✓ Memória liberada.")

    def get_memory_usage(self):
        """Retorna uso de VRAM em GB"""
        if not self.torch.cuda.is_available():
            return 0.0, 0.0
        allocated = self.torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        reserved = self.torch.cuda.memory_reserved(self.device) / (1024 ** 3)
        return allocated, reserved

    def chat_generic(self, prompt, images=None, max_tokens=512):
        """Método genérico para chat"""
        if not self.available: return "Modelo não carregado."

        try:
            from qwen_vl_utils import process_vision_info
            
            content = []
            if images:
                for img in images:
                    content.append({"type": "image", "image": img})
            
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text], 
                images=image_inputs, 
                videos=video_inputs, 
                padding=True, 
                return_tensors="pt"
            ).to(self.device)
            
            with self.torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True
                )
            
            output_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            
            if "assistant" in output_text:
                output_text = output_text.split("assistant")[-1].strip()
            
            return output_text
            
        except Exception as e:
            return f"[ERRO INFERÊNCIA] {str(e)}"

    def start_async_loop(self):
        if not self.available or self.async_mode: return
        self.async_mode = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop_async_loop(self):
        if self.async_mode:
            self._stop_event.set()
            if self._thread: self._thread.join(timeout=2.0)
            self.async_mode = False

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                inputs = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._is_thinking = True
            try:
                rgb_img, depth_img, dist, mode = inputs
                
                # SELEÇÃO DE PROMPT BASEADA NO MODO
                if mode == "search":
                    result = self._run_inference(rgb_img, None, SEARCH_PROMPT)
                elif mode == "attitude_tracking":
                    # NOVO MODO DE ATITUDE
                    prompt = get_attitude_tracking_prompt(dist)
                    result = self._run_inference(rgb_img, depth_img, prompt)
                else:
                    # MODO TRACKING PADRÃO (Legado)
                    prompt = get_tracking_prompt(dist)
                    result = self._run_inference(rgb_img, depth_img, prompt)
                
                if not self._output_queue.empty():
                    try: self._output_queue.get_nowait()
                    except queue.Empty: pass
                self._output_queue.put(result)
            except Exception as e:
                print(f"[ERRO THREAD VLM] {e}")
            finally:
                self._input_queue.task_done()
                self._is_thinking = False

    def submit_async_request(self, rgb_image, depth_image, current_distance, mode="tracking"):
        if not self.async_mode: return False
        if self._input_queue.full() or self._is_thinking: return False
        try:
            self._input_queue.put_nowait((rgb_image, depth_image, current_distance, mode))
            return True
        except queue.Full:
            return False

    def get_async_result(self):
        if not self.async_mode: return None
        try: return self._output_queue.get_nowait()
        except queue.Empty: return None
    
    def process_image(self, image_tensor, target_size=(336, 336)):
        image_np = image_tensor.cpu().numpy()
        if image_np.max() <= 1.0: image_np = (image_np * 255).astype('uint8')
        else: image_np = image_np.astype('uint8')
        if image_np.shape[-1] > 3: image_np = image_np[:, :, :3]
        return Image.fromarray(image_np).resize(target_size, Image.Resampling.LANCZOS)

    def _run_inference(self, image1, image2, prompt):
        try:
            images = []
            if image1: images.append(image1)
            if image2: images.append(image2)
            response_text = self.chat_generic(prompt, images)
            output_text = response_text.replace('```json', '').replace('```', '').strip()
            return self._parse_json_response(output_text)
        except Exception as e:
            print(f"[ERRO VLM] {e}")
            return None
    
    def _parse_json_response(self, text):
        try: return json.loads(text.strip())
        except: pass
        # Tenta encontrar o JSON no texto caso haja lixo ao redor
        json_match = re.search(r'\{[^{}]*"(command|action|status|pos_command)"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try: return json.loads(json_match.group())
            except: pass
        return None