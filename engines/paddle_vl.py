import torch
import os
import inspect
import types
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from engines.base import BaseOCREngine

# THIS DOES NOT WORK -------------------------------------------------------

class PaddleVLEngine(BaseOCREngine):
    """
    Implementation of PaddleOCR-VL (0.9B) using the 'lvyufeng' community port.
    Includes a 'Smart Shim' that only targets custom layers, avoiding PyTorch internals.
    """

    def load(self):
        repo_id = self.config['engines']['paddle_vl']['repo_id']
        device = self.config['system']['device']
        cache_dir = self.config['system']['cache_dir']
        
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device == "cpu":
            torch_dtype = torch.float32

        print(f"   [PaddleVL] Loading model: {repo_id} ({device})...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                cache_dir=cache_dir
            )
            
            self.processor = AutoProcessor.from_pretrained(
                repo_id, 
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            self.model = AutoModel.from_pretrained(
                repo_id, 
                trust_remote_code=True, 
                dtype=torch_dtype,
                device_map=device,
                cache_dir=cache_dir
            ).eval()
            
            # --- THE "SMART SHIM" PATCH ---
            # Transformers passes 'input_ids' recursively.
            # Custom Paddle layers crash on extra args. Standard PyTorch layers do not need patching.
            # We filter arguments ONLY for custom modules.
            
            def patch_module(module):
                # 1. SKIP STANDARD PYTORCH LAYERS (Fixes 'Embedding.forward' crash)
                # If the module is defined in 'torch.nn', leave it alone.
                if module.__module__.startswith('torch.nn'):
                    return

                if not hasattr(module, 'forward'):
                    return

                # 2. Drill down to the real function
                real_forward = module.forward
                while hasattr(real_forward, '__wrapped__'):
                    real_forward = real_forward.__wrapped__
                
                # 3. Analyze signature
                try:
                    sig = inspect.signature(real_forward)
                    allowed_args = set(sig.parameters.keys())
                except Exception:
                    return 

                # 4. Define Shim
                def forward_shim(self_mod, *args, **kwargs):
                    filtered_kwargs = {}
                    for k, v in kwargs.items():
                        if k in allowed_args:
                            filtered_kwargs[k] = v
                        elif any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                            filtered_kwargs[k] = v
                    
                    return real_forward(self_mod, *args, **filtered_kwargs)

                # 5. Apply Shim ONLY if the layer is strict
                accepts_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
                
                if not accepts_var_kwargs:
                    module.forward = types.MethodType(forward_shim, module)

            # Apply recursively
            self.model.apply(patch_module)
            
            print("   [PaddleVL] Applied compatibility shims to custom layers (Skipped torch.nn).")
            # ------------------------
            
            self.name = "PaddleOCR-VL-0.9B"
            return True
            
        except Exception as e:
            print(f"   [PaddleVL] Error loading model: {e}")
            raise e

    def process(self, image_input, **kwargs):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # 1. Handle Image
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            image = Image.open(image_input)
        else:
            image = image_input

        if image.mode != "RGB":
            image = image.convert("RGB")

        # 2. Map Tasks
        task_map = {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:"
        }
        user_task = kwargs.get('task', 'ocr').lower()
        query_text = task_map.get(user_task, "OCR:")

        # 3. Construct Input
        messages = [{"role": "user", "content": query_text}]
        text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # 4. Preprocess
        inputs = self.processor(
            images=image, 
            text=text_prompt, 
            return_tensors="pt", 
            format=True
        ).to(self.model.device)

        # 5. Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                do_sample=False, 
                num_beams=1, 
                max_new_tokens=1024,
                use_cache=False 
            )

        # 6. Decode
        decoded_output = self.processor.decode(
            outputs[0], 
            skip_special_tokens=True
        )

        return {
            "text": decoded_output,
            "task": user_task,
            "raw": decoded_output
        }

    def unload(self):
        self.model = None
        self.processor = None
        self.tokenizer = None