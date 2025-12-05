import torch
import os
from PIL import Image
from transformers import AutoProcessor, AutoModel
from transformers import Qwen2VLForConditionalGeneration 
from engines.base import BaseOCREngine

class QwenEngine(BaseOCREngine):
    """
    Unified Implementation for Qwen2-VL and Qwen3-VL.
    """

    def load(self):
        # 1. Determine which config block to read
        # Default to 'qwen' if the factory didn't set a key (backward compatibility)
        config_key = getattr(self, 'config_key', 'qwen')
        
        repo_id = self.config['engines'][config_key]['repo_id']
        device = self.config['system']['device']
        cache_dir = self.config['system']['cache_dir']
        
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device == "cpu":
            torch_dtype = torch.float32

        self.is_qwen3 = "Qwen3" in repo_id
        model_type = "Qwen3-VL" if self.is_qwen3 else "Qwen2-VL"
        
        print(f"   [{model_type}] Loading model: {repo_id} ({device})...")
        
        try:
            # 2. Load Processor
            self.processor = AutoProcessor.from_pretrained(
                repo_id,
                trust_remote_code=True,
                cache_dir=cache_dir,
                min_pixels=256*28*28, 
                max_pixels=1280*28*28 
            )

            # 3. Load Model (Branching Logic)
            if self.is_qwen3:
                # Qwen3: Try specific class, fallback to AutoModel
                try:
                    from transformers import Qwen3VLForConditionalGeneration
                    ModelClass = Qwen3VLForConditionalGeneration
                except ImportError:
                    print(f"   [{model_type}] Native Qwen3 class not found. Using AutoModel + Remote Code.")
                    ModelClass = AutoModel

                self.model = ModelClass.from_pretrained(
                    repo_id,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map=device,
                    cache_dir=cache_dir
                ).eval()
            else:
                # Qwen2: Stable Class
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    repo_id,
                    torch_dtype=torch_dtype,
                    device_map=device,
                    cache_dir=cache_dir
                ).eval()
            
            self.name = f"{model_type} ({config_key})"
            return True
            
        except Exception as e:
            print(f"   [{model_type}] Error loading model: {e}")
            raise e

    def process(self, image_input, **kwargs):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            image = Image.open(image_input)
        else:
            image = image_input

        if image.mode != "RGB":
            image = image.convert("RGB")

        user_task = kwargs.get('task', 'ocr').lower()
        
        prompts = {
            "ocr": "Read the text in this image line by line.",
            "markdown": "Convert the document in this image to markdown format.",
            "table": "Extract the table in this image to markdown.",
            "formula": "Extract the math formulas in this image to LaTeX.",
            "det": "Detect the bounding boxes of text lines."
        }
        
        prompt_text = prompts.get(user_task, prompts["ocr"])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=2048,
                do_sample=False 
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.get("input_ids"), generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        return {
            "text": output_text,
            "task": user_task,
            "raw": output_text
        }

    def unload(self):
        self.model = None
        self.processor = None