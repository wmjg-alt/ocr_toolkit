import torch
import os
from PIL import Image
from transformers import AutoProcessor, AutoModel
from engines.base import BaseOCREngine

# --- Helper from Hunyuan Docs ---
def clean_repeated_substrings(text):
    """Clean repeated substrings in text"""
    if not text: return ""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:] 
        count = 0
        i = n - length
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            return text[:n - length * (count - 1)]  
    return text
# -------------------------------

class HunyuanEngine(BaseOCREngine):
    """
    Implementation of HunyuanOCR (1B) using the 'lvyufeng/HunyuanOCR' community fix.
    """

    def load(self):
        repo_id = self.config['engines']['hunyuan']['repo_id']
        device = self.config['system']['device']
        cache_dir = self.config['system']['cache_dir']
        
        if device == "cuda" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            print(f"   [Hunyuan] Detected BF16 support. Loading weights in BFloat16 to match model hardcode.")
        else:
            # Fallback for older GPUs/CPU (might be slow due to emulation or require float32)
            torch_dtype = torch.float32
            print(f"   [Hunyuan] Warning: BF16 not natively supported. Fallback to Float32.")

        print(f"   [Hunyuan] Loading model: {repo_id} ({device})...")
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                repo_id,
                trust_remote_code=True,
                cache_dir=cache_dir,
                use_fast=False
            )

            self.model = AutoModel.from_pretrained(
                repo_id,
                trust_remote_code=True,
                torch_dtype=torch_dtype,  # <--- Now BFloat16
                device_map=device,
                cache_dir=cache_dir,
                attn_implementation="eager" 
            ).eval()
            
            self.name = "HunyuanOCR-1B"
            return True
            
        except Exception as e:
            print(f"   [Hunyuan] Error loading model: {e}")
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
        user_task = kwargs.get('task', 'ocr').lower()
        prompts = {
            "det": "检测并识别图片中的文字，将文本坐标格式化输出。", 
            "ocr": "提取文档图片中正文的所有信息用 markdown 格式表示，其中页眉、页脚部分忽略，表格用 html 格式表达，文档中公式用 latex 格式表示，按照阅读顺序组织进行解析。",
            "markdown": "提取文档图片中正文的所有信息用 markdown 格式表示，其中页眉、页脚部分忽略。", 
            "table": "把图中的表格解析为 HTML。",
            "formula": "识别图片中的公式，用 LaTeX 格式表示。"
        }
        prompt_text = prompts.get(user_task, prompts["ocr"])

        # 3. Construct Messages
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}
        ]

        # 4. Preprocess
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        
        # Enforce BFloat16 for inputs to match model weights
        device = self.model.device
        target_dtype = self.model.dtype
        
        new_inputs = {}
        for k, v in inputs.items():
            v = v.to(device)
            if torch.is_floating_point(v):
                v = v.to(target_dtype)
            new_inputs[k] = v
        inputs = new_inputs

        # 5. Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False
            )

        # 6. Decode (THE FIX)
        if "input_ids" in inputs:
            input_len = inputs["input_ids"].shape[1]
        else:
            input_len = inputs["inputs"].shape[1] if "inputs" in inputs else 0

        # FIX: Slice the tensor to get the new tokens...
        new_tokens = generated_ids[0][input_len:]
        
        # ...and use .decode() (Singular) to handle the 1D tensor correctly as one string
        output_text = self.processor.decode(
            new_tokens, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        final_text = clean_repeated_substrings(output_text)

        return {
            "text": final_text,
            "task": user_task,
            "raw": output_text
        }

    def unload(self):
        self.model = None
        self.processor = None