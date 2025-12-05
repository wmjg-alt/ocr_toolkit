import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from engines.base import BaseOCREngine
import os

class FlorenceEngine(BaseOCREngine):
    """
    Implementation of Microsoft's Florence-2 Vision Language Model.
    """

    def load(self):
        # 1. Get settings
        repo_id = self.config['engines']['florence2']['repo_id']
        device = self.config['system']['device']
        cache_dir = self.config['system']['cache_dir']
        quantization = self.config['engines']['florence2']['quantization']

        # Determine dtype
        torch_dtype = torch.float16 if quantization == "float16" and device == "cuda" else torch.float32

        print(f"   [Florence] Loading model: {repo_id} ({device})...")
        
        try:
            # 3. Load the Model
            # FIX APPLIED: attn_implementation="eager" prevents the SDPA crash
            self.model = AutoModelForCausalLM.from_pretrained(
                repo_id, 
                dtype=torch_dtype,          # Changed from torch_dtype= to dtype=
                trust_remote_code=True,
                cache_dir=cache_dir,
                attn_implementation="eager" # <--- The Fix for the crash
            ).to(device)

            # 4. Load the Processor
            self.processor = AutoProcessor.from_pretrained(
                repo_id, 
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            
            self.name = "Florence-2"
            return True
            
        except Exception as e:
            print(f"   [Florence] Error loading model: {e}")
            raise e

    def process(self, image_input, **kwargs):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # 1. Handle Image Input
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            image = Image.open(image_input)
        else:
            image = image_input
            
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 2. Prepare Prompt
        default_task = self.config['engines']['florence2'].get('task_prompt', '<OCR>')
        task_prompt = kwargs.get('task_prompt', default_task)

        # 3. Preprocess
        device = self.config['system']['device']
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Cast pixel_values to half precision if model is half
        if "pixel_values" in inputs and self.model.dtype == torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        # 4. Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
                use_cache=False,  # <--- THE FIX: Disables broken cache logic in Transformers 4.45+
            )

        # 5. Decode
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )

        final_text = parsed_answer.get(task_prompt, "") if isinstance(parsed_answer, dict) else parsed_answer

        return {
            "text": str(final_text),
            "raw": parsed_answer,
            "task": task_prompt
        }
    
    def unload(self):
        self.model = None
        self.processor = None