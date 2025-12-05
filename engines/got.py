import torch
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from engines.base import BaseOCREngine

class GotEngine(BaseOCREngine):
    """
    Implementation of StepFun's GOT-OCR 2.0 (HF Version).
    """

    def load(self):
        repo_id = self.config['engines']['got_ocr']['repo_id']
        device = self.config['system']['device']
        cache_dir = self.config['system']['cache_dir']
        
        # Determine dtype
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device == "cpu":
            torch_dtype = torch.float32

        print(f"   [GOT-OCR] Loading model: {repo_id} ({device})...")
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                repo_id,
                cache_dir=cache_dir
            )

            self.model = AutoModelForImageTextToText.from_pretrained(
                repo_id,
                torch_dtype=torch_dtype,
                device_map=device,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir
            )
            
            self.name = "GOT-OCR-2.0-HF"
            return True
            
        except Exception as e:
            print(f"   [GOT-OCR] Error loading model: {e}")
            raise e

    def process(self, image_input, **kwargs):
        """
        Args:
            image_input: Path or PIL Image.
            kwargs:
                mode (str): 'ocr' (default) or 'format'.
                crop (bool): Enable sliding window cropping.
                box (list): [x1, y1, x2, y2].
                color (str): 'green', 'red', 'blue'.
        """
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

        # 2. Parse Options
        mode = kwargs.get('mode', 'ocr')
        use_format = True if mode == 'format' else False
        use_crop = kwargs.get('crop', False)
        
        box = kwargs.get('box', None)
        color = kwargs.get('color', None)

        device = self.model.device

        # 3. Prepare Inputs (The Clean Way)
        # We only add box/color to the args if they exist. 
        # Passing box=None explicitly causes the current transformers version to crash.
        process_args = {
            "images": image,
            "return_tensors": "pt",
            "format": use_format,
            "crop_to_patches": use_crop
        }
        
        if box is not None:
            process_args["box"] = box
        if color is not None:
            process_args["color"] = color

        inputs = self.processor(**process_args).to(device)

        # 4. Generate
        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self.processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=4096
            )

        # 5. Decode
        decode_start_idx = inputs["input_ids"].shape[1]
        output_text = self.processor.decode(
            generate_ids[0, decode_start_idx:], 
            skip_special_tokens=True
        )

        return {
            "text": output_text,
            "mode": mode,
            "raw": output_text
        }

    def unload(self):
        self.model = None
        self.processor = None