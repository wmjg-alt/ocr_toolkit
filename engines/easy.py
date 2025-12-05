import easyocr
import os
import numpy as np
from PIL import Image
from engines.base import BaseOCREngine

class EasyOCREngine(BaseOCREngine):
    """
    Implementation of JaidedAI's EasyOCR.
    A traditional (Non-Transformer) Deep Learning pipeline.
    Acts as a solid baseline for comparison.
    """

    def load(self):
        langs = self.config['engines']['easyocr'].get('languages', ['en'])
        use_gpu = self.config['system']['device'] == 'cuda'
        
        print(f"   [EasyOCR] Loading model (Langs: {langs}, GPU: {use_gpu})...")
        
        try:
            # EasyOCR initializes model weights immediately upon creation
            self.reader = easyocr.Reader(
                lang_list=langs, 
                gpu=use_gpu,
                verbose=False
            )
            
            self.name = "EasyOCR (Baseline)"
            return True
            
        except Exception as e:
            print(f"   [EasyOCR] Error loading model: {e}")
            raise e

    def process(self, image_input, **kwargs):
        if self.reader is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # EasyOCR supports FilePath, Numpy, or Bytes. 
        # It handles PIL images best if converted to bytes or numpy.
        target_image = image_input
        
        if isinstance(image_input, Image.Image):
            # Convert PIL to Numpy array (RGB)
            target_image = np.array(image_input)

        try:
            # 1. Read
            # paragraph=True combines lines automatically, which is fairer for comparison
            # detail=0 returns just the text list, detail=1 includes coords
            result_list = self.reader.readtext(target_image, detail=0, paragraph=True)
            
            # 2. Join
            final_text = "\n".join(result_list)

            return {
                "text": final_text,
                "task": "ocr",
                "raw": result_list
            }

        except Exception as e:
            raise e

    def unload(self):
        self.reader = None