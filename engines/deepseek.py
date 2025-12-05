import torch
import os
import tempfile
import sys
import contextlib
import re
from io import StringIO
from engines.base import BaseOCREngine
from transformers import AutoModel, AutoTokenizer

class DeepSeekEngine(BaseOCREngine):
    """
    Implementation of DeepSeek-OCR (3B).
    Uses 'unsloth/DeepSeek-OCR'.
    """

    def load(self):
        repo_id = self.config['engines']['deepseek']['repo_id']
        device = self.config['system']['device']
        cache_dir = self.config['system']['cache_dir']
        
        # DeepSeek is a 3B model, needs fp16/bf16
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device == "cpu":
            torch_dtype = torch.float32

        print(f"   [DeepSeek] Loading model: {repo_id} ({device})...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                repo_id, 
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            self.model = AutoModel.from_pretrained(
                repo_id, 
                trust_remote_code=True, 
                use_safetensors=True,
                torch_dtype=torch_dtype,
                attn_implementation="eager", 
                cache_dir=cache_dir
            ).to(device)
            
            self.model = self.model.eval()
            self.name = "DeepSeek-OCR-3B"
            return True
            
        except Exception as e:
            print(f"   [DeepSeek] Error loading model: {e}")
            raise e

    def _clean_output(self, raw_log, strip_markdown=False):
        """
        DeepSeek prints debug info. We strip that out.
        Optionally strips markdown syntax for cleaner plain text comparisons.
        """
        # 1. Remove Debug Lines (BASE:, PATCHES:, =====)
        lines = raw_log.split('\n')
        clean_lines = []
        for line in lines:
            if "BASE:" in line or "PATCHES:" in line or "=====" in line:
                continue
            clean_lines.append(line)
        
        text = "\n".join(clean_lines).strip()
        
        # 2. Remove Internal Tags
        text = text.replace("<|ref|>", "").replace("<|/ref|>", "")
        text = text.replace("<|det|>", "").replace("<|/det|>", "")
        
        # 3. Smart Markdown Stripping (Only if requested)
        if strip_markdown:
            # Remove Headers (# Header -> Header)
            text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
            # Remove Bold/Italic (**text** -> text, *text* -> text)
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            text = re.sub(r'\*([^*]+)\*', r'\1', text)
            text = re.sub(r'__([^_]+)__', r'\1', text)
            text = re.sub(r'_([^_]+)_', r'\1', text)
            # Remove Links ([text](url) -> text)
            text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
            # Remove Images (![alt](url) -> '')
            text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)
            # Remove Blockquotes (> text -> text)
            text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)
            # Remove Horizontal Rules (---)
            text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
            # Remove Backticks (`code` -> code)
            text = re.sub(r'`([^`]+)`', r'\1', text)
            
            # Clean up extra newlines created by removals
            text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def process(self, image_input, **kwargs):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        dummy_dir = tempfile.gettempdir()
        temp_file_path = None
        
        try:
            # 1. Handle Image Input
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image not found: {image_input}")
                image_path = image_input
            else:
                temp_fd, temp_file_path = tempfile.mkstemp(suffix=".png")
                os.close(temp_fd)
                image_input.save(temp_file_path)
                image_path = temp_file_path

            # 2. Smart Prompt Mapping
            user_task = kwargs.get('task', 'ocr').lower()
            
            # Decide prompts
            if user_task in ['det', 'grounding', 'box', 'coordinates']:
                prompt = "<image>\n<|grounding|>Convert the document to markdown."
                should_strip_md = False
            elif user_task == 'markdown':
                # User EXPLICITLY asked for Markdown, so don't strip it
                prompt = "<image>\n<|grounding|>Convert the document to markdown."
                should_strip_md = False
            else:
                # Default 'ocr': User wants text. DeepSeek gives markdown. We strip it.
                prompt = "<image>\nFree OCR."
                should_strip_md = True

            # 3. Inference with WIRETAP
            capture_buffer = StringIO()
            
            with torch.no_grad():
                with contextlib.redirect_stdout(capture_buffer):
                    self.model.infer(
                        self.tokenizer,
                        prompt=prompt,
                        image_file=image_path,
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        test_compress=False,
                        save_results=False, 
                        output_path=dummy_dir
                    )
            
            # 4. Extract & Clean Text
            raw_log = capture_buffer.getvalue()
            
            # We assume if task is 'ocr', we want plain text (strip markdown)
            # If task is 'markdown', we keep it raw.
            final_text = self._clean_output(raw_log, strip_markdown=should_strip_md)

            return {
                "text": final_text,
                "task": user_task,
                "raw": self._clean_output(raw_log, strip_markdown=False) # Keep formatting in raw
            }

        except Exception as e:
            raise e
            
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass

    def unload(self):
        self.model = None
        self.tokenizer = None