from engines.florence import FlorenceEngine
from engines.got import GotEngine
from engines.deepseek import DeepSeekEngine
from engines.qwen import QwenEngine
from engines.hunyuan import HunyuanEngine
from engines.easy import EasyOCREngine # <--- New Baseline

def get_engine(engine_name, config):
    name = engine_name.lower()
    
    if name == 'florence2':
        return FlorenceEngine(config)
    elif name == 'got_ocr':
        return GotEngine(config)
    elif name == 'deepseek':
        return DeepSeekEngine(config)
    elif name == 'hunyuan':
        return HunyuanEngine(config)
    elif name in ['qwen', 'qwen2', 'qwen3']:
        instance = QwenEngine(config)
        instance.config_key = name
        return instance
        
    elif name == 'easyocr':
        return EasyOCREngine(config) 
        
    elif "paddle" in name:
        # Disabled due to incompatibility with Transformers environment (e.g. couldn't get it workin')
        raise NotImplementedError(f"{engine_name} is currently disabled (WIP).")
        
    else:
        raise ValueError(f"Unknown engine name: {engine_name}")