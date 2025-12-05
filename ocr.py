import os
import sys
import logging
import yaml
import traceback

# =========================================================================
# SYSTEM CONFIGURATION & STABILITY HACKS
# =========================================================================
# 1. OpenMP/MKL: Prevent crashes when mixing PyTorch (Intel OpenMP) and other libs.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_THREADING_LAYER"] = "GNU"

# 2. Windows/HuggingFace: Suppress symlink warnings and OneDNN spam.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 3. Transformers: Silence non-critical warnings.
import transformers
transformers.logging.set_verbosity_error()

# 4. Fault Handler: Dumps traceback on hard crashes (SegFaults/Access Violations).
import faulthandler
faulthandler.enable()
# =========================================================================

from core.logger import setup_logger
from core.factory import get_engine

class OCR:
    """
    Main Interface for the Local OCR Toolkit.
    Handles configuration loading, engine switching, and error logging.
    """

    def __init__(self, engine=None, config_path="config.yaml"):
        """
        Initialize the OCR system.
        
        Args:
            engine (str, optional): Name of the engine to load immediately. 
                                    If None, loads default from config.
            config_path (str): Path to the YAML configuration file.
        """
        # 1. Load Configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 2. Setup Logging
        log_level = getattr(logging, self.config['system'].get('log_level', 'INFO'))
        self.logger = setup_logger(level=log_level)
        self.logger.info("Initializing OCR System...")

        # 3. Internal State
        self.active_engine = None
        self.engine_name = None
        
        # 4. Auto-Load Engine
        target_engine = engine or self.config['system'].get('default_engine')
        
        if target_engine:
            try:
                self.load_engine(target_engine)
            except Exception as e:
                self.logger.error(f"Failed to auto-load engine '{target_engine}': {e}")

    def load_engine(self, engine_name):
        """
        Unloads the current model (freeing VRAM) and loads the new one.
        """
        self.logger.info(f"Loading Engine: {engine_name}...")
        
        # 1. Unload previous engine
        if self.active_engine:
            self.logger.info("Unloading previous engine...")
            self.active_engine.unload()
            self.active_engine = None
            
            # Force garbage collection for GPU memory
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 2. Load new engine via Factory
        try:
            engine_instance = get_engine(engine_name, self.config)
            engine_instance.load()
            
            self.active_engine = engine_instance
            self.engine_name = engine_name
            self.logger.info(f"Successfully loaded {engine_name}.")
            
        except Exception as e:
            self.logger.error(f"Failed to load engine '{engine_name}': {e}")
            raise e

    def read(self, image_input, **kwargs):
        """
        Process an image using the active engine.
        
        Args:
            image_input (str|Image): Path to file or PIL Image object.
            **kwargs: Engine-specific parameters (e.g., task='markdown').
            
        Returns:
            dict: {'text': str, 'task': str, 'raw': any} or None on error.
        """
        if not self.active_engine:
            self.logger.error("No engine loaded! Call load_engine() first.")
            return None
        
        self.logger.info(f"Processing image with {self.engine_name}...")
        
        try:
            return self.active_engine.process(image_input, **kwargs)
        except Exception as e:
            self.logger.error(f"Error during processing: {e}")
            self.logger.debug(traceback.format_exc())
            return None