from abc import ABC, abstractmethod

class BaseOCREngine(ABC):
    """
    The Abstract Base Class that all OCR engines must inherit from.
    This ensures that whether we use Florence, GOT, or Paddle, 
    the methods to call them are exactly the same.
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
        self.name = "BaseEngine"

    @abstractmethod
    def load(self):
        """Load the model into memory (GPU/CPU)."""
        pass

    @abstractmethod
    def process(self, image_input, **kwargs):
        """
        Process an image and return text.
        
        Args:
            image_input: Path to image, or PIL Image object.
            **kwargs: Extra arguments specific to the model (e.g. prompts).
            
        Returns:
            dict: Standardized result format. 
                  Example: {'text': "extracted text", 'raw': raw_output}
        """
        pass
    
    def unload(self):
        """Optional: Clear GPU memory if needed."""
        self.model = None
        self.processor = None