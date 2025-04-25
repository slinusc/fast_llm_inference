from abc import ABC, abstractmethod

class BaseBackend(ABC):
    def __init__(self, model_path, quantization=None, max_tokens=256, verbose=False):
        self.model_path = model_path
        self.quantization = quantization
        self.max_tokens = max_tokens
        self.verbose = verbose

    def default_generation_params(self):
        return {
            "temperature": 0.1,
            "max_tokens": self.max_tokens
        }

    @abstractmethod
    def load_model(self):
        """Load the model with appropriate backend-specific logic."""
        pass

    @abstractmethod
    def generate(self, prompt, task_type):
        """Generate output for a single prompt. Returns (output_text, duration_in_sec)."""
        pass

    @abstractmethod
    def measure_ttft(self):
        """Measure time-to-first-token latency. Returns duration in seconds."""
        pass