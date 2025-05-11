from abc import ABC, abstractmethod
from transformers import pipeline, StoppingCriteria, StoppingCriteriaList

class KeywordStopper(StoppingCriteria):
    """
    Stop generation as soon as *any* stop string shows up
    **after** the prompt.
    """
    def __init__(self, stop_strings, tokenizer, prompt_len, window=48):
        super().__init__()
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.window = window

    def __call__(self, input_ids, scores, **kwargs):
        # Only inspect generated tokens (skip the prompt part)
        gen_ids = input_ids[0, self.prompt_len :][-self.window :]
        tail_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        return any(s in tail_text for s in self.stop_strings)


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