# benchmark/backends/lmdeploy_backend.py
import time
from lmdeploy import pipeline               # pip install lmdeploy>=0.2
from benchmark.backends.base import BaseBackend


class LMDeployBackend(BaseBackend):
    """
    Backend wrapper around lmdeploy.pipeline (Turbo or Triton back ends).
    Works with models converted via `lmdeploy convert` or the published
    “*-turbo” repos on Hugging Face.
    """

    def load_model(self):
        # Turbo backend auto-detects GPU and loads AWQ/int4 weights
        self.pipe = pipeline(self.model_path)

    def generate(self, prompts: list[str]) -> str:
        """
        Generate text; returns plain string (same contract as HF/MII/VLLM).
        """
        output = self.pipe(
            [prompts],
            max_new_tokens=self.max_tokens,
            temperature=0.1
        )
        # `output` is an MLCandidate object → use `.text`
        return output.text.strip()

    def measure_ttft(self):
        """Time-to-First-Token for a 1-token completion."""
        start = time.time()
        _ = self.pipe("Test prompt", max_new_tokens=1, temperature=0.1)
        return time.time() - start

    def __del__(self):
        if hasattr(self, "pipe"):
            # Turbo pipeline cleans up CUDA context via __exit__
            self.pipe.release()   # safe even if already freed

if __name__ == "__main__":
    backend = LMDeployBackend(
        model_path="/home/ubuntu/fast_llm_inference/models/llama-3.1-8B-Instruct",
        max_tokens=128,
        verbose=True
    )
    backend.load_model()

    prompt = "What is a KV-cache in Transformer models?"
    print(backend.generate(prompt))
    print("TTFT:", backend.measure_ttft(), "s")
