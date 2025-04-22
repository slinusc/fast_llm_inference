import time
from vllm import LLM, SamplingParams
from benchmark.backends.base import BaseBackend
from llama_cpp import Llama

class LlamaCppBackend(BaseBackend):
    def load_model(self):
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=4096,
            n_gpu_layers=self.quantization if isinstance(self.quantization, int) else -1,
            verbose=False
        )

    def generate(self, prompt):
        start = time.time()
        params = self.default_generation_params()
        response = self.model(
            prompt=prompt,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            stop=params["stop"]
        )

        end = time.time()
        text = response["choices"][0]["text"].strip()
        return text, end - start

    def measure_ttft(self):
        prompt = "Artificial intelligence is a rapidly evolving field with applications in healthcare, finance, education, and more. One of the most transformative technologies is"
        start = time.time()
        _ = self.model(prompt=prompt, max_tokens=1)
        end = time.time()
        return end - start