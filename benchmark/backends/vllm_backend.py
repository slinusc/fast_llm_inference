import time
from vllm import LLM, SamplingParams
from benchmark.backends.base import BaseBackend

class VLLMBackend(BaseBackend):
    def load_model(self):
        self.model = LLM(
            model=self.model_path,
            # quantization="bitsandbytes" if self.quantization and "bnb" in self.quantization else None,
            trust_remote_code=True
        )

    def generate(self, prompt):
        
        outputs = self.model.generate(prompt)
        text = outputs[0].outputs[0].text
        return text

    def measure_ttft(self):
        prompt = "Artificial intelligence is a rapidly evolving field with applications in healthcare, finance, education, and more. One of the most transformative technologies is"
        sampling_params = SamplingParams(max_tokens=1)
        start = time.time()
        _ = self.model.generate(prompt, sampling_params)
        end = time.time()
        return end - start
