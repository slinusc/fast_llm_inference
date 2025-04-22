import time
from benchmark.backends.base import BaseBackend
from lmdeploy import pipeline

class LMDeployBackend(BaseBackend):
    def load_model(self):
        self.pipeline = pipeline(self.model_path, backend="turbo")

    def generate(self, prompt):
        start = time.time()
        params = self.default_generation_params()
        response = self.pipeline(prompt, max_new_tokens=params["max_tokens"])
        end = time.time()
        return response.text.strip(), end - start

    def measure_ttft(self):
        prompt = "Artificial intelligence is a rapidly evolving field with applications in healthcare, finance, education, and more. One of the most transformative technologies is"
        start = time.time()
        _ = self.pipeline(prompt, max_new_tokens=1)
        end = time.time()
        return end - start

        