import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from benchmark.backends.base import BaseBackend

class HuggingFaceBackend(BaseBackend):
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        inputs.pop("token_type_ids", None)  # Remove unsupported key for decoder-only models

        params = self.default_generation_params()

        outputs = self.model.generate(
            **inputs,
            temperature=params["temperature"],
            max_new_tokens=params["max_tokens"]
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text


    def measure_ttft(self):
        prompt = "Artificial intelligence is a rapidly evolving field with applications in healthcare, finance, education, and more. One of the most transformative technologies is"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        start = time.time()
        _ = self.model.generate(**inputs, max_new_tokens=1)
        end = time.time()
        return end - start
