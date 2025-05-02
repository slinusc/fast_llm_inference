import time
from benchmark.backends.base import BaseBackend
from llama_cpp import Llama

class LlamaCppBackend(BaseBackend):
    def load_model(self):
        self.model = Llama(
            model_path=self.model_path,
            n_gpu_layers= -1,
            n_ctx=8192,
            verbose=False
        )

    def generate(self, prompt, task_type=None):

        stop_strs = {
            "qa":  ["Context:", "Question:", "Answer:", "<|eot_id|>"],
            "sql": [";" , "Question:", "<|eot_id|>"],
            "summarization": ["Summary:", "<|eot_id|>"],
            None: [],
        }[task_type]

        max_new = {"qa": 64, "sql": 128, "summarization": 256}.get(
            task_type, self.max_tokens
        )

        response = self.model(
            prompt=prompt,
            max_tokens=max_new,
            temperature=0.1,
            stop=stop_strs       # list can be empty
        )

        text = response["choices"][0]["text"].strip()
        return text

    def measure_ttft(self):
        prompt = "Artificial intelligence is a rapidly evolving field with applications in healthcare, finance, education, and more. One of the most transformative technologies is"
        start = time.time()
        _ = self.model(prompt=prompt, max_tokens=1)
        end = time.time()
        return end - start


if __name__ == "__main__":
    # Example usage
    backend = LlamaCppBackend(model_path="/home/ubuntu/fast_llm_inference/models/qwen2.5-7B-instruct-f16.gguf")
    backend.load_model()
    prompt = "What is the capital of France?"
    response = backend.generate(prompt)
    print(response)
    ttft = backend.measure_ttft()
    print(f"TTFT: {ttft} seconds")