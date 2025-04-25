import time
from vllm import LLM, SamplingParams
from benchmark.backends.base import BaseBackend

class VLLMBackend(BaseBackend):
    def load_model(self):
        self.model = LLM(
            model=self.model_path,
            # quantization="bitsandbytes" if self.quantization and "bnb" in self.quantization else None,
            trust_remote_code=True,

        )

    def generate(self, prompt, task_type=None):

        stop_strs = {
            "qa":  ["\n", " Context:", "Question:", "Answer:"],
            "sql": [";", "\n", "Answer:"],
            "summarization": ["Summary:"], 
            None: None,
        }

        max_new = {"qa": 32, "sql": 64, "summarization": 256}.get(task_type, self.max_tokens)

        params = SamplingParams(
            temperature=0.1,
            max_tokens=max_new,
            stop=stop_strs[task_type]
        )

        outputs = self.model.generate(prompt, params)
        text = outputs[0].outputs[0].text
        return text.lstrip()

    def measure_ttft(self):
        prompt = "Artificial intelligence is a rapidly evolving field with applications in healthcare, finance, education, and more. One of the most transformative technologies is"
        sampling_params = SamplingParams(max_tokens=1)
        start = time.time()
        _ = self.model.generate(prompt, sampling_params)
        end = time.time()
        return end - start


if __name__ == "__main__":
    backend = VLLMBackend(
        model_path="meta-llama/Llama-2-7b-chat-hf",
        task="text-generation",
        max_tokens=256,
        quantization=None,
        verbose=True
    )
    backend.load_model()
    prompt = "Artificial intelligence is a rapidly evolving field with applications in healthcare, finance, education, and more. One of the most transformative technologies is"
    print(backend.generate(prompt))
    print(backend.measure_ttft())