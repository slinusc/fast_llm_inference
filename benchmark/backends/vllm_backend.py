import time
from vllm import LLM, SamplingParams
from benchmark.backends.base import BaseBackend
import os

class VLLMBackend(BaseBackend):
    def load_model(self):
        self.model = LLM(
            model=self.model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            max_model_len=4096
        )


    def generate(self, prompts, task_type=None):

        # Normalize to list
        is_batch = isinstance(prompts, list)
        prompt_list = prompts if is_batch else [prompts]

        # --- build stop_strs & max_new exactly as before ---
        stop_strs_dict = {
            "llama": ["<|eot_id|>", "<|end_of_text|>"],
            "qwen":  ["<|im_end|>", "<|endoftext|>"],
            "gemma": ["<end_of_turn>"],
        }
        model_dir = os.path.basename(self.model_path or "").lower()
        if "llama" in model_dir:
            key = "llama"
        elif "qwen" in model_dir:
            key = "qwen"
        elif "gemma" in model_dir:
            key = "gemma"
        else:
            key = None
        stop_strs = stop_strs_dict.get(key, None)

        max_new = {"qa": 32, "sql": 64, "summarization": 256}.get(task_type, self.max_tokens)
        params = SamplingParams(
            temperature=0.1,
            max_tokens=max_new,
            stop=stop_strs
        )
        # -----------------------------------------------------

        # Call vLLM with the entire batch
        outputs = self.model.generate(prompt_list, params)

        # Extract text from each GenerateOutput
        texts = []
        for gen_out in outputs:
            # gen_out.outputs is a list of SamplingResult; take the first one
            txt = gen_out.outputs[0].text.lstrip()
            texts.append(txt)

        # Return a single string or list accordingly
        return texts if is_batch else texts[0]



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
        max_tokens=256,
        quantization=None,
        verbose=True
    )
    backend.load_model()
    prompt = "Artificial intelligence is a rapidly evolving field with applications in healthcare, finance, education, and more. One of the most transformative technologies is"
    print(backend.generate(prompt))
    print(backend.measure_ttft())