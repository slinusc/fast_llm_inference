import os
import time
from benchmark.backends.base import BaseBackend
from llama_cpp import Llama


class LlamaCppBackend(BaseBackend):
    """
    Backend for llama.cpp, now with batch support and VLLM‐style
    stop‐string & max‐token logic.
    """

    def load_model(self):
        self.model = Llama(
            model_path=self.model_path,
            n_gpu_layers=-1,
            n_ctx=8192,
            verbose=False
        )

    def generate(self, prompts, task_type=None):
        # 1) normalize to list
        is_batch = isinstance(prompts, list)
        prompt_list = prompts if is_batch else [prompts]

        # 2) build stop_strs based on model family (like VLLM)
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

        # 3) task‐specific max_new_tokens
        max_new = {
            "qa": 64,
            "sql": 128,
            "summarization": 256,
        }.get(task_type, self.max_tokens)

        # 4) generate for each prompt
        texts = []
        for p in prompt_list:
            resp = self.model(
                prompt=p,
                max_tokens=max_new,
                temperature=0.1,
                stop=stop_strs  # list or None
            )
            text = resp["choices"][0]["text"].lstrip()
            # optional: trim at first stop token if present
            if stop_strs:
                for tok in stop_strs:
                    if tok in text:
                        text = text.split(tok, 1)[0].strip()
                        break
            texts.append(text)

        # 5) return same type as input
        return texts if is_batch else texts[0]

    def measure_ttft(self):
        prompt = "Artificial intelligence is a rapidly evolving field with applications in healthcare, finance, education, and more. One of the most transformative technologies is"
        start = time.time()
        _ = self.model(prompt=prompt, max_tokens=1)
        return time.time() - start


if __name__ == "__main__":
    backend = LlamaCppBackend(
        model_path="/path/to/model.gguf",
        max_tokens=256,
        quantization=None,
        verbose=True
    )
    backend.load_model()
    # single
    print(backend.generate("What is the capital of France?", task_type="qa"))
    # batch
    prompts = ["Hello, world!", "The future of AI is"]
    print(backend.generate(prompts, task_type="summarization"))
    print(f"TTFT: {backend.measure_ttft():.4f}s")
