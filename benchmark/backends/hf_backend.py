import os
import time
from transformers import pipeline, StoppingCriteriaList
from benchmark.backends.base import BaseBackend, KeywordStopper


class HuggingFaceBackend(BaseBackend):
    """
    Wraps HF `pipeline("text-generation")` with the same
    batching, stop-strs and max_new_tokens logic as your VLLM backend.
    """

    def load_model(self):
        self.pipe = pipeline(
            "text-generation",
            model=self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = self.pipe.tokenizer

    def generate(self, prompts, task_type=None):
        # Normalize to list
        is_batch = isinstance(prompts, list)
        prompt_list = prompts if is_batch else [prompts]

        # --- build stop_strs & max_new exactly as in VLLM ---
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
        # -----------------------------------------------------

        # Build a single Stopper based on first prompt (if any)
        stopper = None
        if stop_strs is not None:
            enc = self.tokenizer(prompt_list[0], return_tensors="pt")
            prompt_len = enc.input_ids.shape[1]
            stopper = StoppingCriteriaList([
                KeywordStopper(stop_strs, self.tokenizer, prompt_len)
            ])

        # Call HF pipeline on the batch
        pipe_kwargs = {
            "max_new_tokens": max_new,
            "temperature": 0.1,
            "do_sample": False,
            "return_full_text": False,
        }
        if stopper:
            pipe_kwargs["stopping_criteria"] = stopper

        raw_outputs = self.pipe(prompt_list, **pipe_kwargs)

        # Extract & clean—take the first generation per prompt
        texts = [out[0]["generated_text"].lstrip() for out in raw_outputs]

        return texts if is_batch else texts[0]

    def measure_ttft(self):
        start = time.time()
        _ = self.pipe(
            "Artificial intelligence is",
            max_new_tokens=1,
            temperature=0.1,
            return_full_text=False,
        )
        return time.time() - start


if __name__ == "__main__":
    backend = HuggingFaceBackend(
        model_path="meta-llama/Llama-2-7b-chat-hf",
        max_tokens=256,
        quantization=None,
        verbose=True,
    )
    backend.load_model()
    prompts = [
        "Artificial intelligence is a rapidly evolving field…",
        "The future of NLP is",
    ]
    print(backend.generate(prompts, task_type="summarization"))
    print(backend.measure_ttft())