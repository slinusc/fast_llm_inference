# benchmark/backends/deepspeed_mii_backend.py
import os
import time
import mii
from benchmark.backends.base import BaseBackend


class MIIDeepSpeedBackend(BaseBackend):
    """
    Backend for DeepSpeed-MII (v0.3.x), with VLLM‐style
    model‐type stop‐string logic and task‐type max‐token mapping.
    """

    def load_model(self):
        self.pipeline = mii.pipeline(
            self.model_path,           # local path or HF model ID
            max_length=self.max_tokens
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
        max_new = {"qa": 64, "sql": 128, "summarization": 256}.get(
            task_type, self.max_tokens
        )

        # 4) compute prompt_len from first prompt (fallback if tokenizer fails)
        try:
            prompt_len = len(self.pipeline.tokenizer.encode(prompt_list[0]))
        except Exception:
            prompt_len = int(len(prompt_list[0].split()) * 2)

        # ensure max_length covers prompt + new + margin
        max_len = prompt_len + max_new + 16

        # 5) call MII pipeline on the batch
        outputs = self.pipeline(
            prompt_list,
            max_new_tokens=max_new,
            max_length=max_len,
            temperature=0.1,
            stop=stop_strs,          # None or list of strings
        )

        # 6) post-process each output
        texts = []
        for out in outputs:
            text = out.generated_text.lstrip()
            # trim at first newline
            text = text.split("\n", 1)[0].strip()
            # strip trailing stop token if present
            if stop_strs:
                for tok in stop_strs:
                    if tok in text:
                        text = text.split(tok, 1)[0].strip()
                        break
            texts.append(text)

        # 7) return list or single string to match input type
        return texts if is_batch else texts[0]

    def measure_ttft(self):
        start = time.time()
        _ = self.pipeline(
            ["Test prompt"],
            max_new_tokens=1,
            temperature=0.1
        )
        return time.time() - start

    def __del__(self):
        if hasattr(self, "pipeline"):
            self.pipeline.destroy()
