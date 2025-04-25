# benchmark/backends/deepspeed_mii_backend.py
import time
import mii
from benchmark.backends.base import BaseBackend


class MIIDeepSpeedBackend(BaseBackend):
    """
    Backend for DeepSpeed-MII (v0.3.x).
    Uses the MII pipeline interface.
    """

    def load_model(self):
        self.pipeline = mii.pipeline(
            self.model_path,           # local path or HF model ID
            max_length=self.max_tokens
        )


    def generate(self, prompt, task_type=None):
        # ─── Stop words tuned per task ──────────────────────────────
        stop_strs = {
            "qa":  ["Context:", "Question:", "Answer:", "<|eot_id|>"],
            "sql": [";" , "Question:", "<|eot_id|>"],
            "summarization": ["Summary:", "<|eot_id|>"],
            None: [],
        }[task_type]

        max_new = {"qa": 64, "sql": 128, "summarization": 256}.get(
            task_type, self.max_tokens
        )

        # ─── *Safe* prompt_len: true token count if possible, else fallback ───
        try:
            prompt_len = len(self.pipeline.tokenizer.encode(prompt))
        except Exception:
            prompt_len = int(len(prompt.split()) * 2)        # generous buffer

        max_len = prompt_len + max_new + 16                 # always > prompt_len

        # ─── Generate ───────────────────────────────────────────────
        outputs = self.pipeline(
            [prompt],
            max_new_tokens=max_new,
            max_length=max_len,
            temperature=0.1,
            stop=stop_strs,          # list can be empty
        )

        text = outputs[0].generated_text.lstrip()

        # ─── Post-trim just in case ─────────────────────────────────
        text = text.split("\n", 1)[0].strip()
        
        if "<|eot_id|>" in text:
            text = text.split("<|eot_id|>", 1)[0].strip()

        return text



    def measure_ttft(self):
        """
        Measure Time-To-First-Token (TTFT).
        """
        start = time.time()
        _ = self.pipeline(
            ["Test prompt"],
            max_new_tokens=1,
            temperature=0.1
        )
        return time.time() - start

    def __del__(self):
        """
        Clean up MII pipeline properly on deletion.
        """
        if hasattr(self, "pipeline"):
            self.pipeline.destroy()
