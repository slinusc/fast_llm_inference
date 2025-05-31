import os
import time
import mii
from benchmark.backends.base import BaseBackend


class MIIDeepSpeedBackend(BaseBackend):
    """
    Backend for DeepSpeed-MII (v0.3.x), styled exactly like VLLMBackend.
    """

    def load_model(self):
        self.pipeline = mii.pipeline(
            self.model_path
        )

    def generate(self, prompts, task_type=None, perplexity=False):
        # Normalize to list and extract string prompt
        is_batch = isinstance(prompts, list)
        prompt_list = prompts if is_batch else [prompts]

        # Task-specific token limits
        max_new = {"qa": 32, "sql": 64, "summarization": 256}.get(task_type, self.max_tokens)


        # Call MII pipeline
        outputs = self.pipeline(
            prompt_list,
            max_new_tokens=max_new,
            do_sample=False,
        )

        outputs = [out.generated_text for out in outputs]

        return outputs

    def measure_ttft(self):
        start = time.time()
        _ = self.pipeline(["Test prompt"], max_new_tokens=1, temperature=0.1)
        return time.time() - start

    def __del__(self):
        if hasattr(self, "pipeline"):
            self.pipeline.destroy()
