import time
from transformers import pipeline, StoppingCriteria, StoppingCriteriaList
from benchmark.backends.base import BaseBackend


class KeywordStopper(StoppingCriteria):
    """
    Stop generation as soon as *any* stop string shows up
    **after** the prompt.
    """
    def __init__(self, stop_strings, tokenizer, prompt_len, window=48):
        super().__init__()
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.window = window

    def __call__(self, input_ids, scores, **kwargs):
        # Only inspect generated tokens (skip the prompt part)
        gen_ids = input_ids[0, self.prompt_len :][-self.window :]
        tail_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        return any(s in tail_text for s in self.stop_strings)


class HuggingFaceBackend(BaseBackend):
    """
    Wraps HF `pipeline("text-generation")`
    • return_full_text=False strips the prompt
    • device_map="auto"     handles sharding
    """

    # ────────────────────────────────────────────────────────────────
    def load_model(self):
        self.pipe = pipeline(
            "text-generation",
            model=self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = self.pipe.tokenizer

    # ────────────────────────────────────────────────────────────────
    def generate(self, prompt, task_type=None):
        stop_strs = {
            "qa":  ["<|assistant|>"],
            "sql": ["<|assistant|>"],
            "summarization": ["<|assistant|>"], 
            None: None,
        }[task_type]

        max_new = {"qa": 32, "sql": 64, "summarization": 256}.get(task_type, self.max_tokens)

        # Encode once to get prompt length for the stopper
        enc = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = enc.input_ids.shape[1]

        stopper = (
            StoppingCriteriaList(
                [KeywordStopper(stop_strs, self.tokenizer, prompt_len)]
            )
            if stop_strs else None
        )

        out = self.pipe(
            prompt,
            max_new_tokens=max_new,
            temperature=0.1,
            do_sample=False,
            return_full_text=False,
            stopping_criteria=stopper,
        )

        return out[0]["generated_text"].strip()

    # ────────────────────────────────────────────────────────────────
    def measure_ttft(self):
        start = time.time()
        _ = self.pipe(
            "Artificial intelligence is",
            max_new_tokens=1,
            temperature=0.1,
            return_full_text=False,
        )
        return time.time() - start