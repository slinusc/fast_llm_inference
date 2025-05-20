import time
from vllm import LLM, SamplingParams
from benchmark.backends.base import BaseBackend
import os
import math

class VLLMBackend(BaseBackend):
    def load_model(self):
        self.model = LLM(
            model=self.model_path,
            trust_remote_code=True
        )


    def generate(self, prompts, task_type=None, perplexity=False):
        """
        Generate text (and optional per-token perplexities) with vLLM.

        Args:
            prompts (str or List[str]): A single prompt or a batch of prompts.
            task_type (str, optional): One of "qa", "sql", "summarization", or None.
            perplexity (bool): If True, return per-token log-probs and perplexities.

        Returns:
            If perplexity=False:
                str or List[str]: The generated text(s).
            If perplexity=True:
                dict or List[dict]: Each dict contains:
                - "text": str, the generated text
                - "token_ids": List[int], the vLLM token IDs
                - "tokens": List[str], the decoded token strings
                - "logprobs": List[float], the log-prob of each token
                - "perplexities": List[float], exp(–log-prob) per token
        """
        # Normalize to list
        is_batch = isinstance(prompts, list)
        prompt_list = prompts if is_batch else [prompts]

        # Build stop strings based on model
        stop_strs_map = {
            "llama": ["<|eot_id|>", "<|end_of_text|>"],
            "qwen":  ["<|im_end|>", "<|endoftext|>"],
            "gemma": ["<end_of_turn>"],
        }
        model_dir = os.path.basename(self.model_path or "").lower()
        key = next((k for k in stop_strs_map if k in model_dir), None)
        stop_strs = stop_strs_map.get(key, None)

        # Determine how many new tokens to generate
        max_new = {
            "qa":             32,
            "sql":            64,
            "summarization": 256
        }.get(task_type, self.max_tokens)

        # Configure SamplingParams
        if perplexity:
            params = SamplingParams(
                temperature=0.1,
                max_tokens=max_new,
                stop=stop_strs,
                logprobs=1,
                prompt_logprobs=1,
            )
        else:
            params = SamplingParams(
                temperature=0.1,
                max_tokens=max_new,
                stop=stop_strs
            )

        # Call vLLM
        outputs = self.model.generate(prompt_list, params)

        results = []
        for gen_out in outputs:
            # gen_out.outputs is a list; take the first CompletionOutput
            sample = gen_out.outputs[0]
            text = sample.text.lstrip()

            if perplexity:
                # Raw token IDs
                token_ids = sample.token_ids                           # :contentReference[oaicite:0]{index=0}

                # sample.logprobs is an OpenAI‐style dict:
                # {
                #    "tokens": [...],                # str tokens
                #    "token_logprobs": [...],        # float log-probs
                #    "top_logprobs": [...],          # list of dicts (optional)
                #    "text_offset": [...]
                # }
                lp_dict  = sample.logprobs                             
                tokens   = lp_dict["tokens"]                            # 
                logps    = lp_dict["token_logprobs"]
                # Compute per-token perplexity
                ppl      = [math.exp(-lp) for lp in logps]

                results.append({
                    "text":         text,
                    "token_ids":    token_ids,
                    "tokens":       tokens,
                    "logprobs":     logps,
                    "perplexities": ppl,
                })
            else:
                results.append({"text": text})

        return results if is_batch else results[0]



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