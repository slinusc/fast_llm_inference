import torch
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPowerUsage,
    nvmlShutdown
)
import time
import pandas as pd
import os
from huggingface_hub import hf_hub_download
import json
import numpy as np
import math
from llama_cpp import Llama


class ModelBenchmark:
    def __init__(
        self,
        model=None,
        tokenizer=None,
        max_tokens: int = 256,
        backend="huggingface",
        llama_model_path=None,
        llama_gpu_layers=35,
        model_size="N/A",
        verbose=False
    ):
        """
        Initialize benchmarking for Hugging Face, vLLM, and llama.cpp models.

        :param model: Hugging Face/vLLM model or None for llama.cpp
        :param tokenizer: HF tokenizer (optional if using llama.cpp)
        :param max_tokens: Max tokens to generate
        :param backend: 'huggingface', 'vllm', or 'llama.cpp'
        :param llama_model_path: GGUF model path (required if backend == llama.cpp)
        :param llama_gpu_layers: Number of layers to offload to GPU (llama.cpp only)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.backend = backend
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_size = model_size
        self.verbose = verbose

        # llama.cpp initialization
        if self.backend == "llama.cpp":
            if llama_model_path is None:
                raise ValueError("You must provide llama_model_path for llama.cpp backend.")
            if self.verbose:
                print(f"Loading llama.cpp model from {llama_model_path} ...")
            self.llm = Llama(
                model_path=llama_model_path,
                n_ctx=4096,
                n_gpu_layers=llama_gpu_layers,
                logits_all=True,
                verbose=False
            )
        else:
            self.llm = None

        # GPU power/memory tracking
        if torch.cuda.is_available():
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(0)
        else:
            self.handle = None

    def _get_gpu_memory_usage(self):
        """Get GPU memory usage in MB."""
        if self.device == "cuda" and self.handle:
            info = nvmlDeviceGetMemoryInfo(self.handle)
            return round(info.used / (1024 ** 2), 2)
        return 0

    def _get_gpu_power_usage(self):
        """Get GPU power usage in Watts."""
        if self.device == "cuda" and self.handle:
            return round(nvmlDeviceGetPowerUsage(self.handle) / 1000.0, 2)
        return 0

    def close(self):
        """Shutdown NVML (clean-up)."""
        if self.device == "cuda":
            nvmlShutdown()

    def generate_once(self, prompt: str):
        """Generates text and measures generation time."""
        start_time = time.time()

        if self.backend == "huggingface":
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        elif self.backend == "vllm":
            # Assuming self.model.generate returns the text (adjust if not)
            generated_text = self.model.generate(prompt, max_tokens=self.max_tokens)

        elif self.backend == "llama.cpp":
            response = self.llm(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=0.0,
                stop=["</s>"]
            )
            generated_text = response["choices"][0]["text"].strip()

        else:
            raise ValueError(f"Unsupported backend '{self.backend}'")

        end_time = time.time()
        generation_time = end_time - start_time

        if self.verbose:
            print(f"\nPrompt:\n{prompt}\n\nAnswer:\n{generated_text}\n")
        return generated_text, generation_time

    def measure_storage(self):
        """Estimate memory + KV cache + model size for HuggingFace (optional for llama.cpp)."""
        mem_usage = self._get_gpu_memory_usage()

        try:
            # HuggingFace model size (optional, if llama.cpp skip this)
            if self.backend == "huggingface":
                model_path = hf_hub_download(
                    self.model.config._name_or_path,
                    filename="config.json",
                    repo_type="model"
                )
                model_dir = os.path.dirname(model_path)
                model_size = sum(
                    f.stat().st_size for f in os.scandir(model_dir) if f.is_file()
                ) / (1024 ** 2)
            else:
                model_size = self.model_size
        except Exception:
            model_size = "N/A"

        kv_cache_estimation = (
            mem_usage - model_size if isinstance(model_size, float) else "N/A"
        )

        return {
            "Memory Usage (MB)": mem_usage,
            "Model Size (MB)": model_size,
            "KV-Cache Size Estimation (MB)": kv_cache_estimation
        }

    def measure_energy(self, num_tokens, num_sentences, generation_time):
        """Compute energy consumption based on time/power."""
        power_watts = self._get_gpu_power_usage()
        total_energy_wh = (power_watts * generation_time) / 3600

        energy_per_token = (
            total_energy_wh * 3600 / num_tokens if num_tokens > 0 else 0
        )
        energy_per_sentence = (
            total_energy_wh * 3600 / num_sentences if num_sentences > 0 else 0
        )

        return {
            "Total Energy (Wh)": round(total_energy_wh, 6),
            "Energy per Token (J/token)": round(energy_per_token, 6),
            "Energy per Sentence (J/sentence)": round(energy_per_sentence, 6),
            "Energy per Second (W)": round(power_watts, 6)
        }


    def _compute_perplexity(self, text):
        """
        Compute perplexity for a given text.
        Perplexity (PPL) = exp(-mean log probability)
        
        Works for Hugging Face and llama.cpp models.
        """
        if self.backend == "huggingface":
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            ppl = math.exp(loss)
            return round(ppl, 4)

        elif self.backend == "llama.cpp":
            response = self.llm(
                prompt=text,
                max_tokens=0,  # No generation, just evaluate log probabilities
                logprobs=True  # Get logprobs
                
            )

            # Extract log probabilities
            logprobs = response["choices"][0].get("logprobs", {}).get("token_logprobs", None)

            if logprobs is None or len(logprobs) == 0:
                print(f"Warning: No logprobs returned for text: {text}")
                return float('inf')  # Can't compute PPL

            # Compute perplexity
            avg_neg_logprob = -sum(logprobs) / len(logprobs)
            ppl = math.exp(avg_neg_logprob)
            return round(ppl, 4)

        else:
            raise ValueError("Perplexity computation not supported for this backend")




    def benchmark(self, prompts):
        """
        Run benchmarking on a list of prompts.
        Measures latency, throughput, memory, energy, perplexity for prompt and generation.
        
        Args:
            prompts (List[str]): List of input prompts.
        
        Returns:
            pd.DataFrame: Benchmark results for each prompt.
        """
        results = []

        for prompt in prompts:
            if self.verbose:
                print(f"\nEvaluating prompt ({len(prompt)} characters)...")

            # Step 1: Generate text and measure generation time
            generated_text, generation_time = self.generate_once(prompt)

            # Step 2: Count tokens and sentences in generation
            if self.backend == "huggingface":
                num_tokens = len(self.tokenizer(generated_text).input_ids)
            elif self.backend in ["vllm", "llama.cpp"]:
                num_tokens = len(generated_text.split())
            else:
                num_tokens = 0

            num_sentences = generated_text.count('.') + 1

            # Step 3: Latency metrics
            FTL = generation_time / num_tokens if num_tokens > 0 else generation_time
            ATL = (
                (generation_time - FTL) / (num_tokens - 1)
                if num_tokens > 1 else 0
            )

            latency = {
                "FTL (s)": round(FTL, 4),
                "ATL (s)": round(ATL, 4),
                "GL (s)": round(generation_time, 4)
            }

            # Step 4: Throughput
            TPS = num_tokens / generation_time if generation_time > 0 else 0
            SPS = num_sentences / generation_time if generation_time > 0 else 0

            throughput = {
                "TPS (tokens/s)": round(TPS, 2),
                "SPS (sentences/s)": round(SPS, 2)
            }

            # Step 5: Storage & energy
            storage = self.measure_storage()
            energy = self.measure_energy(num_tokens, num_sentences, generation_time)

            # Step 6: Perplexity
            #perplexity_prompt = self._compute_perplexity(prompt)
            #perplexity_generation = self._compute_perplexity("generated_text")

            # Step 7: Collect results
            result = {
                "Prompt Length": len(prompt),
                "Question (Prompt)": prompt,
                "Generated Answer": generated_text,
                #"Perplexity (Prompt)": perplexity_prompt,
                #"Perplexity (Generation)": perplexity_generation,
                **latency,
                **throughput,
                **storage,
                **energy
            }

            results.append(result)

            if self.verbose:
                print(f"\nResults for this prompt:\n{result}")

        # Return as dataframe
        df = pd.DataFrame(results)
        return df
