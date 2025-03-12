import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage
import time
import pandas as pd
import os
from huggingface_hub import hf_hub_download
import json


class ModelBenchmark:
    def __init__(self, model, tokenizer=None, max_tokens: int = 256, backend="huggingface"):
        """
        Initialize benchmarking for both Hugging Face and vLLM models.

        :param model: Hugging Face model or vLLM handler.
        :param tokenizer: Tokenizer (only required for HF).
        :param max_tokens: Maximum number of tokens to generate.
        :param backend: 'huggingface' or 'vllm'.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.backend = backend

        # GPU Tracking Setup
        if torch.cuda.is_available():
            nvmlInit()
            self.device = torch.device("cuda")
            self.handle = nvmlDeviceGetHandleByIndex(0)  # Track GPU memory and power
        else:
            self.device = torch.device("cpu")
            self.handle = None  # No GPU tracking on CPU

    def _get_gpu_memory_usage(self):
        """Get GPU memory usage in MB."""
        if self.device.type == "cuda":
            info = nvmlDeviceGetMemoryInfo(self.handle)
            return round(info.used / (1024 ** 2), 2)  # Convert bytes to MB
        return 0

    def _get_gpu_power_usage(self):
        """Get GPU power usage in Watts."""
        if self.device.type == "cuda":
            return nvmlDeviceGetPowerUsage(self.handle) / 1000  # Convert mW to W
        return 0

    def generate_once(self, prompt: str):
        """Generate text once and time it."""
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
            generated_text = self.model.generate(prompt, max_tokens=self.max_tokens)

        else:
            raise ValueError("Unsupported backend. Choose 'huggingface' or 'vllm'.")

        end_time = time.time()
        generation_time = end_time - start_time


        print(json.dumps(generated_text, indent=2))  # Pretty-print with indentation


        return generated_text, generation_time

    def measure_storage(self):
        """Measure GPU memory usage and model storage size."""
        mem_usage = self._get_gpu_memory_usage()

        try:
            model_path = hf_hub_download(self.model.config._name_or_path, filename="config.json", repo_type="model")
            model_dir = os.path.dirname(model_path)
            model_size = sum(
                f.stat().st_size for f in os.scandir(model_dir) if f.is_file()
            ) / (1024 ** 2)  # MB
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
        """Measure energy consumption based on generation time."""
        power_watts = self._get_gpu_power_usage()
        total_energy_wh = (power_watts * generation_time) / 3600  # W*s to Wh

        energy_per_token = (
            total_energy_wh * 3600 / num_tokens if num_tokens > 0 else 0
        )
        energy_per_sentence = (
            total_energy_wh * 3600 / num_sentences if num_sentences > 0 else 0
        )
        energy_per_second = power_watts  # Instantaneous power consumption in Watts

        return {
            "Total Energy (Wh)": round(total_energy_wh, 6),
            "Energy per Token (J/token)": round(energy_per_token, 6),
            "Energy per Sentence (J/sentence)": round(energy_per_sentence, 6),
            "Energy per Second (W)": round(energy_per_second, 6)
        }

    def benchmark(self, prompts):
        """Run the full benchmark for Hugging Face and vLLM models."""
        results = []

        for prompt in prompts:
            print(f"Evaluating prompt (length {len(prompt)} characters)...")

            # Step 1: Generate text once and time it
            generated_text, generation_time = self.generate_once(prompt)

            # Step 2: Count tokens and sentences
            if self.backend == "huggingface":
                num_tokens = len(self.tokenizer(generated_text).input_ids)
            elif self.backend == "vllm":
                num_tokens = len(generated_text.split())
            else:
                num_tokens = 0

            num_sentences = generated_text.count('.') + 1

            # Step 3: Calculate latency
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

            # Step 4: Calculate throughput
            TPS = num_tokens / generation_time if generation_time > 0 else 0
            SPS = num_sentences / generation_time if generation_time > 0 else 0
            throughput = {
                "TPS (tokens/s)": round(TPS, 2),
                "SPS (sentences/s)": round(SPS, 2)
            }

            # Step 5: Measure memory and model storage
            storage = self.measure_storage()

            # Step 6: Measure energy usage
            energy = self.measure_energy(num_tokens, num_sentences, generation_time)

            # Combine all results
            result = {
                "Prompt Length": len(prompt),
                **latency,
                **throughput,
                **storage,
                **energy
            }

            results.append(result)

        return pd.DataFrame(results)