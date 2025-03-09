import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage
import time
import pandas as pd
import os
from huggingface_hub import hf_hub_download

class ModelBenchmark:
    def __init__(self, model, tokenizer=None, max_tokens: int = 512, backend="huggingface"):
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
        """Get GPU memory usage in MB for both HF and vLLM."""
        if self.device.type == "cuda":
            info = nvmlDeviceGetMemoryInfo(self.handle)
            return round(info.used / (1024 ** 2), 2)  # Convert bytes to MB
        return 0  # No GPU memory usage on CPU

    def _get_gpu_power_usage(self):
        """Get GPU power usage in Watts."""
        if self.device.type == "cuda":
            return nvmlDeviceGetPowerUsage(self.handle) / 1000  # Convert mW to W
        return 0

    def measure_latency(self, prompt: str):
        """Measure FTL, ATL, and GL for both HF and vLLM."""
        start_time = time.time()

        if self.backend == "huggingface":
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_tokens, 
                pad_token_id=self.tokenizer.eos_token_id  # Explicitly set pad_token_id
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        else:  # vLLM Backend
            generated_text = self.model.generate(prompt, max_tokens=self.max_tokens)

        end_time = time.time()
        total_time = end_time - start_time

        num_tokens = len(generated_text.split())  # Approximate token count
        FTL = total_time / num_tokens if num_tokens > 0 else total_time
        ATL = (total_time - FTL) / (num_tokens - 1) if num_tokens > 1 else 0

        return {"FTL (s)": round(FTL, 4), "ATL (s)": round(ATL, 4), "GL (s)": round(total_time, 4)}

    def measure_throughput(self, prompt: str, latency):
        """Measure TPS (tokens/sec) and SPS (sentences/sec) based on real token count."""
        
        if self.backend == "huggingface":
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_tokens, 
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            num_tokens = len(self.tokenizer(generated_text).input_ids)  # Get actual token count

        elif self.backend == "vllm":
            generated_text = self.model.generate(prompt, max_tokens=self.max_tokens)
            num_tokens = len(generated_text.split())  # Approximate token count from words

        else:
            raise ValueError("Unsupported backend. Choose 'huggingface' or 'vllm'.")

        num_sentences = generated_text.count('.') + 1  # Approximate number of sentences

        TPS = num_tokens / latency["GL (s)"]
        SPS = num_sentences / latency["GL (s)"]

        return {"TPS (tokens/s)": round(TPS, 2), "SPS (sentences/s)": round(SPS, 2)}


    def measure_storage(self):
        """Measure GPU memory and model storage."""
        mem_usage = self._get_gpu_memory_usage()

        try:
            model_path = hf_hub_download(self.model.config._name_or_path, filename="config.json", repo_type="model")
            model_dir = os.path.dirname(model_path)
            model_size = sum(f.stat().st_size for f in os.scandir(model_dir) if f.is_file()) / (1024 ** 2)  # MB
        except Exception:
            model_size = "N/A"

        return {
            "Memory Usage (MB)": mem_usage,
            "Model Size (MB)": model_size,
            "KV-Cache Size Estimation (MB)": mem_usage - model_size if isinstance(model_size, float) else "N/A"
        }

    def measure_energy(self, num_tokens, num_sentences, generation_time):
        """Measure energy per token, per sentence, and per second using GPU power data."""
        power_watts = self._get_gpu_power_usage()
        total_energy_wh = (power_watts * generation_time) / 3600  # Convert W*s to Wh

        energy_per_token = total_energy_wh * 3600 / num_tokens if num_tokens > 0 else 0
        energy_per_sentence = total_energy_wh * 3600 / num_sentences if num_sentences > 0 else 0
        energy_per_second = power_watts  # Power consumption in Watts

        return {
            "Total Energy (Wh)": round(total_energy_wh, 6),
            "Energy per Token (J/token)": round(energy_per_token, 6),
            "Energy per Sentence (J/sentence)": round(energy_per_sentence, 6),
            "Energy per Second (W)": round(energy_per_second, 6)
        }

    def benchmark(self, prompts):
        """Run the full benchmark for HF and vLLM."""
        results = []

        for prompt in prompts:
            print(f"Evaluating prompt (length {len(prompt)} characters)...")
            latency = self.measure_latency(prompt)
            throughput = self.measure_throughput(prompt, latency)
            storage = self.measure_storage()

            num_tokens = throughput["TPS (tokens/s)"] * latency["GL (s)"]
            num_sentences = throughput["SPS (sentences/s)"] * latency["GL (s)"]
            generation_time = latency["GL (s)"]

            energy = self.measure_energy(num_tokens, num_sentences, generation_time)

            results.append({
                "Prompt Length": len(prompt),
                **latency,
                **throughput,
                **storage,
                **energy
            })

        return pd.DataFrame(results)
