import torch
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPowerUsage,
    nvmlShutdown,
    nvmlDeviceGetUtilizationRates,
)
import time
import pandas as pd
import os
import torch

from benchmark.backends.backend_factory import get_backend


class ModelBenchmark:
    def __init__(
        self,
        backend="huggingface",
        task="summarization",
        model_name="",
        model_path=None,
        max_tokens=256,
        quantization=None,
        verbose=False
    ):
        self.backend = backend
        self.task = task
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.verbose = verbose

        self.backend_handler = get_backend(
            name=backend,
            model_path=model_path,
            quantization=quantization,
            max_tokens=max_tokens,
            verbose=verbose
        )
        self.backend_handler.load_model()
        self.model_size = self.get_model_size_mb(model_path)

        # GPU monitoring
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
            info = nvmlDeviceGetPowerUsage(self.handle)
            return round(info / 1000.0, 2)
        return 0

    def _get_gpu_utilization(self):
        """Get GPU utilization percentage."""
        if self.device == "cuda" and self.handle:
            info = nvmlDeviceGetUtilizationRates(self.handle)
            return round(info.gpu, 2)


    def close(self):
        """Shutdown NVML (clean-up)."""
        if self.device == "cuda":
            nvmlShutdown()
    

    @staticmethod
    def get_model_size_mb(path):
        """
        Calculate model size in MB. Requires the model path.
        - If `path` is a GGUF file, return its size.
        - If `path` is a directory, sum safetensors, json, tokenizer, etc.
        """
        if os.path.isfile(path) and path.endswith(".gguf"):
            return os.path.getsize(path) / (1024 * 1024)

        elif os.path.isdir(path):
            extensions = (".safetensors", ".bin", ".json", ".txt", ".model", ".tokenizer")
            total_size = 0
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    if f.endswith(extensions):
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total_size += os.path.getsize(fp)
            return total_size / (1024 * 1024)

        else:
            raise ValueError("Path must be a .gguf file or a model directory.")


    def measure_storage(self):
        """Estimate memory + KV cache + model size for HuggingFace (optional for llama.cpp)."""
        mem_usage = self._get_gpu_memory_usage()
        model_size = self.model_size

        return {
            "Memory Usage (MB)": mem_usage,
            "Model Size (MB)": model_size,
            "Overhead (MB)": mem_usage - model_size if mem_usage > model_size else 0,
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

    def measure_gpu_utilization(self):
        """Measure GPU utilization percentage."""
        if self.device == "cuda" and self.handle:
            utilization = self._get_gpu_utilization()
            return {
                "GPU_Utilization (%)": utilization,
            }
        return {}


    def generate_single(self, prompt: str, task=None):
        """Generates text and measures generation time."""

        if task is None:
            task = self.task

        start_time = time.time()

        generated_text = self.backend_handler.generate(prompt, task_type=task)

        end_time = time.time()
        generation_time = end_time - start_time

        return generated_text, generation_time
    

    def run(self, samples=100):
        results = []

        if self.task == "summarization":
            from benchmark.tasks.summarization import SummarizationTask
            task_ = SummarizationTask()
        elif self.task == "qa":
            from benchmark.tasks.qa import QATask
            task_ = QATask()
        elif self.task == "sql":
            from benchmark.tasks.sql import SQLTask
            task_ = SQLTask()
        else:
            raise ValueError(f"Task {self.task} not supported.")

        prompts, references = task_.generate_prompts(num_examples=samples)

        for i, prompt in enumerate(prompts):
            if self.verbose:
                print(f"\nEvaluating prompt ({len(prompt)} characters)...")

            # Step 1: Generate text and measure generation time
            generated_text, generation_time = self.generate_single(prompt)

            # Clean the generated text
            generated_text = task_.clean_prediction(generated_text)
            if self.verbose:
                print(f"Generated text:\n{generated_text}\n")
                print(f"Reference text:\n{references[i]}\n")

            # Step 2: Count tokens and sentences
            num_tokens = len(generated_text.split())
            num_sentences = generated_text.count('.') + 1

            # Step 3: Latency metrics
            TTFT = self.backend_handler.measure_ttft()
            ATL = generation_time / (num_tokens - 1) if num_tokens > 1 else 0

            latency = {
                "TTFT": round(TTFT, 4),
                "ATL": round(ATL, 4),
                "GL": round(generation_time, 4)
            }

            # Step 4: Throughput
            TPS = num_tokens / generation_time if generation_time > 0 else 0
            SPS = num_sentences / generation_time if generation_time > 0 else 0

            throughput = {
                "TPS": round(TPS, 2),
                "SPS": round(SPS, 2)
            }

            # Step 5: Storage & energy
            storage = self.measure_storage()
            gpu_utilization = self.measure_gpu_utilization()
            energy = self.measure_energy(num_tokens, num_sentences, generation_time)

            # Step 6: Quality metrics
            reference = references[i] if isinstance(references, list) else references
            quality_metrics = task_.quality_metrics(generated_text, reference)

            # Step 7: Collect results
            result = {
                "prompt_length": len(prompt),
                "prompt": prompt,
                "generated_answer": generated_text,
                "reference_answer": reference,
                **latency,
                **throughput,
                **storage,
                **gpu_utilization,
                **energy,
                **quality_metrics
            }

            results.append(result)

            if self.verbose:
                print(f"\nResults for this prompt:\n{result}")
            
            torch.cuda.empty_cache()

        results = pd.DataFrame(results)

        # Compute statistics
        numeric_results = results.select_dtypes(include='number')
        averages = numeric_results.mean()
        stds = numeric_results.std()

        # Combine mean ± std into a formatted string
        summary = averages.combine(stds, lambda mean, std: f"{mean:.6f} ± {std:.6f}")

        # Print formatted summary
        print(f"Statistics (mean ± std) for {self.model_name}, with {self.backend}, for task: {self.task}:")
        print(summary)

        results.to_csv(f"/home/ubuntu/fast_llm_inference/results/{self.backend}_{self.model_name}_{self.task}.csv", index=False)

        self.backend_handler = None
        if self.device == "cuda":
            nvmlShutdown()

        return results