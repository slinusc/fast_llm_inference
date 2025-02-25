import torch
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo
)
import time
import pandas as pd
import evaluate
import subprocess
import os
from IPython.display import display, Markdown


class ModelBenchmark:
    def __init__(self, model, tokenizer, max_tokens: int = 512, sampling_rate: float = 0.2):
        """
        Initialize the benchmarking class with an already loaded model and tokenizer.

        :param model: Pre-loaded Hugging Face model.
        :param tokenizer: Pre-loaded tokenizer corresponding to the model.
        :param max_tokens: Maximum tokens to generate per prompt.
        :param sampling_rate: Sampling interval in seconds for power logging.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.sampling_rate = sampling_rate
        self.power_log_file = "power_log.csv"
        self.power_logging_process = None

        # Initialize NVML for GPU tracking
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)

    def _start_power_logging(self):
        """Start logging GPU power usage using nvidia-smi for a single prompt."""
        if os.path.exists(self.power_log_file):
            os.remove(self.power_log_file)
        self.power_logging_process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,nounits,noheader",
                "-lms",
                str(int(self.sampling_rate * 1000))  # Convert seconds to milliseconds
            ],
            stdout=open(self.power_log_file, "w"),
            stderr=subprocess.DEVNULL
        )
        print("Power logging started for prompt.")

    def _stop_power_logging(self):
        """Stop GPU power usage logging for a single prompt."""
        if self.power_logging_process:
            self.power_logging_process.terminate()
            self.power_logging_process.wait()
            self.power_logging_process = None
            print("Power logging stopped for prompt.")

    def _calculate_energy_from_log(self):
        """Calculate total energy consumption from the logged power data for a single prompt."""
        if not os.path.exists(self.power_log_file):
            return 0

        with open(self.power_log_file, "r") as file:
            lines = file.readlines()
            power_readings = [float(line.strip()) for line in lines if line.strip()]

        avg_power = sum(power_readings) / len(power_readings) if power_readings else 0
        total_time = len(power_readings) * self.sampling_rate
        total_energy_wh = (avg_power * total_time) / 3600  # Convert W*s to Wh
        return round(total_energy_wh, 6)

    def measure_latency(self, prompt: str):
        """Compute latency metrics: FTL, ATL, and GL."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        start_time = time.time()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )
        end_time = time.time()

        total_time = end_time - start_time
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        num_tokens = len(self.tokenizer(generated_text).input_ids)

        FTL = total_time / num_tokens if num_tokens > 0 else total_time
        ATL = (total_time - FTL) / (num_tokens - 1) if num_tokens > 1 else 0

        return {"FTL (s)": round(FTL, 4), "ATL (s)": round(ATL, 4), "GL (s)": round(total_time, 4)}

    def measure_throughput(self, prompt: str):
        """Compute throughput metrics: TPS and SPS."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        start_time = time.time()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )
        end_time = time.time()

        total_time = end_time - start_time
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        num_tokens = len(self.tokenizer(generated_text).input_ids)
        num_sentences = generated_text.count('.')

        TPS = num_tokens / total_time
        SPS = num_sentences / total_time

        return {"TPS (tokens/s)": round(TPS, 2), "SPS (sentences/s)": round(SPS, 2)}

    def measure_storage(self):
        """Measure storage utilization including model memory and KV-cache size."""
        mem_info = nvmlDeviceGetMemoryInfo(self.handle)
        memory_usage = mem_info.used / 1024 ** 2
        return {"Memory Usage (MB)": round(memory_usage, 2)}

    def measure_energy(self):
        """Calculate energy consumption from the power log for a single prompt."""
        total_energy_wh = self._calculate_energy_from_log()
        return {"Total Energy Consumption (Wh)": total_energy_wh}

    def measure_quality(self, predictions, references):
        """Compute ROUGE score for summarization tasks."""
        rouge = evaluate.load("rouge")
        scores = rouge.compute(predictions=predictions, references=references)
        return scores

    def benchmark(self, prompts):
        """
        Run full benchmark on a list of prompts with per-prompt power logging.

        :param prompts: List of prompt strings.
        :return: DataFrame with all metrics combined.
        """
        results = []
        for prompt in prompts:
            print(f"Evaluating prompt (length {len(prompt)} characters)...")
            self._start_power_logging()
            latency = self.measure_latency(prompt)
            throughput = self.measure_throughput(prompt)
            storage = self.measure_storage()
            self._stop_power_logging()
            energy = self.measure_energy()

            results.append({
                "Prompt Length": len(prompt),
                **latency,
                **throughput,
                **storage,
                **energy
            })

        df_results = pd.DataFrame(results)
        return df_results

    def display_results(self, df: pd.DataFrame):
        """Display benchmark results as a Markdown table in Jupyter Notebook."""
        display(Markdown(df.to_markdown(index=False)))

    def save_results(self, df: pd.DataFrame, file_path: str):
        """Save benchmark results to a CSV file."""
        df.to_csv(file_path, index=False)
        print(f"Results saved to {file_path}")


if __name__ == "__main__":
    pass
