from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPowerUsage,
    nvmlShutdown,
    nvmlDeviceGetUtilizationRates,
)
import time
import threading   
import pandas as pd
import os
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from typing import Optional
import random

from benchmark.backends.backend_factory import get_backend
from benchmark.utils import clean_prediction

DEFAULT_SEED = 42
random.seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
torch.manual_seed(DEFAULT_SEED)

class ModelBenchmark:
    def __init__(
        self,
        backend="huggingface",
        task="summarization",
        model_name="",
        model_path=None,
        base_path="/home/ubuntu/fast_llm_inference/",
        verbose=False
    ):
        self.backend = backend
        self.task = task
        self.model_name = model_name
        self.base_path = base_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.verbose = verbose

        self.backend_handler = get_backend(
            name=backend,
            model_path=model_path,
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
        """Return current GPU memory used (in MB)."""
        if self.device == "cuda" and self.handle:
            info = nvmlDeviceGetMemoryInfo(self.handle)
            return round(info.used / (1024 ** 2), 2)
        return 0.0

    def _get_gpu_power_usage(self):
        """Return current GPU power draw (in W)."""
        if self.device == "cuda" and self.handle:
            # nvmlDeviceGetPowerUsage returns milliwatts
            power_mw = nvmlDeviceGetPowerUsage(self.handle)
            return round(power_mw / 1000.0, 2)
        return 0.0

    def _get_gpu_utilization(self):
        """Return current GPU utilization percentage."""
        if self.device == "cuda" and self.handle:
            util = nvmlDeviceGetUtilizationRates(self.handle)
            return util.gpu  # the .gpu field is a percentage
        return 0.0

    def _metrics_monitor(self, readings: dict, stop_evt: threading.Event, sample_interval: float):
            """
            Background worker: every `sample_interval` seconds, append readings
            for memory (MB), power (W) and util (%) until `stop_evt` is set.
            """
            while not stop_evt.is_set():
                readings["memory"].append(self._get_gpu_memory_usage())
                readings["power"].append(self._get_gpu_power_usage())
                readings["util"].append(self._get_gpu_utilization())
                time.sleep(sample_interval)

    def close(self):
        if self.device == "cuda" and self.handle:
            nvmlShutdown()
            self.handle = None  # prevent double‐shutdown

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


    def measure_energy(self, num_tokens, num_sentences, generation_time, sample_interval=0.1):
        """
        Sample GPU power every `sample_interval` seconds over the duration of
        `generation_time`, then compute:
        - Total Energy (Wh)
        - Energy per Token (J/token)
        - Energy per Sentence (J/sentence)
        - Avg Power (W)
        """
        readings = []
        start = time.time()
        while time.time() - start < generation_time:
            readings.append(self._get_gpu_power_usage())
            time.sleep(sample_interval)

        avg_power = sum(readings) / len(readings) if readings else 0.0
        # Total energy in Wh
        total_energy_wh = avg_power * generation_time / 3600

        # Convert Wh back to joules for per‐unit metrics: 1 Wh = 3600 J
        energy_per_token = (total_energy_wh * 3600 / num_tokens) if num_tokens > 0 else 0.0
        energy_per_sentence = (total_energy_wh * 3600 / num_sentences) if num_sentences > 0 else 0.0

        return {
            "Total Energy (Wh)": round(total_energy_wh, 6),
            "Avg Power (W)": round(avg_power, 2),
            "Energy per Token (J/token)": round(energy_per_token, 6),
            "Energy per Sentence (J/sentence)": round(energy_per_sentence, 6),
        }


    def measure_gpu_utilization(self):
        """Measure GPU utilization percentage."""
        if self.device == "cuda" and self.handle:
            utilization = self._get_gpu_utilization()
            return {
                "GPU_Utilization (%)": utilization,
            }
        return {}


    def generate(self, prompts, task_type=None):
        if task_type is None:
            task_type = self.task
        start_time = time.time()
        generated_text = self.backend_handler.generate(prompts, task_type=task_type)
        return generated_text, time.time() - start_time


    def _run_basic(self, samples=100, batch_size=1, sample_interval=0.1):
        results = []

        # 1) Prepare task
        if self.task == "summarization":
            from benchmark.tasks.summarization import SummarizationTask
            task_ = SummarizationTask()
        elif self.task == "qa":
            from benchmark.tasks.qa import QATask
            task_ = QATask()
        elif self.task == "sql":
            from benchmark.tasks.sql import SQLTask
            task_ = SQLTask(tables_path=self.base_path + "benchmark/tasks/tables.json")
        else:
            raise ValueError(f"Task {self.task} not supported.")

        # 2) Generate prompts & references
        prompts, references = task_.generate_prompts(num_examples=samples)

        # 3) Helper to chunk a list
        def chunker(seq, size):
            for i in range(0, len(seq), size):
                yield seq[i : i + size]

        # 4) Loop over prompt‐batches
        for subbatch_prompts, subbatch_refs in zip(
            chunker(prompts, batch_size),
            chunker(references, batch_size)
        ):
            if self.verbose:
                print(f"\nProcessing batch of {len(subbatch_prompts)} prompts...")

            # 5) Spin up background monitor
            readings = {"memory": [], "power": [], "util": []}
            stop_evt = threading.Event()
            monitor = threading.Thread(
                target=self._metrics_monitor,
                args=(readings, stop_evt, sample_interval),
                daemon=True
            )
            monitor.start()

            # 6) Generate the entire subbatch (generate() returns both outputs and elapsed time)
            generated_texts, generation_time = self.generate(
                subbatch_prompts,
                task_type=self.task
            )

            # 6.a) Clean the whole batch at once
            cleaned_texts = clean_prediction(generated_texts)

            # 7) Stop monitoring
            stop_evt.set()
            monitor.join()

            # 8) Aggregate GPU stats
            avg_mem = sum(readings["memory"]) / len(readings["memory"]) if readings["memory"] else 0.0
            peak_mem = max(readings["memory"]) if readings["memory"] else 0.0
            avg_power = sum(readings["power"]) / len(readings["power"]) if readings["power"] else 0.0
            peak_power = max(readings["power"]) if readings["power"] else 0.0
            avg_util  = sum(readings["util"])  / len(readings["util"])  if readings["util"]  else 0.0
            peak_util = max(readings["util"])  if readings["util"]  else 0.0

            total_wh = avg_power * generation_time / 3600.0
            joules_per_token = lambda n: (total_wh * 3600.0 / n) if n > 0 else 0.0
            joules_per_sentence = lambda s: (total_wh * 3600.0 / s) if s > 0 else 0.0

            # 9) Measure TTFT for this batch (if available)
            TTFT = (
                self.backend_handler.measure_ttft()
                if hasattr(self.backend_handler, "measure_ttft")
                else None
            )

            # 10) Unpack per-prompt results
            for prompt, pred, reference in zip(
                subbatch_prompts, cleaned_texts, subbatch_refs
            ):
                if self.verbose:
                    print(f"\nPrediction:\n{pred}\nRef:\n{reference}\n")

                num_tokens = len(pred.split())
                num_sentences = pred.count('.') + 1

                ATL = generation_time / num_tokens if num_tokens > 0 else 0
                TPS = num_tokens / generation_time if generation_time > 0 else 0
                SPS = num_sentences / generation_time if generation_time > 0 else 0

                latency = {
                    "TTFT": round(TTFT, 4) if TTFT is not None else None,
                    "ATL": round(ATL, 4),
                    "GL":  round(generation_time, 4),
                }
                throughput = {
                    "TPS": round(TPS, 2),
                    "SPS": round(SPS, 2),
                }
                gpu_metrics = {
                    "Avg GPU Mem (MB)": round(avg_mem, 2),
                    "Peak GPU Mem (MB)": round(peak_mem, 2),
                    "Avg GPU Util (%)":  round(avg_util, 2),
                    "Peak GPU Util (%)": round(peak_util, 2),
                }
                energy_metrics = {
                    "Total Energy (Wh)":             round(total_wh, 6),
                    "Avg Power (W)":                 round(avg_power, 2),
                    "Peak Power (W)":                round(peak_power, 2),
                    "Energy per Token (J/token)":    round(joules_per_token(num_tokens), 6),
                    "Energy per Sentence (J/sentence)": round(joules_per_sentence(num_sentences), 6),
                }
                storage = self.measure_storage()
                quality = task_.quality_metrics(pred, reference)

                result = {
                    "prompt_length": len(prompt),
                    "prompt": prompt,
                    "generated_answer": pred,
                    "reference_answer": reference,
                    **latency,
                    **throughput,
                    **gpu_metrics,
                    **energy_metrics,
                    **storage,
                    **quality,
                }
                results.append(result)
                torch.cuda.empty_cache()

        # 11) Aggregate to a DataFrame and summarize
        df = pd.DataFrame(results)
        numeric = df.select_dtypes(include="number")
        means = numeric.mean()
        stds = numeric.std()
        summary = means.combine(stds, lambda m, s: f"{m:.6f} ± {s:.6f}")

        print(f"Stats for {self.model_name} on {self.backend}/{self.task}:")
        print(summary)

        # 13) Cleanup
        self.backend_handler = None
        self.close()

        return df

    def _run_server(self,
                   run_time: float,
                   requests_per_sec: float,
                   sample_interval: float = 0.1,
                   max_batch_size: Optional[int] = None,
                   quality_metric: bool = False) -> pd.DataFrame:
        """
        Server Poisson simulation over a fixed run_time (s), stopping based
        on real wall-clock time rather than simulated offset. Returns a DataFrame.
        Each record will include a batch_id for grouping.
        """
        # 1) Task setup
        if self.task == "summarization":
            from benchmark.tasks.summarization import SummarizationTask
            task_ = SummarizationTask()
        elif self.task == "qa":
            from benchmark.tasks.qa import QATask
            task_ = QATask()
        elif self.task == "sql":
            from benchmark.tasks.sql import SQLTask
            task_ = SQLTask(tables_path=self.base_path + "/benchmark/tasks/tables.json")
        else:
            raise ValueError(f"Task {self.task} not supported.")

        # 2) Pre-sample Poisson arrivals
        est_n = int(requests_per_sec * run_time * 2) + 10
        inter_arrivals = np.random.exponential(scale=1.0/requests_per_sec, size=est_n)
        arrival_times = np.cumsum(inter_arrivals)
        arrival_times = arrival_times[arrival_times <= run_time]
        num_requests = len(arrival_times)

        # 3) Generate prompts & refs
        prompts, references = task_.generate_prompts(num_examples=num_requests)

        # 4) Simulation state
        t0          = time.time()
        deadline    = t0 + run_time
        next_idx    = 0
        queue       = []
        batch_id    = 0
        TTFT = (
            self.backend_handler.measure_ttft()
            if hasattr(self.backend_handler, "measure_ttft") else None
        )
        results     = []

        # 5) Real-time loop
        while time.time() < deadline:
            offset = time.time() - t0
            # a) Enqueue arrivals
            while next_idx < num_requests and arrival_times[next_idx] <= offset:
                queue.append((next_idx,
                              prompts[next_idx],
                              references[next_idx],
                              arrival_times[next_idx]))
                next_idx += 1

            if queue:
                # increment batch counter
                batch_id += 1
                # pull batch
                batch = queue if max_batch_size is None else queue[:max_batch_size]
                queue = queue[len(batch):]

                # start telemetry monitor
                readings = {"memory": [], "power": [], "util": []}
                evt = threading.Event()
                monitor = threading.Thread(
                    target=self._metrics_monitor,
                    args=(readings, evt, sample_interval),
                    daemon=True
                )
                monitor.start()

                # run generation
                prompts_batch = [e[1] for e in batch]
                outputs, gen_time = self.generate(prompts_batch, task_type=self.task)
                cleaned = clean_prediction(outputs)

                # stop monitor
                evt.set()
                monitor.join()

                # aggregate some metrics
                avg_power = sum(readings["power"]) / len(readings["power"]) if readings["power"] else 0
                avg_mem   = sum(readings["memory"]) / len(readings["memory"]) if readings["memory"] else 0
                avg_util  = sum(readings["util"]) / len(readings["util"]) if readings["util"] else 0

                # record per-request
                for (idx, _, ref, sched_ts), pred in zip(batch, cleaned):
                    start_ts      = offset
                    wait_time     = start_ts - sched_ts
                    response_time = wait_time + gen_time
                    num_tokens    = len(pred.split())
                    num_sentences = pred.count(".") + 1
                    ATL = gen_time / num_tokens if num_tokens > 0 else 0
                    TPS = num_tokens / gen_time if gen_time > 0 else 0
                    SPS = num_sentences / gen_time if gen_time > 0 else 0
                    quality = task_.quality_metrics(pred, ref) if quality_metric else {}

                    results.append({
                        "batch_id":              batch_id,
                        "prompt_length":         len(prompts[idx]),
                        "prompt":                prompts[idx],
                        "generated_answer":      pred,
                        "reference_answer":      ref,
                        "scheduled_ts":          round(sched_ts, 4),
                        "start_ts":              round(start_ts, 4),
                        "wait_time":             round(wait_time, 4),
                        "response_time":         round(response_time, 4),
                        "gen_latency":           round(gen_time, 4),
                        "ATL":                   round(ATL, 4),
                        "TTFT":                  round(TTFT, 4) if TTFT is not None else None,
                        "TPS":                   round(TPS, 2),
                        "SPS":                   round(SPS, 2),
                        "Avg GPU Mem (MB)":      round(avg_mem, 2),
                        "Avg GPU Util (%)":      round(avg_util, 2),
                        "Avg Power (W)":         round(avg_power, 2),
                        **self.measure_storage(),
                        **quality
                    })
                    torch.cuda.empty_cache()
            else:
                time.sleep(sample_interval)

        # 6) Build DataFrame and return
        df = pd.DataFrame(results).sort_values(["batch_id", "scheduled_ts"]).reset_index(drop=True)
        return df


    def run(self,
            samples: int = 100,
            batch_size: int = 1,
            sample_interval: float = 0.1,
            scenario: str = "batch",
            requests_per_sec: Optional[float] = None,
            run_time: Optional[float] = None,
            max_batch_size: int = 128,
            export_path: Optional[str] = None,
            quality_metric: bool = False
           ) -> pd.DataFrame:
        """
        Dispatch to the appropriate execution method, then handle exporting.
        """
        if scenario == "single":
            df = self._run_basic(
                samples=samples,
                batch_size=1,
                sample_interval=sample_interval
            )

        elif scenario == "batch":
            df = self._run_basic(
                samples=samples,
                batch_size=batch_size,
                sample_interval=sample_interval
            )

        elif scenario == "server":
            assert requests_per_sec is not None, \
                "Must set requests_per_sec in server mode"
            assert run_time is not None, \
                "Must set run_time in seconds in server mode"
            df = self._run_server(
                run_time=run_time,
                requests_per_sec=requests_per_sec,
                sample_interval=sample_interval,
                max_batch_size=max_batch_size,
                quality_metric=quality_metric
            )

        else:
            raise ValueError(f"Unknown scenario '{scenario}'. "
                             "Choose 'single', 'batch', or 'server'.")

        # Exporting logic: save DataFrame if path provided or default
        import os
        if export_path:
            out_path = export_path
        else:
            # construct a default filename
            out_path = (
                f"{self.base_path}/results/"
                f"{self.backend}_{self.model_name}_{self.task}_"
                f"{scenario}"
                + (f"_{int(requests_per_sec)}rps" if scenario == 'server' else '')
                + (f"_{batch_size}bs" if scenario == 'batch' else '')
                + ".csv"
            )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        return df