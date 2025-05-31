import time
import requests 
import threading
import os
import torch
import numpy as np
import torch
from typing import Optional
import random
import multiprocessing as mp
import psutil
import subprocess
import math
import pandas as pd
from difflib import get_close_matches
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPowerUsage,
    nvmlShutdown,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetComputeRunningProcesses
)

from benchmark.inference_engine_client import InferenceEngineClient
from benchmark.utils import clean_prediction, tok_cnt, sent_cnt, chunker

DEFAULT_SEED = 42
random.seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
torch.manual_seed(DEFAULT_SEED)

class ModelBenchmark:
    def __init__(
        self,
        backend="tgi",
        model_name="",
        model_path=None,
        model_size=None,
        verbose=False
    ):
        
        if backend not in ("tgi", "mii", "sglang", "vllm", "lmdeploy"):
            raise ValueError(f"Unsupported backend: {backend}. Supported backends are: tgi, mii, sglang, vllm, lmdeploy.")
        self.backend = backend
        self.model_path = model_path
        self.model_name = model_name
        self.model_size = model_size  # in MB, if known
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.verbose = verbose
        self.iec = InferenceEngineClient()

        # GPU monitoring
        if torch.cuda.is_available():
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(0)
        else:
            self.handle = None
        self._init_cpu_monitor_from_gpu(device_index=0)

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


    def _init_cpu_monitor_from_gpu(self, device_index: int = 0):
        """
        Initialize NVML (once), find the PID holding the most GPU memory on
        `device_index`, and set up a psutil.Process for cpu_percent() sampling.
        Falls back to the current process if NVML fails or finds no compute procs.
        """

        try:
            handle    = nvmlDeviceGetHandleByIndex(device_index)
            gpu_procs = nvmlDeviceGetComputeRunningProcesses(handle)
            if gpu_procs:
                pid = max(gpu_procs, key=lambda p: p.usedGpuMemory).pid
            else:
                # no one is on the GPU yet → monitor this process
                pid = os.getpid()
        except Exception:
            # any NVML error → monitor this process
            pid = os.getpid()

        # stash the psutil handle
        self._ps_proc = psutil.Process(pid)
        # prime the counter (first call returns 0.0)
        self._ps_proc.cpu_percent(None)

    def _get_cpu_utilization(self) -> float:
        """
        Return the %CPU for the monitored process since the last call.
        Raises if you forgot to call _init_cpu_monitor_from_gpu.
        """
        if not hasattr(self, "_ps_proc"):
            raise RuntimeError(
                "CPU monitor not initialized – please call "
                "_init_cpu_monitor_from_gpu() before sampling."
            )

        raw_pct = self._ps_proc.cpu_percent(None)
        return raw_pct / psutil.cpu_count()


    def _metrics_monitor(self, readings: dict, stop_evt: threading.Event, sample_interval: float):
            """
            Background worker: every `sample_interval` seconds, append readings
            for memory (MB), power (W) and util (%) until `stop_evt` is set.
            """
            while not stop_evt.is_set():
                readings["memory"].append(self._get_gpu_memory_usage())
                readings["power"].append(self._get_gpu_power_usage())
                readings["util"].append(self._get_gpu_utilization())
                readings["cpu"].append(self._get_cpu_utilization())
                time.sleep(sample_interval)

    def close(self):
        if self.device == "cuda" and self.handle:
            nvmlShutdown()
            self.handle = None  # prevent double‐shutdown


    def get_model_size(self, path):
        """
        Calculate model size in MB. Requires the model path.
        - If `path` is a GGUF file, return its size.
        - If `path` is a directory, sum safetensors, json, tokenizer, etc.
        """
        if self.model_size is None:

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
        else:
            # If model_size_mb is already set, return it
            return self.model_size


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


    def generate(self, prompts):
        start_time = time.time()
        out = self.iec.completion(prompts)
        return out, time.time() - start_time

    @staticmethod
    def estimate_local_query_cost(inf_time_sec: float,
                                power_watts: float = 0.0,
                                electricity_usd_per_kwh: float = 0.31, # swiss average in usd
                                csv_path: str = "/home/ubuntu/fast_llm_inference/benchmark/lookup/nvidia_llm_gpus.csv") -> dict:
        """
        Estimate amortization + energy cost for a single LLM query.
        Uses GPU name + VRAM from nvidia-smi and fuzzy matches CSV table.
        """
        # 1) Load table
        df = pd.read_csv(csv_path)

        # 2) Get GPU info
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        name, mem = [x.strip() for x in out.split(",")]
        vram = round(math.ceil(float(mem) / 1024))

        # 3) Match row
        subset = df[df["VRAM"].between(vram - 2, vram + 2)]
        if subset.empty:
            raise ValueError(f"No GPU match for VRAM={vram} GB")
        names = subset["GPU"].tolist()
        match = next((n for n in names if n.lower() == name.lower()), 
                next((n for n in names if n.lower().startswith(name.lower())), 
                next(iter(get_close_matches(name, names, n=1, cutoff=0.4)), None)))
        if not match:
            return None  # no match found

        # 4) Cost computation
        amort_hr = subset[subset["GPU"] == match].iloc[0]["Amortization_USD_hr"]
        amort_cost = (inf_time_sec / 3600) * amort_hr
        energy_cost = (power_watts * inf_time_sec / 3.6e6) * electricity_usd_per_kwh if power_watts else 0.0

        return round(amort_cost + energy_cost, 6)

    def _initialize(self):
        """
        Instead of instantiating a local backend handler, this version
        spins up the inference engine via the `launch_engine.sh` script
        (which now only launches the container) and then polls the
        /v1/completions endpoint every second until it returns HTTP 200.
        We measure how long it takes for the server to become ready,
        then record model size and metadata.
        """

        # Ensure our script is executable
        

        # 0.a) Launch the engine via bash script (non-blocking)
        t0 = time.time()
        self.iec.launch(backend=self.backend, model_path=self.model_path)

        # 0.c) Compute elapsed time for startup 
        startup = time.time() - t0

        # 0.d) Compute model size
        #TODO: Uncomment when model size is available
        self.model_size = 0 #self.get_model_size(self.model_path) 

        # 0.e) TTFT already covered by first poll, set to 0
        ttft = self.iec.measure_ttft()

        # Build metadata row
        meta = {
            "model_name":           self.model_name,
            "model_size_mb":        self.model_size,
            "backend":              self.backend,
            "startup":              round(startup, 4),
            "ttft_sec":             round(ttft, 4),
            "coldstart":            round(startup + ttft, 4)
        }

        return pd.DataFrame([meta])


    def _prepare_prompts(self, task_, scenario, samples, run_time, rps):
        """
        task_: an already-instantiated Task object (SummarizationTask, QATask, or SQLTask)
        Returns prompts, references, and scheduled timestamps list.
        """
        import numpy as np

        # batch/single: no scheduling
        if scenario in ("single", "batch"):
            prompts, refs = task_.generate_prompts(num_examples=samples)
            sched_ts = [0.0] * len(prompts)
        else:  # server
            est = int(rps * run_time * 2) + 10
            ia = np.random.exponential(scale=1.0/rps, size=est)
            arrivals = np.cumsum(ia)
            arrivals = arrivals[arrivals <= run_time]
            prompts, refs = task_.generate_prompts(num_examples=len(arrivals))
            sched_ts = list(arrivals)

        return prompts, refs, sched_ts


    def _batch_generator(self, prompts, refs, sched_ts, scenario, batch_size, max_batch):

        if scenario in ("single", "batch"):
            for batch_id, (ps, rs) in enumerate(zip(
                chunker(prompts, batch_size),
                chunker(refs,   batch_size)
            )):
                yield batch_id, ps, rs, [0.0] * len(ps)
        else:  # server
            queue = list(zip(prompts, refs, sched_ts))
            batch_id = 0
            while queue:
                batch_id += 1
                batch = queue if max_batch is None else queue[:max_batch]
                queue = queue[len(batch):]
                ps, rs, ts = zip(*batch)
                yield batch_id, list(ps), list(rs), list(ts)


    def _run_scenario(
        self,
        task: str,
        scenario: str = "batch",
        samples: int = 100,
        batch_size: int = 1,
        run_time: Optional[float] = None,
        requests_per_sec: Optional[float] = None,
        max_batch_size: Optional[int] = None,
        sample_interval: float = 0.1,
        quality_metric: bool = True
    ):
        # ─── 1) instantiate the Task once ────────────────────────────────
        if task == "summarization":
            from benchmark.tasks.summarization import SummarizationTask
            task_ = SummarizationTask()
        elif task == "qa":
            from benchmark.tasks.qa import QATask
            task_ = QATask()
        elif task == "sql":
            from benchmark.tasks.sql import SQLTask
            task_ = SQLTask(tables_path="fast_llm_inference/benchmark/lookup/tables.json")
        else:
            raise ValueError(f"Task {task!r} not supported.")

        # ─── 2) initialization & metadata ────────────────────────────────
        meta_df = self._initialize()
        total_gen_time = 0.0
        energy_wh_total = 0.0

        # Start telemetry monitor
        global_readings = {"memory": [], "power": [], "util": [], "cpu": []}
        stop_evt = threading.Event()
        mon = threading.Thread(
            target=self._metrics_monitor,
            args=(global_readings, stop_evt, sample_interval),
            daemon=True
        )
        mon.start()

        # scenario-specific fields
        extra = {}
        if scenario == "server":
            extra["requests_per_sec"] = requests_per_sec
            extra["run_time_s"] = run_time
        else:
            extra["batch_size"] = batch_size
            extra["num_queries"] = samples
        meta_df = meta_df.assign(**extra)

        # ─── 3) prompts + scheduled timestamps ───────────────────────────
        prompts, refs, sched_ts = self._prepare_prompts(
            task_, scenario, samples, run_time, requests_per_sec
        )

        query_records = []

        # ─── 4) iterate batches ──────────────────────────────────────────
        for batch_id, ps, rs, ts in self._batch_generator(
            prompts, refs, sched_ts,
            scenario, batch_size, max_batch_size
        ):
            # generation + clean
            raw, gen_time = self.generate(ps)
            
            cleaned = clean_prediction(raw)

            # accumulate global metrics
            total_gen_time += gen_time
            avg_power = sum(global_readings["power"]) / len(global_readings["power"]) if global_readings["power"] else 0
            energy_wh_total += (avg_power * gen_time) / 3600

            # token/sentence counts
            tok_counts = [tok_cnt(t) for t in cleaned]
            sent_counts = [sent_cnt(t) for t in cleaned]
            total_tokens = sum(tok_counts)
            total_sentences = sum(sent_counts)

            # build per-query records
            for i, (p, pred, ref, sched) in enumerate(zip(ps, cleaned, rs, ts)):
                nt = tok_counts[i]
                ns = sent_counts[i]
                atl = gen_time / total_tokens if total_tokens else 0
                gl = atl * nt
                tps = nt / gl if gl else 0
                sps = ns / gl if gl else 0
                wait = max(0.0, time.time() - sched) if scenario == "server" else 0.0

                quality = task_.quality_metrics(pred, ref) if quality_metric else {}

                qm = {
                    "prompt": p,
                    "generated_answer": pred,
                    "reference_answer": ref,
                    "ATL": round(atl, 6),
                    "GL": round(gl, 6),
                    "TPS": round(tps, 2),
                    "SPS": round(sps, 2),
                    "energy_per_token": round((avg_power * gen_time) / total_tokens, 6) if total_tokens else 0,
                    "energy_per_sentence": round((avg_power * gen_time) / total_sentences, 6) if total_sentences else 0,
                    "estimated_query_cost_usd": self.estimate_local_query_cost(
                        inf_time_sec=gen_time / batch_size,
                        power_watts=avg_power
                    ),
                    **quality
                }

                if scenario == "server":
                    qm.update({
                        "scheduled_ts": sched,
                        "wait_time_s": round(wait, 6),
                    })

                query_records.append(qm)

        # stop telemetry after all batches
        stop_evt.set()
        mon.join()

        # ─── 5) assemble outputs ──────────────────────────────────────────
        details_df = pd.DataFrame(query_records)

        # compute overall metrics
        if global_readings["memory"]:
            avg_gpu_mem_mb = round(sum(global_readings["memory"]) / len(global_readings["memory"]), 2)
            peak_gpu_mem_mb = round(max(global_readings["memory"]), 2)
        else:
            avg_gpu_mem_mb = peak_gpu_mem_mb = 0

        overhead_mb = round(max(peak_gpu_mem_mb - self.model_size, 0), 2)

        if global_readings["util"]:
            avg_gpu_util_pct = round(sum(global_readings["util"]) / len(global_readings["util"]), 2)
            peak_gpu_util_pct = round(max(global_readings["util"]), 2)
        else:
            avg_gpu_util_pct = peak_gpu_util_pct = 0

        if global_readings["cpu"]:
            avg_cpu_util_pct = round(sum(global_readings["cpu"]) / len(global_readings["cpu"]), 2)
            peak_cpu_util_pct = round(max(global_readings["cpu"]), 2)
        else:
            avg_cpu_util_pct = peak_cpu_util_pct = 0

        if global_readings["power"]:
            avg_power_w = round(sum(global_readings["power"]) / len(global_readings["power"]), 2)
            peak_power_w = round(max(global_readings["power"]), 2)
        else:
            avg_power_w = peak_power_w = 0

        total_energy_wh = round(energy_wh_total, 6)

        # finalize report
        run_report = meta_df.copy().drop(columns=["batch_id"], errors="ignore")
        run_report["total_generation_time_s"] = round(total_gen_time, 6)
        run_report["avg_gpu_mem_mb"] = avg_gpu_mem_mb
        run_report["peak_gpu_mem_mb"] = peak_gpu_mem_mb
        run_report["overhead_mb"] = overhead_mb
        run_report["avg_gpu_util_pct"] = avg_gpu_util_pct
        run_report["peak_gpu_util_pct"] = peak_gpu_util_pct
        run_report["avg_cpu_util_pct"] = avg_cpu_util_pct
        run_report["peak_cpu_util_pct"] = peak_cpu_util_pct
        run_report["avg_power_w"] = avg_power_w
        run_report["peak_power_w"] = peak_power_w
        run_report["total_energy_wh"] = total_energy_wh

        # add average values from details report
        if not details_df.empty:
            detail_nums = details_df.select_dtypes(include="number")
            for col in detail_nums.columns:
                run_report[f"avg_{col}"] = round(detail_nums[col].mean(), 6)

        self.iec.close()

        return run_report, details_df



    def run(self, *, task, scenario, samples=None, batch_size=None,
            run_time=None, requests_per_sec=None, max_batch_size=None,
            sample_interval=0.1, quality_metric=True):
        """
        Public entrypoint. Dispatches to the unified _run_scenario.
        """
        return self._run_scenario(
            task=task,
            scenario=scenario,
            samples=samples,
            batch_size=batch_size,
            run_time=run_time,
            requests_per_sec=requests_per_sec,
            max_batch_size=max_batch_size,
            sample_interval=sample_interval,
            quality_metric=quality_metric
        )

