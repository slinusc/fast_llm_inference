import time
from datetime import datetime
import threading
from pathlib import Path
import os
import torch
import numpy as np
from typing import Optional
import random
import multiprocessing as mp
import concurrent.futures
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

# Setting a seed globally can affect downstream randomness; consider scoping in methods
DEFAULT_SEED = 42
random.seed(DEFAULT_SEED)
numpy_rng = np.random.default_rng(DEFAULT_SEED)
torch.manual_seed(DEFAULT_SEED)

PROJECT_ROOT = Path(os.getenv("FASTLLM_HOME", Path(__file__).resolve().parents[1]))
LOOKUP_DIR = PROJECT_ROOT / "benchmark" / "lookup"
RESULTS_DIR = PROJECT_ROOT / "results_benchmark"

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

        # GPU monitoring setup
        if self.device == "cuda":
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
            return util.gpu
        return 0.0

    def _init_cpu_monitor_from_gpu(self, device_index: int = 0):
        """
        Initialize NVML and set up CPU util sampling for the PID using GPU memory.
        """
        try:
            handle = nvmlDeviceGetHandleByIndex(device_index)
            gpu_procs = nvmlDeviceGetComputeRunningProcesses(handle)
            pid = gpu_procs[0].pid if gpu_procs else os.getpid()
        except Exception:
            pid = os.getpid()
        self._ps_proc = psutil.Process(pid)
        self._ps_proc.cpu_percent(None)

    def _get_cpu_utilization(self) -> float:
        """
        Return the %CPU for the monitored process since last call.
        """
        if not hasattr(self, "_ps_proc"):
            raise RuntimeError("CPU monitor not initialized.")
        raw_pct = self._ps_proc.cpu_percent(None)
        return raw_pct / psutil.cpu_count()

    def _get_ram_usage(self) -> float:
        """
        Return current RAM (RSS) usage of the monitored process, in MB.
        """
        # memory_info().rss is in bytes
        rss_bytes = self._ps_proc.memory_info().rss
        return round(rss_bytes / (1024 ** 2), 2)  # MB

    def _metrics_monitor(self, readings: dict, stop_evt: threading.Event, sample_interval: float):
        """
        Sample GPU/CPU/RAM metrics until stop_evt is set.
        """
        while not stop_evt.is_set():
            # GPU‐side metrics (same as before)
            readings["memory"].append(self._get_gpu_memory_usage())
            readings["power"].append(self._get_gpu_power_usage())
            readings["util"].append(self._get_gpu_utilization())

            # CPU + RAM metrics (new)
            readings["cpu"].append(self._get_cpu_utilization())
            readings["ram"].append(self._get_ram_usage())

            time.sleep(sample_interval)

    def close(self):
        if self.device == "cuda" and self.handle:
            nvmlShutdown()
            self.handle = None

    def get_model_size(self, path):
        """
        Calculate model size in MB. """
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
                            total_size += os.path.getsize(fp)
                return total_size / (1024 * 1024)
            else:
                raise ValueError("Path must be a .gguf file or a model directory.")
        else:
            return self.model_size

    def measure_energy(self, num_tokens, num_sentences, generation_time, sample_interval=0.1):
        """
        Sample GPU power over generation_time and compute energy metrics.
        """
        readings = []
        start = time.time()
        while time.time() - start < generation_time:
            readings.append(self._get_gpu_power_usage())
            time.sleep(sample_interval)
        avg_power = sum(readings) / len(readings) if readings else 0.0
        total_energy_wh = avg_power * generation_time / 3600
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
            return {"GPU_Utilization (%)": self._get_gpu_utilization()}
        return {}

    def generate(self, prompts):
        return self.iec.completion(prompts)

    @staticmethod
    def estimate_local_query_cost(
        inf_time_sec: float,
        power_watts: float = 0.0,
        electricity_usd_per_kwh: float = 0.31,
        csv_path: Path | str = LOOKUP_DIR / "nvidia_llm_gpus.csv"
    ) -> dict:
        """
        Estimate amortization + energy cost for a single LLM query.
        """
        df = pd.read_csv(csv_path)
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        name, mem = [x.strip() for x in out.split(",")]
        vram = round(math.ceil(float(mem) / 1024))
        subset = df[df["VRAM"].between(vram - 2, vram + 2)]
        if subset.empty:
            raise ValueError(f"No GPU match for VRAM={vram} GB")
        names = subset["GPU"].tolist()
        match = next((n for n in names if n.lower() == name.lower()),
                     next((n for n in names if n.lower().startswith(name.lower())),
                          get_close_matches(name, names, n=1, cutoff=0.4) or [None])[0])
        if not match:
            return None
        amort_hr = subset[subset["GPU"] == match].iloc[0]["Amortization_USD_hr"]
        amort_cost = (inf_time_sec / 3600) * amort_hr
        energy_cost = (power_watts * inf_time_sec / 3.6e6) * electricity_usd_per_kwh if power_watts else 0.0
        return round(amort_cost + energy_cost, 6)

    def _initialize(self, task: str, scenario: str):
        # ── Fast-path: already launched ───────────────────────────
        if getattr(self, "_already_launched", False):
            # clone cached base row to avoid mutating the original
            meta = self._meta_base.copy()
            meta["task"]      = task
            meta["scenario"]  = scenario
            # no cold-start fields change here
            return meta

        # ── First launch: cold start ──────────────────────────────
        t0 = time.time()
        self.iec.launch(backend=self.backend, model=self.model_path)
        startup = time.time() - t0

        # optional warm-up
        self.iec.warmup()
        ttft = self.iec.measure_ttft()

        self.model_size = 0  # TODO: compute real size if needed

        meta = {
            "model_name":   self.model_name,
            "model_size_mb": self.model_size,
            "task":          task,
            "scenario":      scenario,
            "backend":       self.backend,
            "startup":       round(startup, 4),
            "ttft_sec":      round(ttft, 4),
            "coldstart":     round(startup + ttft, 4)
        }

        # ── cache for later calls ─────────────────────────────────
        self._already_launched = True
        self._meta_base = pd.DataFrame([meta])  # store DataFrame copy
        return self._meta_base



    def _batch_generator(
        self,
        prompts: list[str],
        refs:   list[str],
        batch_size: int
    ):
        """
        Yield (batch_prompts, batch_refs) of size `batch_size`.

        - If `batch_size` is None or ≤ 0, returns all items in one batch.
        - Otherwise, splits prompts & refs into chunks of at most `batch_size`.
        """
        if not batch_size or batch_size <= 0:
            batch_size = len(prompts)

        for ps, rs in zip(
            chunker(prompts, batch_size),
            chunker(refs,    batch_size)
        ):
            yield ps, rs



    def _prepare_prompts_per_user(
        self,
        task_obj,
        run_time: float,
        num_users: int,
        requests_per_user_per_min: float
    ):
        """
        Generate per-user arrival times and collect prompts.
        """
        rps_user = requests_per_user_per_min / 60.0
        events = self._generate_per_user_arrivals(num_users, rps_user, run_time)
        n_arrivals = len(events)
        if n_arrivals == 0:
            return [], [], [], []
        # Ensure get_n_prompts_refs returns exactly n_arrivals items
        prompts, refs = task_obj.generate_prompts(n_arrivals)
        sched_ts = [e[0] for e in events]
        user_ids = [e[1] for e in events]
        return prompts, refs, sched_ts, user_ids

    def _generate_per_user_arrivals(
        self,
        num_users: int,
        rps_per_user: float,
        run_time: float
    ):
        """
        Independently sample arrival times per user; merge sorted.
        """
        all_events = []
        for u in range(num_users):
            t = 0.0
            while t <= run_time:
                gap = numpy_rng.exponential(1.0 / rps_per_user)
                t += gap
                if t > run_time:
                    break
                all_events.append((t, u))
        all_events.sort(key=lambda x: x[0])  # expensive for large num_users, consider heap
        return all_events

    def _run_single_request_capture(
        self,
        prompt,
        reference,
        start_wall,
        scheduled_ts,
        user_id
    ):
        """
        Capture raw outputs and timing without computing quality.
        """
        send_time = time.time()
        wait_time = send_time - (start_wall + scheduled_ts)
        t0 = time.time()
        raw_out = self.generate([prompt])[0]
        gen_time = time.time() - t0
        return {
            "user_id": user_id,
            "prompt": prompt,
            "generated_raw": raw_out,
            "reference": reference,
            "send_time": send_time,       # when request was received
            "start_time": t0,             # exact moment generation began
            "generation_time": gen_time,  # duration of generate() call
            "scheduled_ts": scheduled_ts,  # when the request was scheduled
            "wait_time_s": round(wait_time, 6)
        }


    def _run_scenario(
        self,
        task: str,
        scenario: str = "server",
        run_time: float = 600.0,
        concurrent_users: int = 32,
        requests_per_user_per_min: float = 60.0,
        batch_size: Optional[int] = None,
        samples: Optional[int] = None,
        sample_interval: float = 0.1,
        quality_metric: bool = True
    ):
        # 1) instantiate Task
        if task == "summarization":
            from benchmark.tasks.summarization import SummarizationTask
            task_ = SummarizationTask()
        elif task == "qa":
            from benchmark.tasks.qa import QATask
            task_ = QATask()
        elif task == "sql":
            from benchmark.tasks.sql import SQLTask
            task_ = SQLTask()
        elif task == "long_context_qa":
            from benchmark.tasks.long_context import LongContextQATask
            task_ = LongContextQATask()
        else:
            raise ValueError(f"Task {task!r} not supported.")

        # 2) initialize & metadata
        meta_df = self._initialize(task=task, scenario=scenario)

        # 3) set meta_df fields
        if scenario == "server":
            rps_user = requests_per_user_per_min / 60.0
            total_rps = concurrent_users * rps_user
            meta_df = meta_df.assign(
                concurrent_users=concurrent_users,
                requests_per_user_per_min=requests_per_user_per_min,
                requests_per_sec=total_rps,
                run_time_s=run_time
            )
        elif scenario == "batch":
            meta_df = meta_df.assign(
                batch_size=batch_size or 0,
                num_queries= samples or 0
            )
        elif scenario == "long_context":
            meta_df = meta_df.assign(
                num_queries_per_context=samples or 0,
            )
        else:
            meta_df = meta_df.assign(
                num_queries=samples or 0
            )

        # 4) prepare prompts/schedules
        if scenario == "server":
            prompts, refs, sched_ts, user_ids = self._prepare_prompts_per_user(
                task_, run_time, concurrent_users, requests_per_user_per_min
            )
            events = list(zip(sched_ts, prompts, refs, user_ids))
            events.sort(key=lambda x: x[0])
        elif scenario in ["batch", "single"]:
            prompts, refs = task_.generate_prompts(num_examples=samples or 0)
        elif scenario == "long_context":
            prompts, refs, lengths, crs = task_.generate_prompts(num_samples_per_level=samples or 0)


        # 5) start metrics monitor
        global_readings = {"memory": [], "power": [], "util": [], "cpu": [], "ram": []}
        stop_evt = threading.Event()
        mon = threading.Thread(
            target=self._metrics_monitor,
            args=(global_readings, stop_evt, sample_interval),
            daemon=True
        )
        mon.start()

        # 6) setup executor for server
        executor = None
        if scenario == "server":
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users)

        # 7) run inference, capture raw outputs
        wall_start = time.time()
        intermediate_records = []

        if scenario == "server":
            futures = []
            for ts, prompt, reference, uid in events:
                now_elapsed = time.time() - wall_start
                wait = ts - now_elapsed
                if wait > 0:
                    time.sleep(wait)

                futures.append(
                    executor.submit(
                        self._run_single_request_capture,
                        prompt,
                        reference,
                        wall_start,   # still pass wall_start if used in capture
                        ts,
                        uid
                    )
                )

            executor.shutdown(wait=True)
            for fut in futures:
                intermediate_records.append(fut.result())

        elif scenario == "batch":
            for ps, rs in self._batch_generator(prompts, refs, batch_size):
                t0 = time.time()
                raw_outs = self.generate(ps)
                gen_time = time.time() - t0
                for prompt, ref, raw in zip(ps, rs, raw_outs):
                    intermediate_records.append({
                        "prompt": prompt,
                        "generated_raw": raw,
                        "reference": ref,
                        "generation_time": gen_time / len(ps) if ps else 0,
                    })
        
        elif scenario == "long_context":
            for prompt, ref, length, crs in zip(prompts, refs, lengths, crs):
                t0 = time.time()
                try:
                    raw_out = self.generate([prompt])[0]
                    success = True
                except Exception as e:
                    raw_out = str(e)
                    success = False

                gen_time = time.time() - t0

                intermediate_records.append({
                    "context_range": crs,      # e.g. "3k" or "4k"
                    "length": length,         # original context length in tokens
                    "prompt": prompt,
                    "generated_raw": raw_out,
                    "reference": ref,
                    "generation_time": gen_time,
                    "successful": success
                })


        else:  # single or other non-server, non-batch scenario
            for prompt, ref in zip(prompts, refs):
                t0 = time.time()
                raw_out = self.generate([prompt])[0]
                gen_time = time.time() - t0
                intermediate_records.append({
                    "prompt": prompt,
                    "generated_raw": raw_out,
                    "reference": ref,
                    "generation_time": gen_time,
                })

        # 8) stop monitor
        stop_evt.set()
        mon.join()

        # 9) aggregate GPU/CPU/power stats
        if global_readings["memory"]:
            mem_arr = np.array(global_readings["memory"])
            avg_gpu_mem_mb = float(np.mean(mem_arr))
            peak_gpu_mem_mb = float(np.max(mem_arr))
        else:
            avg_gpu_mem_mb = peak_gpu_mem_mb = 0
        overhead_mb = round(max(peak_gpu_mem_mb - getattr(self, "model_size", 0), 0), 2)

        if global_readings["util"]:
            util_arr = np.array(global_readings["util"])
            avg_gpu_util_pct = float(np.mean(util_arr))
            peak_gpu_util_pct = float(np.max(util_arr))
        else:
            avg_gpu_util_pct = peak_gpu_util_pct = 0

        if global_readings["cpu"]:
            cpu_arr = np.array(global_readings["cpu"])
            avg_cpu_util_pct = float(np.mean(cpu_arr))
            peak_cpu_util_pct = float(np.max(cpu_arr))
        else:
            avg_cpu_util_pct = peak_cpu_util_pct = 0

        if global_readings["ram"]:
            ram_arr = np.array(global_readings["ram"])
            avg_ram_mb = float(np.mean(ram_arr))
            peak_ram_mb = float(np.max(ram_arr))
        else:
            avg_ram_mb = peak_ram_mb = 0

        if global_readings["power"]:
            power_arr = np.array(global_readings["power"])
            avg_power_w = float(np.mean(power_arr))
            peak_power_w = float(np.max(power_arr))
            total_energy_wh = float(np.sum(power_arr) * sample_interval / 3600)
        else:
            avg_power_w = peak_power_w = 0
            total_energy_wh = 0.0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        readings_csv_path = (
            RESULTS_DIR
            / "readings"
            / f"ts_{self.backend}_{self.model_name}_{scenario}_{task}_{timestamp}.csv"
        )

        readings_df = pd.DataFrame({
            "gpu_memory_mb": global_readings["memory"],
            "gpu_power_w": global_readings["power"],
            "gpu_util_pct": global_readings["util"],
            "cpu_util_pct": global_readings["cpu"],
            "ram_mb": global_readings["ram"]
        })
        readings_df.to_csv(readings_csv_path, index=False)

        # 10) assemble run_report
        run_report = meta_df.copy().drop(columns=["batch_id"], errors="ignore")

        if scenario == "server":
            wall_end = time.time()
            run_report["total_generation_time_s"] = round(wall_end - wall_start, 6)
        else:
            run_report["total_generation_time_s"] = round(
                sum(r["generation_time"] for r in intermediate_records), 6
            )

        # GPU Memory
        run_report["avg_gpu_mem_mb"] = round(avg_gpu_mem_mb, 2)
        run_report["peak_gpu_mem_mb"] = round(peak_gpu_mem_mb, 2)
        run_report["overhead_mb"] = overhead_mb
        # GPU Utilization
        run_report["avg_gpu_util_pct"] = round(avg_gpu_util_pct, 2)
        run_report["peak_gpu_util_pct"] = round(peak_gpu_util_pct, 2)
        # CPU
        run_report["avg_cpu_util_pct"] = round(avg_cpu_util_pct, 2)
        run_report["peak_cpu_util_pct"] = round(peak_cpu_util_pct, 2)
        # RAM
        run_report["avg_ram_mb"] = round(avg_ram_mb, 2)
        run_report["peak_ram_mb"] = round(peak_ram_mb, 2)
        # Power
        run_report["avg_power_w"] = round(avg_power_w, 2)
        run_report["peak_power_w"] = round(peak_power_w, 2)
        run_report["total_energy_wh"] = total_energy_wh

        # 11) compute quality + finalize counts
        details_list = []

        for rec in intermediate_records:
            generated = clean_prediction([rec["generated_raw"]])[0]
            nt = tok_cnt(generated, self.model_path)     # number of tokens generated
            ns = sent_cnt(generated, mode = scenario)    # number of sentences generated
            gen_time = rec["generation_time"]

            # 1) Average Token Latency (seconds per token)
            ATL = gen_time / nt if nt > 0 else 0.0

            # 1.1) Generate Latency (seconds)
            GL = gen_time

            # 2) Tokens-Per-Second
            TPS = nt / gen_time if gen_time > 0 else 0.0

            # 3) Sentences-Per-Second
            SPS = ns / gen_time if gen_time > 0 else 0.0

            # 4) Total Joules consumed
            joules = avg_power_w * gen_time

            # 5) Energy per token (J/token)
            energy_per_token = joules / nt if nt > 0 else 0.0

            # 6) Energy per sentence (J/sentence)
            energy_per_sentence = joules / ns if ns > 0 else 0.0

            quality = task_.quality_metrics(generated, rec["reference"]) if quality_metric else {}

            # If you’re in “server” mode, also include scheduling info:
            if scenario == "server":
                final_rec = {
                    "user_id":           rec["user_id"],         # user ID for this request
                    "send_time":          rec["send_time"],       # when request was received
                    "start_time":         rec["start_time"],      # when generation started
                    "scheduled_ts":       rec["scheduled_ts"],    # when the request was scheduled
                }
            if scenario == "long_context":
                final_rec = {
                    "context_range":     rec["context_range"],   # e.g. "3k" or "4k"
                    "length":            rec["length"] + 160 + 10, # original context length in tokens + 153 for the prompt + 10 for the question
                    "successful":        rec['successful'],      # whether generation was successful
                }
            else:
                final_rec = {
                }

            add_final_rec = {
                
                "prompt":              rec["prompt"],
                "generated_answer":    generated,
                "reference_answer":    rec["reference"],
                "generation_time":     gen_time,
                "tokens_generated":    nt,
                "sentences_generated": ns,
                "ATL":                 round(ATL, 6),
                "GL" :                 round(GL, 6),
                "TPS":                 round(TPS, 2),
                "SPS":                 round(SPS, 2),
                "energy_per_token":    round(energy_per_token, 6),      # in J/token
                "energy_per_sentence": round(energy_per_sentence, 6),   # in J/sentence
                **quality
            }

            final_rec.update(add_final_rec)
            details_list.append(final_rec)

        details_df = pd.DataFrame(details_list)

        # 12) Compute averages for numeric columns in details_df
        if not details_df.empty:
            detail_nums = details_df.select_dtypes(include="number")
            for col in detail_nums.columns:
                run_report[f"avg_{col}"] = round(detail_nums[col].mean(), 6)

        # 13) Close the inference engine client
        # if self.iec:
        #    self.iec.close()

        # 14) Return the run report and details DataFrame
        return run_report, details_df


    def run(
        self,
        *,
        scenario: str = "server",
        samples = None,
        task: Optional[str] = None,        
        batch_size: Optional[int] = None,
        run_time: Optional[float] = None,
        concurrent_users: int = 32,
        requests_per_user_per_min: float = 60.0,
        sample_interval: float = 0.1,
        quality_metric: bool = True
    ):
        # Validate scenario name
        if scenario not in ("server", "batch", "single", "long_context"):
            raise ValueError(
                f"Unsupported scenario: {scenario!r}. Supported scenarios: 'server', 'batch', 'single', 'long_context."
            )

        # SERVER scenario: require run_time, disallow samples or batch_size
        if scenario == "server":
            if run_time is None or concurrent_users is None or requests_per_user_per_min is None:
                raise ValueError("For 'server' scenario, run_time , concurrent_users, and requests_per_user_per_min must be specified.")
            if samples is not None or batch_size is not None:
                raise ValueError(
                    "For 'server' scenario, do not set 'samples' or 'batch_size'; "
                    "those are only for 'batch' or 'single'."
                )

        # BATCH scenario: require both samples and batch_size, disallow run_time
        elif scenario == "batch":
            if samples is None or batch_size is None:
                raise ValueError("For 'batch' scenario, both 'samples' and 'batch_size' must be specified.")
            if run_time is not None:
                raise ValueError("For 'batch' scenario, do not set 'run_time'; run_time is only for 'server'.")

        # SINGLE scenario: ignore samples and batch_size and run_time; enforce batch_size = 1
        elif scenario == "single":
            batch_size = 1
            run_time = None  # not used in single mode
        
        else:
            # LONG_CONTEXT scenario: samples is the number of queries per context
            if samples is None:
                raise ValueError("For 'long_context' scenario, 'samples' must be specified as queries per context.")
            if batch_size is not None:
                raise ValueError("For 'long_context' scenario, do not set 'batch_size'; it is not applicable.")

        return self._run_scenario(
            task=task,
            scenario=scenario,
            samples=samples,
            batch_size=batch_size,
            run_time=run_time,
            concurrent_users=concurrent_users,
            requests_per_user_per_min=requests_per_user_per_min,
            sample_interval=sample_interval,
            quality_metric=quality_metric
        )
