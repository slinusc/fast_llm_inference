#!/usr/bin/env python3
"""
launch_benchmark.py – resilient single-/multi-run launcher for ModelBenchmark.

• **Lists in YAML  → multi-run** (Cartesian product)
• **Scalars only   → single-run**

Each successful run writes three CSVs under   `results_benchmark_<experiment_name>/…`
    ◦ run_report/<stem>.csv
    ◦ details/<stem>.csv
    ◦ readings/ts_<stem>.csv

Crashes are appended to   `results_benchmark_<experiment_name>/failed_runs.log`  **together with the
full traceback** and will be retried automatically (up to **3 attempts**) the next
time you invoke the launcher. Already‑completed runs are detected by their
hash‑based stem and are silently skipped.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple
from rich.live import Live
from rich.panel import Panel

import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from benchmark.benchmark import ModelBenchmark
from benchmark.utils_multi import load_multi_cfg

# ── Silence noisy logs up‑front ─────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

console = Console(stderr=True)

# ── Globals filled by _ensure_output_dirs() ─────────────────────────────────
FAILED_LOG: str = ""
BASE_DIR: str = ""

# ── Directory helpers ───────────────────────────────────────────────────────

def _ensure_output_dirs(exp_name: str) -> None:
    """Create required sub‑directories *iff* they do not yet exist.
    If the experiment folder already exists we do **not** delete any content –
    this enables incremental / resumed benchmarking.
    """
    global FAILED_LOG, BASE_DIR

    BASE_DIR = f"{exp_name}"
    Path(BASE_DIR).mkdir(exist_ok=True)
    for sub in ("details", "run_report", "readings"):
        Path(BASE_DIR, sub).mkdir(exist_ok=True)

    FAILED_LOG = os.path.join(BASE_DIR, "failed_runs.log")
    # Ensure the file exists so open(..., "a") later never fails.
    Path(FAILED_LOG).touch(exist_ok=True)

# ── Utility: result‑hash / stem / existence check ───────────────────────────

def _run_multi_cfgs(cfgs: List[Dict[str, Any]], verbose: bool = False):
    prior_failures = _load_prior_failures()
    has_prior_failures = len(prior_failures) > 0


    # Combine + de-duplicate
    all_runs = {json.dumps(c, sort_keys=True): c for c in cfgs + prior_failures}
    pending_cfgs = [cfg for cfg in all_runs.values() if not _results_exist(cfg)]

    MAX_ATTEMPTS = 3
    attempt = 1
    unresolved: List[Tuple[Dict[str, Any], str]] = []

    while pending_cfgs and attempt <= MAX_ATTEMPTS:
        # Only show attempt counter if retrying known failed configs
        show_attempts = has_prior_failures or attempt > 1
        failures = _multi_run_cycle(pending_cfgs, verbose, attempt, MAX_ATTEMPTS, show_attempts)
        pending_cfgs = [cfg for cfg in (f[0] for f in failures) if not _results_exist(cfg)]
        unresolved = failures
        attempt += 1

    _rewrite_failed_log(unresolved)
    if unresolved:
        console.print(
            f"[yellow]⚠  {len(unresolved)} runs still failing after {MAX_ATTEMPTS} attempts – see {FAILED_LOG} for details[/]"
        )


def _cfg_hash(cfg: Dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def _param_bundle(cfg: Dict[str, Any]) -> str:
    sc = cfg["scenario"]
    if sc == "batch":
        return f"bs{cfg['batch_size']}"
    if sc == "server":
        return f"cu{cfg['concurrent_users']}_rpm{cfg['requests_per_user_per_min']}"
    return ""


def _stem(cfg: Dict[str, Any]) -> str:
    parts = [
        cfg["backend"],
        cfg["model_name"],
        cfg["task"],
        cfg["scenario"],
    ]
    bundle = _param_bundle(cfg)
    if bundle:
        parts.append(bundle)
    parts.append(_cfg_hash(cfg))
    return "_".join(parts)


def _results_exist(cfg: Dict[str, Any]) -> bool:
    """Return *True* if the main run_report CSV is already present."""
    return Path(BASE_DIR, "run_report", f"{_stem(cfg)}.csv").is_file()

# ── Failure logging helpers ────────────────────────────────────────────────

def _log_failure(cfg: Dict[str, Any], err_trace: str) -> None:
    """Append the *cfg* and *err_trace* JSON‑encoded to FAILED_LOG."""
    with open(FAILED_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"cfg": cfg, "error": err_trace}, sort_keys=True) + "\n")


def _load_prior_failures() -> List[Dict[str, Any]]:
    """Load JSON records from FAILED_LOG. Each line may be either a raw cfg or
    an object with a `cfg` key. We only return the cfg portion.
    """
    runs: List[Dict[str, Any]] = []
    if not Path(FAILED_LOG).is_file():
        return runs
    with open(FAILED_LOG, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "cfg" in obj:
                    runs.append(obj["cfg"])
                else:
                    runs.append(obj)
            except json.JSONDecodeError:
                continue
    return runs


def _rewrite_failed_log(entries: List[Tuple[Dict[str, Any], str]]) -> None:
    """Overwrite FAILED_LOG with *entries* (list of (cfg, error))."""
    with open(FAILED_LOG, "w", encoding="utf-8") as fh:
        for cfg, err in entries:
            fh.write(json.dumps({"cfg": cfg, "error": err}, sort_keys=True) + "\n")


def _render_log(client):
    if not hasattr(client, "_log_buf"):
        return Panel("[dim]‹log tailer not started yet›", title="engine-logs", border_style="grey37", height=12)
    lines = list(client._log_buf)
    body = "\n".join(lines[-10:]) or "[dim]‹no output yet›"
    return Panel(body, title="engine-logs", border_style="grey37", height=12)


# ── Pretty printing helper for single‑run summaries ─────────────────────────

def _display_vertical(df):
    tbl = Table(show_header=False, box=None)
    tbl.add_column("Metric", style="bold cyan", no_wrap=True)
    tbl.add_column("Value", style="white")
    for k, v in df.iloc[0].items():
        tbl.add_row(k, f"{v:.6f}" if isinstance(v, float) else str(v))
    console.print(tbl)

# ── Core: run one benchmark & store results ─────────────────────────────────

def _save_results(report, details, readings, cfg):
    stem = _stem(cfg)
    report.to_csv(Path(BASE_DIR, "run_report", f"{stem}.csv"), index=False)
    details.to_csv(Path(BASE_DIR, "details", f"{stem}.csv"), index=False)
    readings.to_csv(Path(BASE_DIR, "readings", f"ts_{stem}.csv"), index=False)
    console.print(f"[green]Saved[/] {BASE_DIR}/run_report/{stem}.csv")


def _run_one(cfg: Dict[str, Any], bm: ModelBenchmark):
    report, details, readings = bm.run(
        task=cfg["task"],
        scenario=cfg["scenario"],
        samples=cfg.get("samples"),
        batch_size=cfg.get("batch_size"),
        run_time=cfg.get("run_time"),
        concurrent_users=cfg.get("concurrent_users"),
        requests_per_user_per_min=cfg.get("requests_per_user_per_min"),
        sample_interval=cfg.get("sample_interval", 0.1),
        quality_metric=cfg.get("quality_metric", True),
    )
    _save_results(report, details, readings, cfg)
    return report

# ── Multi‑run orchestration with retry logic ────────────────────────────────

def _multi_run_cycle(
    runs: List[Dict[str, Any]],
    verbose: bool,
    attempt: int,
    total_attempts: int,
    show_attempts: bool = True
) -> List[Tuple[Dict[str, Any], str]]:
    """Execute *runs*. Return list of tuples (cfg, error_trace) that failed."""
    if not runs:
        return []

    if show_attempts:
        console.rule(f"Attempt {attempt}/{total_attempts} • {len(runs)} pending")
    else:
        console.rule(f"Running {len(runs)} benchmark(s)…")

    progress = Progress(
        SpinnerColumn(style="cyan"),
        "[progress.percentage]{task.percentage:>3.0f}%",
        BarColumn(bar_width=None),
        TimeElapsedColumn(),
        "• ETA",
        TimeRemainingColumn(compact=True),
        console=console,
        transient=True,
    )
    task_id = progress.add_task("[cyan]benchmarks", total=len(runs))

    failures: List[Tuple[Dict[str, Any], str]] = []
    current: ModelBenchmark | None = None
    active_key = None  # (backend, hf_model)

    with progress:
        for idx, cfg in enumerate(runs, 1):
            desc = f"{cfg['backend']}/{cfg['model_name']} {cfg['task']}:{cfg['scenario']} {_param_bundle(cfg)}"
            progress.update(task_id, description=desc)

            key = (cfg["backend"], cfg["hf_model"])
            if key != active_key:
                if current is not None:
                    current.iec.close()
                    current.close()
                current = ModelBenchmark(
                    backend=cfg["backend"],
                    model_name=cfg["model_name"],
                    model_path=cfg["hf_model"],
                    verbose=verbose,
                )
                active_key = key

            if hasattr(current.iec, "_start_log_tailer"):
                current.iec._start_log_tailer()

            try:
                report = _run_one(cfg, current)
                console.print(
                    f"[green]✓[/] {idx}/{len(runs)} {desc} → {report.iloc[0]['total_generation_time_s']:.1f}s"
                )
            except Exception:
                err_trace = traceback.format_exc()
                console.print(f"[red]✗[/] {idx}/{len(runs)} {desc}\n{err_trace}")
                failures.append((cfg, err_trace))
                _log_failure(cfg, err_trace)

            # Print the latest logs right below the result
            if hasattr(current.iec, "_log_buf"):
                console.print(_render_log(current.iec))

            progress.advance(task_id)

    if current is not None:
        current.iec.close()
        current.close()

    return failures



def _run_multi(yaml_path: str, verbose: bool = False):
    all_cfgs = load_multi_cfg(yaml_path)
    _run_multi_cfgs(all_cfgs, verbose)


def _run_single(yaml_path: str, verbose: bool = False):
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    run_cfg = raw | {"model_name": raw.get("model_name", "")}
    _run_multi_cfgs([run_cfg], verbose)


def main():
    p = argparse.ArgumentParser(description="Run one or more benchmark configs")
    p.add_argument("configs", nargs="+", help="One or more YAML config paths")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    for yaml_path in args.configs:
        try:
            with open(yaml_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            exp_name = raw.get("experiment_name", Path(yaml_path).stem)
            _ensure_output_dirs(exp_name)

            is_multi = any(isinstance(v, list) for v in raw.values())
            if is_multi:
                console.print(f"[cyan]Multi-run mode detected[/] → {yaml_path}")
                _run_multi(yaml_path, verbose=args.verbose)
            else:
                _run_single(yaml_path, verbose=args.verbose)

        except Exception:
            console.print(f"[red]✗ Failed to load or run config:[/] {yaml_path}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
