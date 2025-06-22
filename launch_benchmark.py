#!/usr/bin/env python3
"""CLI wrapper for ModelBenchmark.

This version aligns with the updated ``ModelBenchmark.run`` signature that
now supports four scenarios (server, batch, single, long_context) and the
new parameters ``concurrent_users`` and ``requests_per_user_per_min``.

Usage
-----
    benchmark_cli.py config.yaml [-v]

The YAML config **must** include at least these keys:

* backend  – inference backend identifier (e.g. "vllm", "tgi")
* model_path – HF repo, GGUF path, etc. (whatever your backend expects)
* task – benchmark task (e.g. "summarization", "qa")
* scenario – one of "server", "batch", "single", "long_context"

Optional keys (all passed straight to ``ModelBenchmark.run``):

* samples – int, sample count (batch or long_context only)
* batch_size – int, batch size (batch scenario only)
* run_time – float, seconds to run (server only)
* concurrent_users – int, number of simultaneous users (server only)
* requests_per_user_per_min – float, request rate per user (server only)
* sample_interval – float, seconds between profiler samples
* quality_metric – bool, whether to compute quality metrics
* max_batch_size – int, cap for dynamic batching backends

All console output except the rich spinner is suppressed unless ``-v`` is
passed so you can embed the tool in scripts without extra noise.
"""

import os
import sys
import argparse
import yaml
import warnings
import logging
import contextlib
from typing import Any, Dict
from datetime import datetime

# ─── Silence native logs up‑front ────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Further suppression
warnings.filterwarnings("ignore")

from rich.console import Console
from rich.table import Table

from benchmark.benchmark import ModelBenchmark

console = Console(stderr=True)  # write spinner and tables to stderr


# ─── Helpers ─────────────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML file and return the parsed dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_benchmark(cfg: Dict[str, Any], verbose: bool):
    """Create ``ModelBenchmark`` and execute with parameters from *cfg*."""

    bm = ModelBenchmark(
        backend=cfg["backend"],
        model_name=cfg.get("model_name", ""),
        model_path=cfg["model_path"],
        verbose=verbose,
    )

    return bm.run(
        task=cfg.get("task"),
        scenario=cfg["scenario"],
        samples=cfg.get("samples"),
        batch_size=cfg.get("batch_size"),
        run_time=cfg.get("run_time"),
        concurrent_users=cfg.get("concurrent_users", 32),
        requests_per_user_per_min=cfg.get("requests_per_user_per_min", 60.0),
        sample_interval=cfg.get("sample_interval", 0.1),
        quality_metric=cfg.get("quality_metric", True),
    )


# ─── Pretty print helpers ────────────────────────────────────────

def display_vertical(report):
    """Print each metric of the *report* DataFrame on its own line."""
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")

    row = report.iloc[0]
    for metric, val in row.items():
        v = f"{val:.6f}" if isinstance(val, float) else str(val)
        table.add_row(metric, v)

    console.print(table)


# ─── Main entry point ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run ModelBenchmark from a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config",
        help="YAML file with backend, model_path, task, scenario, etc.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show all logs and prints.",
    )
    args = parser.parse_args()

    # Load the configuration file
    try:
        cfg = load_config(args.config)
    except Exception as e:
        sys.exit(f"ERROR loading '{args.config}': {e}")

    # Minimal sanity check for required keys
    for key in ("backend", "model_path", "task", "scenario"):
        if key not in cfg:
            sys.exit(f"ERROR: '{key}' missing in {args.config}")

    # In verbose mode, re‑enable warnings & raise benchmark log level
    if args.verbose:
        warnings.resetwarnings()
        logging.getLogger("benchmark").setLevel(logging.INFO)
        report, details = run_benchmark(cfg, verbose=True)

    else:
        # Suppress stdout but keep stderr (spinner) during the benchmark run
        devnull = open(os.devnull, "w")
        with contextlib.redirect_stdout(devnull):
            with console.status("[bold green]Running benchmark…", spinner="dots"):
                report, details = run_benchmark(cfg, verbose=False)
        devnull.close()

    # ── Summary ────────────────────────────────────────────────
    console.print("\n[bold underline]Benchmark Summary[/]\n")
    display_vertical(report)

    # ── Persist details CSV ────────────────────────────────────
    prefix = f"{cfg['backend']}_{cfg['model_name']}_{cfg['task']}_{cfg['scenario']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_details = f"results_benchmark/details/{prefix}_details.csv"
    out_report = f"results_benchmark/run_report/{prefix}_report.csv"

    report.to_csv(out_report, index=False)
    console.print(f"\nReport → [bold]{out_report}[/]")
    details.to_csv(out_details, index=False)
    console.print(f"\nDetails → [bold]{out_details}[/]")


if __name__ == "__main__":
    main()
