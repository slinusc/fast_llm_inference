#!/usr/bin/env python3
import os
# ─── Silence native logs up-front ───────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse
import yaml
import warnings
import logging
import contextlib

# Further suppression
warnings.filterwarnings("ignore")

from rich.console import Console
from rich.table   import Table

from benchmark.benchmark import ModelBenchmark

console = Console(stderr=True)  # write spinner and tables to stderr

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_benchmark(cfg, verbose):
    bm = ModelBenchmark(
        backend    = cfg["backend"],
        model_name = cfg.get("model_name", ""),
        model_path = cfg["model_path"],
        verbose    = verbose,
    )
    return bm.run(
        task             = cfg["task"],
        scenario         = cfg["scenario"],
        samples          = cfg.get("samples"),
        batch_size       = cfg.get("batch_size"),
        run_time         = cfg.get("run_time"),
        requests_per_sec = cfg.get("requests_per_sec"),
        max_batch_size   = cfg.get("max_batch_size"),
        sample_interval  = cfg.get("sample_interval", 0.1),
        quality_metric   = cfg.get("quality_metric", True),
    )

def display_vertical(report: Table):
    """Print each metric on its own line."""
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")

    row = report.iloc[0]
    for metric, val in row.items():
        if isinstance(val, float):
            v = f"{val:.6f}"
        else:
            v = str(val)
        table.add_row(metric, v)

    console.print(table)

def main():
    p = argparse.ArgumentParser(
        description="Run ModelBenchmark from a YAML config."
    )
    p.add_argument("config", help="YAML file with backend, model_path, task, scenario, etc.")
    p.add_argument("-v", "--verbose", action="store_true", help="Show all logs and prints.")
    args = p.parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as e:
        sys.exit(f"ERROR loading '{args.config}': {e}")

    for k in ("backend", "model_path", "task", "scenario"):
        if k not in cfg:
            sys.exit(f"ERROR: '{k}' missing in {args.config}")

    # In verbose mode, re-enable warnings & logs
    if args.verbose:
        warnings.resetwarnings()
        logging.getLogger("benchmark").setLevel(logging.INFO)

        report, details = run_benchmark(cfg, verbose=True)
    else:
        # suppress only stdout (the benchmark prints) but keep stderr for spinner
        devnull = open(os.devnull, "w")
        with contextlib.redirect_stdout(devnull):
            with console.status("[bold green]Running benchmark…", spinner="dots"):
                report, details = run_benchmark(cfg, verbose=False)
        devnull.close()

    # print summary vertically
    console.print("\n[bold underline]Benchmark Summary[/]\n")
    display_vertical(report)

    # write out full details
    prefix  = cfg.get("output_prefix", f"{cfg['task']}_{cfg['scenario']}")
    out_csv = f"{prefix}_details.csv"
    details.to_csv(out_csv, index=False)
    console.print(f"\nDetails → [bold]{out_csv}[/]")

if __name__ == "__main__":
    main()
