#!/usr/bin/env python3
"""
launch_benchmark.py – robust single-/multi-run wrapper for ModelBenchmark.

• Lists in YAML  → multi-run (Cartesian product)
• Scalars only   → single-run

Each run writes:
  results_benchmark/run_report/<stem>.csv
  results_benchmark/details/<stem>.csv
Crashes are appended to:
  results_benchmark/failed_runs.log
"""

import os, sys, argparse, warnings, json, hashlib, traceback
from typing import Dict, Any

import yaml
from rich.console import Console
from rich.table   import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TimeElapsedColumn, TimeRemainingColumn
)

from benchmark.utils_multi import load_multi_cfg
from benchmark.benchmark import ModelBenchmark

console = Console(stderr=True)
FAILED_LOG = ""

# ───────────────────────────────────────── dirs & filenames
def _ensure_output_dirs():
    base = "results_benchmark"
    # create sub-dirs
    for sub in ("details", "run_report", "readings"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    # set & clear failure log (module-level)
    global FAILED_LOG
    FAILED_LOG = os.path.join(base, "failed_runs.log")
    open(FAILED_LOG, "w").close()           # truncate file

def _cfg_hash(cfg: Dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

def _param_bundle(cfg: Dict[str, Any]) -> str:
    sc = cfg["scenario"]
    if sc == "batch":
        return f"bs{cfg['batch_size']}"
    if sc == "server":
        return f"cu{cfg['concurrent_users']}_rpm{cfg['requests_per_user_per_min']}"
    return ""                             # single / others

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

def _save_results(report, details, cfg):
    stem = _stem(cfg)
    report.to_csv(f"results_benchmark/run_report/{stem}.csv", index=False)
    details.to_csv(f"results_benchmark/details/{stem}.csv", index=False)
    console.print(f"[green]Saved[/] results_benchmark/run_report/{stem}.csv")

# ───────────────────────────────────────── pretty print helper
def _display_vertical(df):
    tbl = Table(show_header=False, box=None)
    tbl.add_column("Metric", style="bold cyan", no_wrap=True)
    tbl.add_column("Value",  style="white")
    for k, v in df.iloc[0].items():
        tbl.add_row(k, f"{v:.6f}" if isinstance(v, float) else str(v))
    console.print(tbl)

# ───────────────────────────────────────── one run
def _run_one(cfg: Dict[str, Any], bm: ModelBenchmark):
    report, details = bm.run(
        task   = cfg["task"],
        scenario = cfg["scenario"],
        samples  = cfg.get("samples"),
        batch_size = cfg.get("batch_size"),
        run_time   = cfg.get("run_time"),
        concurrent_users        = cfg.get("concurrent_users"),
        requests_per_user_per_min = cfg.get("requests_per_user_per_min"),
        sample_interval = cfg.get("sample_interval", 0.1),
        quality_metric  = cfg.get("quality_metric", True),
    )
    _save_results(report, details, cfg)
    return report

# ───────────────────────────────────────── multi-run
def _run_multi(yaml_path: str, verbose=False):
    runs  = load_multi_cfg(yaml_path)
    total = len(runs)
    current, active_key = None, None

    progress = Progress(
        SpinnerColumn(style="cyan"),
        "[progress.percentage]{task.percentage:>3.0f}%",
        BarColumn(bar_width=None),
        TimeElapsedColumn(), "• ETA", TimeRemainingColumn(compact=True),
        console=console, transient=True,
    )
    task_id = progress.add_task("[cyan]benchmarks", total=total)

    with progress:
        for idx, cfg in enumerate(runs, 1):
            desc = f"{cfg['backend']}/{cfg['model_name']} {cfg['task']}:{cfg['scenario']} {_param_bundle(cfg)}"
            progress.update(task_id, description=desc)

            key = (cfg["backend"], cfg["hf_model"])
            if key != active_key:
                if current:
                    current.iec.close(); current.close()
                current = ModelBenchmark(
                    backend=cfg["backend"],
                    model_name=cfg["model_name"],
                    model_path=cfg["hf_model"],
                    verbose=verbose,
                )
                active_key = key

            try:
                report = _run_one(cfg, current)
                console.print(f"[green]✓[/] {idx}/{total} {desc} "
                              f"→ {report.iloc[0]['total_generation_time_s']:.1f}s")
            except Exception as e:
                console.print(f"[red]✗[/] {idx}/{total} {desc} — {type(e).__name__}: {e}")
                with open(FAILED_LOG, "a") as f:
                    f.write(json.dumps(cfg) + "\n")
                traceback.print_exc(file=sys.stderr)  # optional detail
            finally:
                progress.advance(task_id)

    if current:
        current.iec.close(); current.close()

# ───────────────────────────────────────── entry
def main():
    warnings.filterwarnings("ignore")
    _ensure_output_dirs()

    p = argparse.ArgumentParser(description="Robust benchmark launcher")
    p.add_argument("config", help="YAML file (single or multi)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    is_multi = any(isinstance(v, list) for v in raw.values())

    if is_multi:
        console.print("[cyan]Multi-run mode detected[/]")
        _run_multi(args.config, verbose=args.verbose)
    else:
        bm = ModelBenchmark(
            backend    = raw["backend"],
            model_name = raw.get("model_name", ""),
            model_path = raw["hf_model"],
            verbose    = args.verbose,
        )
        try:
            with console.status("[cyan]running…", spinner="dots"):
                report, details = bm.run(
                    task         = raw["task"],
                    scenario     = raw["scenario"],
                    samples      = raw.get("samples"),
                    batch_size   = raw.get("batch_size"),
                    run_time     = raw.get("run_time"),
                    concurrent_users = raw.get("concurrent_users"),
                    requests_per_user_per_min = raw.get("requests_per_user_per_min"),
                    sample_interval = raw.get("sample_interval", 0.1),
                    quality_metric  = raw.get("quality_metric", True),
                )
            _display_vertical(report)
            _save_results(report, details, raw | {"model_name": raw.get("model_name","")})
        except Exception as e:
            console.print(f"[red]✗  single-run crash — {type(e).__name__}: {e}")
            with open(FAILED_LOG, "a") as f:
                f.write(json.dumps(raw) + "\n")
            traceback.print_exc(file=sys.stderr)
        finally:
            bm.iec.close(); bm.close()

if __name__ == "__main__":
    main()