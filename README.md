# Bench360 — Benchmarking Local LLMs from 360°

A lightweight benchmarking toolkit for evaluating the performance, resource usage, and cost of locally deployed Large Language Models (LLMs) across multiple inference backends (Hugging Face Transformers, vLLM, llama.cpp, DeepSpeed-MII, LMDeploy).

---

## Table of Contents

* [Features](#features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Cloning the Repository](#cloning-the-repository)
* [Configuration (`config.yaml`)](#configuration-configyaml)
* [CLI Usage](#cli-usage)

  * [Basic Run (Quiet Mode)](#basic-run-quiet-mode)
  * [Verbose Mode](#verbose-mode)
* [Output](#output)
* [Directory Structure](#directory-structure)
* [Contributing](#contributing)
* [License](#license)

---

## Features

* **Multi‑backend support**: Hugging Face, vLLM, llama.cpp, DeepSpeed-MII, LMDeploy.
* **Flexible scenarios**: single, batch, and server (Poisson arrival) modes.
* **Resource monitoring**: GPU memory, GPU utilization, CPU usage, and power sampling via NVML.
* **Cost estimation**: amortization + energy cost per query using a GPU lookup table and local power measurements.
* **Rich CLI**: YAML‑driven configuration, quiet/default and verbose modes, progress spinner, and human‑friendly summary.

---

## Prerequisites

* **OS**: Linux (tested on Ubuntu 22.04+)
* **GPU**: NVIDIA with NVML drivers
* **Python**: 3.8 or newer

Install the following system packages (Ubuntu example):

```bash
sudo apt-get update \
    && sudo apt-get install -y python3 python3-venv python3-pip \
       libllvm15 \
       libcurl4 libssl-dev \
       build-essential
```

Ensure your NVIDIA driver and `nvidia-smi` are installed and accessible.

---

## Installation

1. **Clone** this repo (see next section).
2. **Create** and activate a Python virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install** dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If you don’t have a `requirements.txt`, install core packages:

   ```bash
   pip install pandas torch transformers vllm llama-cpp-python mii rich yaml psutil pynvml
   ```

---

## Cloning the Repository

```bash
git clone https://github.com/slinusc/fast_llm_inference.git
cd fast_llm_inference
```

Or, if you prefer SSH:

```bash
git clone git@github.com:slinusc/fast_llm_inference.git
cd fast_llm_inference
```

---

## Configuration (`config.yaml`)

Create a YAML file (e.g. `config.yaml`) with the following structure:

```yaml
# Choose one: huggingface, vllm, llama.cpp, deepspeed_mii, lmdeploy
backend: vllm

# Local path to model directory or .gguf file
model_path: /home/ubuntu/fast_llm_inference/models/llama-3.1-8B-Instruct
model_name: llama-3.1-8B

# Task to benchmark: summarization, qa, or sql
task: summarization

# Scenario: single, batch, or server
scenario: batch

# For batch/single mode
data:
  samples: 100
  batch_size: 4

# For server mode (choose only if scenario: server)
# run_time: 30            # total wall‑clock seconds
# requests_per_sec: 5     # arrival rate (Poisson)
# max_batch_size: 8

# Common options
sample_interval: 0.1       # telemetry sample interval (seconds)
quality_metric: true       # enable quality metrics
output_prefix: results/llama-summ-batch  # prefix for CSV output
```

---

## CLI Usage

We provide a CLI script `launch_benchmark.py`. It reads your YAML config and runs the benchmark.

### Basic Run (Quiet Mode)

Suppresses all internal logs and prints a spinner plus final summary:

```bash
python launch_benchmark.py config.yaml
```

### Verbose Mode

Show all logs, warnings, and detailed prints:

```bash
python launch_benchmark.py config.yaml --verbose
```

---

## Output

After the run completes, you’ll see:

1. **Benchmark Summary** (vertical, human‑readable table) printed to your terminal.
2. **Per‑query CSV** at `<output_prefix>_details.csv` (e.g. `results/llama-summ-batch_details.csv`).

---

## Directory Structure

```
fast_llm_inference/
├── benchmark/              # core benchmarking code
│   ├── benchmark.py        # ModelBenchmark class
│   └── backends/           # wrappers for each inference backend
├── launch_benchmark.py     # CLI wrapper
├── requirements.txt        # Python dependencies
├── config.yaml.sample      # example config file
└── README.md               # this file
```

---

## Contributing

Contributions welcome! Please:

1. Fork and create a branch.
2. Add tests or validate your changes.
3. Submit a pull request.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
