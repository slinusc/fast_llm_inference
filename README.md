# Bench360 — Benchmarking Local LLMs from 360°

Bench360 is a modular benchmarking suite for local LLM inference. It offers a full-stack, extensible pipeline to evaluate the latency, throughput, quality, and cost of large language model inference on consumer and enterprise GPUs. Bench360 supports flexible backends, tasks, and scenarios, enabling fair and reproducible comparisons for researchers and practitioners.

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

* **Multi-backend support**: Hugging Face, vLLM, llama.cpp, DeepSpeed-MII, LMDeploy.
* **Flexible scenarios**: single, batch, and server (Poisson arrival) modes.
* **Supported tasks**: summarization, question answering (QA), and SQL generation.
* **Quality metrics**: Task specific metrics like ROUGE, F1 or AST.
* **Custom tasks**: easily add new tasks by implementing a Task class in `benchmark/tasks`.
* **Latency**: measures average generation latency (e.g., ATL and GL metrics).
* **Throughput**: measures tokens per second (TPS) and sentences per second (SPS).
* **Resource monitoring**: GPU memory, GPU utilization, CPU usage, and power sampling via NVML.
* **Cost estimation**: amortization + energy cost per query.

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
---

## Cloning the Repository

```bash
git clone https://github.com/slinusc/fast_llm_inference.git
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
│   ├── backends/           # wrappers for each inference backend
│   ├── lookup/             # look up tables
│   └── tasks/              # task definitions (summarization, QA, SQL)
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
