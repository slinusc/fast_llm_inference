# Bench360 – Local LLM Inference Benchmark Suite

**Bench360** is a modular benchmarking framework for evaluating **local LLM inference pipelines** across backends, quantization formats, model architectures, and deployment scenarios.

It enables researchers and practitioners to analyze **latency, throughput, quality, efficiency, and cost** in real-world tasks like summarization, QA, and SQL generation—under both consumer and data center conditions.

---

## 🔍 Why Bench360?

When deploying LLMs locally, there’s no one-size-fits-all. Bench360 helps answer:

- **Which model + quant format** yields the best performance for my use case?
- **What’s the latency/throughput trade-off** for vLLM vs. TGI vs. SGLang vs. LMDeploy?
- **How do batch and concurrent scenarios behave under load?**
- **How much GPU memory, power, and time per query do I save with quantization?**
- **Is the quality degradation from INT4 acceptable on e.g. SQL generation?**

---

## ⚙️ Features

| Category            | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **Tasks**           | Summarization, Question Answering (QA), Text-to-SQL                         |
| **Scenarios**       | `single`, `batch`, and `server` (Poisson arrival multi-threads)             |
| **Metrics**         | Latency (ATL/GL), Throughput (TPS, SPS), GPU/CPU util, Energy, Quality (F1, ROUGE, AST) |
| **Backends**        | vLLM, TGI, SGLang, LMDeploy                                                 |
| **Quantization**    | Support for FP16, INT8, INT4 (GPTQ, AWQ, GGUF)                              |
| **Cost Estimation** | Energy and amortized GPU cost per request                                   |
| **Output Format**   | CSV (run-level + per-sample details), logs, and visual plots ready          |

---

## 🧱 Installation

### Requirements

- OS: Ubuntu Linux
- NVIDIA GPU with NVML support
- CUDA 12.x
- Python 3.8+
- Docker

### Setup

Clone the repository:

```bash
git clone https://github.com/slinusc/fast_llm_inference.git
cd fast_llm_inference
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Pull all official backend docker images:

```bash
docker pull lmsysorg/sglang:latest
docker pull openmmlab/lmdeploy:latest
docker pull vllm/vllm-openai:latest
docker pull ghcr.io/huggingface/text-generation-inference:latest
````

> Optional system dependencies:
>
> ```bash
> sudo apt install libssl-dev libcurl4 build-essential libllvm15
> ```

---

## 🚀 Usage

### ✅ Single Run

```yaml
# config.yaml
backend: tgi
hf_model: mistralai/Mistral-7B-Instruct-v0.3
model_name: Mistral-7B
task: qa
scenario: single
samples: 256
```

---

### 🔁 Multi-run Sweep

Use **lists** to define a Cartesian product:

```yaml
backend: [tgi, vllm]
hf_model:
  - mistralai/Mistral-7B-Instruct-v0.3
  - Qwen/Qwen2.5-7B-Instruct
task: [summarization, sql, qa]
scenario: [single, batch, server]

samples: 256
batch_size: [16, 64]
run_time: 300
concurrent_users: [8, 16, 32]
requests_per_user_per_min: 12
```

```bash
python launch_benchmark.py config.yaml
```

---

## 📦 Output

Each experiment generates:

```
results_<timestamp>/
├── run_report/          # One CSV per experiment (summary)
├── details/             # Per-query logs
├── readings/            # GPU/CPU/power metrics
└── failed_runs.log      # List of failed configs
```

Each filename includes:

* backend
* model
* task
* scenario
* parameters (e.g. batch size, concurrent users)
* config hash

This enables reproducible comparisons & tracking.

---

## 🗂 Project Structure

```
fast_llm_inference/
├── benchmark/
│   ├── benchmark.py               # Main benchmarking logic
│   ├── inference_engine_client.py # Backend launcher
│   ├── tasks/                     # Task-specific eval logic
│   ├── backends/                  # Inference wrapper modules
├── launch_benchmark.py            # CLI entry point
├── utils_multi.py                 # Multi-run config handling
├── config.yaml                    # Example config file
└── requirements.txt
```

---

## 🧪 Contributing

Pull requests, bug reports, and ideas are welcome!
Fork the repo, create a feature branch, and submit your PR.

---

## 📄 License

Bench360 is released under the [MIT License](LICENSE).

```
