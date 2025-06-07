#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_engine  —  Launch one of several LLM inference engines via Docker
#
# Usage:
#   ./launch_engine --engine=<engine> --model=<model>
#
#   <engine> ∈ { tgi | vllm | mii | sglang | lmdeploy }
#   <model>  is the Hugging Face model ID (e.g. “mistralai/Mistral-7B-Instruct-v0.3”)
#
# Example:
#   ./launch_engine --engine=tgi --model=mistralai/Mistral-7B-Instruct-v0.3
#
# Notes:
#   • Expects HF_TOKEN or HUGGING_FACE_HUB_TOKEN in your environment.
#   • Always listens on 127.0.0.1:23333 inside the container→host.
#   • Uses $HOME/.cache/huggingface as cache Dir.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ─── Parse arguments “--engine=…” and “--model=…” ────────────────────────────
ENGINE=""
MODEL=""

for ARG in "$@"; do
  case "$ARG" in
    --engine=*)
      ENGINE="${ARG#--engine=}"
      ;;
    --model=*)
      MODEL="${ARG#--model=}"
      ;;
    *)
      echo "Unknown argument: $ARG"
      echo "Usage: $0 --engine=<tgi|vllm|mii|sglang|lmdeploy> --model=<your-org/your-model-name>"
      exit 1
      ;;
  esac
done

if [[ -z "$ENGINE" || -z "$MODEL" ]]; then
  echo "Error: both --engine and --model must be provided."
  echo "Usage: $0 --engine=<tgi|vllm|mii|sglang|lmdeploy> --model=<your-org/your-model-name>"
  exit 1
fi

# ─── Common variables ───────────────────────────────────────────────────────
PORT=23333
CACHE_DIR="$HOME/.cache/huggingface"

# Ensure at least one token is set
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "Error: You must export HF_TOKEN or HUGGING_FACE_HUB_TOKEN in your environment."
  exit 1
fi

# ─── Select and run the requested engine ────────────────────────────────────
case "$ENGINE" in

  tgi)
    # ────────────────────────────────────────────────────────────────────────
    # TGI (Text Generation Inference) container:
    #
    # docker run --rm \
    #   --gpus all \
    #   -v "$HOME/.cache/huggingface:/data" \
    #   -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    #   -e HF_TOKEN="$HF_TOKEN" \
    #   -p 127.0.0.1:23333:23333 \
    #   ghcr.io/huggingface/text-generation-inference:3.3.1 \
    #     --model-id mistralai/Mistral-7B-Instruct-v0.3 \
    #     --trust-remote-code \
    #     --port 23333 \
    #     --max-client-batch-size 128
    # ────────────────────────────────────────────────────────────────────────
    docker run --rm \
      --gpus all \
      -v "$HOME/.cache/huggingface:/data" \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HF_TOKEN="$HF_TOKEN" \
      -p 127.0.0.1:${PORT}:${PORT} \
      ghcr.io/huggingface/text-generation-inference:3.3.1 \
        --model-id "$MODEL" \
        --trust-remote-code \
        --port "$PORT" \
        --max-client-batch-size 512
    ;;

  vllm)
    # ────────────────────────────────────────────────────────────────────────
    # vLLM (OpenAI-compatible) container:
    #
    # docker run --rm \
    #   --runtime=nvidia --gpus all \
    #   -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    #   -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    #   -p 127.0.0.1:23333:23333 \
    #   --ipc=host \
    #   vllm/vllm-openai:latest \
    #     --model mistralai/Mistral-7B-Instruct-v0.3 \
    #     --port 23333
    # ────────────────────────────────────────────────────────────────────────
    docker run --rm \
      --runtime=nvidia --gpus all \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      vllm/vllm-openai:latest \
        --model "$MODEL" \
        --port "$PORT"
    ;;

  lmdeploy)
    # ────────────────────────────────────────────────────────────────────────
    # LMDeploy container:
    #
    # docker run --rm \
    #   --runtime=nvidia --gpus all \
    #   -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    #   -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    #   -p 127.0.0.1:23333:23333 \
    #   --ipc=host \
    #   openmmlab/lmdeploy:latest \
    #     lmdeploy serve api_server mistralai/Mistral-7B-Instruct-v0.3 \
    #     --server-port 23333
    # ────────────────────────────────────────────────────────────────────────
    docker run --rm \
      --runtime=nvidia --gpus all \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      openmmlab/lmdeploy:latest \
        lmdeploy serve api_server "$MODEL" \
        --server-port "$PORT"
    ;;

  sglang)
    # ────────────────────────────────────────────────────────────────────────
    # SGLang (Slim Graph Language) container:
    #
    # docker run --gpus all \
    #   -p 127.0.0.1:23333:23333 \
    #   -v ~/.cache/huggingface:/root/.cache/huggingface \
    #   --ipc=host \
    #   lmsysorg/sglang:latest \
    #   bash -c "\
    #     pip install --no-cache-dir protobuf sentencepiece && \
    #     python3 -m sglang.launch_server \
    #       --model-path mistralai/Mistral-7B-Instruct-v0.3 \
    #       --host 0.0.0.0 \
    #       --port 23333 \
    #   "
    # ────────────────────────────────────────────────────────────────────────
    docker run --rm \
      --gpus all \
      -p 127.0.0.1:${PORT}:${PORT} \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      --ipc=host \
      lmsysorg/sglang:latest \
      bash -c "\
        pip install --no-cache-dir protobuf sentencepiece && \
        python3 -m sglang.launch_server \
          --model-path $MODEL \
          --host 0.0.0.0 \
          --port $PORT \
      "
    ;;

  mii)
    # ────────────────────────────────────────────────────────────────────────
    # DeepSpeed-MII container:
    #
    # docker run --runtime=nvidia --gpus all \
    #   -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    #   -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    #   -p 127.0.0.1:23333:23333 \
    #   --ipc=host \
    #   slinusc/deepspeed-mii:latest \
    #   --model mistralai/Mistral-7B-Instruct-v0.3 \
    #   --port 23333
    # ────────────────────────────────────────────────────────────────────────
    docker run --rm \
      --runtime=nvidia --gpus all \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
      -p 127.0.0.1:${PORT}:${PORT} \
      --ipc=host \
      slinusc/deepspeed-mii:latest \
      --model "$MODEL" \
      --port "$PORT"
    ;;

  *)
    echo "Error: unsupported engine '$ENGINE'."
    echo "Please choose one of: tgi, vllm, lmdeploy, sglang, mii."
    exit 1
    ;;
esac
