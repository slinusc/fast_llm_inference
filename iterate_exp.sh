#!/bin/bash

MODELS=("mistralai/Mistral-7B-Instruct-v0.3" "Qwen/Qwen2.5-7B-Instruct")
ENGINES=("tgi" "vllm" "sglang" "lmdeploy")
TASKS=("qa" "sql" "summarization")

BATCH_SIZES=(8 16 32 64 128)
CONCURRENT_USERS=(4 8 16 32 64)
REQUESTS_PER_USER=(10)

mkdir -p configs logs results results/details results/run_report
: > failed_runs.log  # clear or create failure log

for MODEL in "${MODELS[@]}"; do
  MODEL_NAME=$(basename "$MODEL")

  for ENGINE in "${ENGINES[@]}"; do

    for TASK in "${TASKS[@]}"; do
      
      ## ───── SINGLE SCENARIO ─────
      CONFIG_ID="${ENGINE}_${MODEL_NAME}_${TASK}_single"
      CONFIG_PATH="configs/${CONFIG_ID}.yaml"
      cat > "$CONFIG_PATH" <<EOF
backend: $ENGINE
model_path: $MODEL
model_name: $MODEL_NAME
scenario: single
task: $TASK
samples: 512
sample_interval: 0.1
quality_metric: true
EOF
      echo "▶ Running: $CONFIG_ID"
      if ! python3 launch_benchmark.py "$CONFIG_PATH" > logs/${CONFIG_ID}.log 2>&1; then
        echo "$CONFIG_ID" >> failed_runs.log
      fi

      ## ───── BATCH SCENARIOS ─────
      for BS in "${BATCH_SIZES[@]}"; do
        CONFIG_ID="${ENGINE}_${MODEL_NAME}_${TASK}_batch_bs${BS}"
        CONFIG_PATH="configs/${CONFIG_ID}.yaml"
        cat > "$CONFIG_PATH" <<EOF
backend: $ENGINE
model_path: $MODEL
model_name: $MODEL_NAME
scenario: batch
task: $TASK
samples: 512
batch_size: $BS
sample_interval: 0.1
quality_metric: true
EOF
        echo "▶ Running: $CONFIG_ID"
        if ! python3 launch_benchmark.py "$CONFIG_PATH" > logs/${CONFIG_ID}.log 2>&1; then
          echo "$CONFIG_ID" >> failed_runs.log
        fi
      done

      ## ───── SERVER SCENARIOS ─────
      for CU in "${CONCURRENT_USERS[@]}"; do
        for RPM in "${REQUESTS_PER_USER[@]}"; do
          CONFIG_ID="${ENGINE}_${MODEL_NAME}_${TASK}_server_cu${CU}_rpm${RPM}"
          CONFIG_PATH="configs/${CONFIG_ID}.yaml"
          cat > "$CONFIG_PATH" <<EOF
backend: $ENGINE
model_path: $MODEL
model_name: $MODEL_NAME
scenario: server
task: $TASK
run_time: 300
concurrent_users: $CU
requests_per_user_per_min: $RPM
sample_interval: 0.1
quality_metric: true
EOF
          echo "▶ Running: $CONFIG_ID"
          if ! python3 launch_benchmark.py "$CONFIG_PATH" > logs/${CONFIG_ID}.log 2>&1; then
            echo "$CONFIG_ID" >> failed_runs.log
          fi
        done
      done

    done
  done
done

echo "All runs complete. Failed ones are in failed_runs.log"
