from benchmark.benchmark import ModelBenchmark
import torch


def run_benchmark(backend, model_name, task, base_path="/home/ubuntu/fast_llm_inference/models", samples=500, verbose=False, batch_size=100):
    print(f"Running benchmark for {model_name} with {backend} on {task}")
    try:
        bm = ModelBenchmark(
            backend=backend,
            model_name=model_name,
            model_path=f"{base_path}/{model_name}",
            task=task,
            verbose=verbose,
        )
        bm.run(samples=samples, batch_size=batch_size)
        bm.close()
        del bm
        torch.cuda.empty_cache()
        print(f"✅ Completed: {model_name} | {backend} | {task}")
    except Exception as e:
        print(f"❌ Failed: {model_name} | {backend} | {task} -- {e}")
        torch.cuda.empty_cache()  # ensure no memory leak on error


base_path = "/home/ubuntu/fast_llm_inference/models"

backends = ["vllm"] #, "huggingface","deepspeed_mii", "llama.cpp"]
models   = [
    "llama-3.1-8B-Instruct",
    "llama-3.2-3b-instruct",
    "llama-3.2-1b-instruct",
    "llama-3.1-8B-Instruct-4bit",
    "llama-3.2-3b-instruct-4bit",
    "llama-3.2-1b-instruct-4bit",
    "llama-3.1-8B-Instruct-8bit",
    "llama-3.2-3b-instruct-8bit",
    "llama-3.2-1b-instruct-8bit",
   
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-7B-Instruct-4bit",
    "Qwen2.5-3B-Instruct-4bit",
    "Qwen2.5-1.5B-Instruct-4bit",
    "Qwen2.5-0.5B-Instruct-4bit",
    "Qwen2.5-7B-Instruct-8bit",
    "Qwen2.5-3B-Instruct-8bit",
    "Qwen2.5-1.5B-Instruct-8bit",
    "Qwen2.5-0.5B-Instruct-8bit",

    "gemma-2-9b-it", 
    "gemma-2-2b-it",
    "gemma-2-9b-it-4bit",
    "gemma-2-9b-it-8bit",
    "gemma-2-2b-it-4bit",
    "gemma-2-2b-it-8bit",
]
tasks    = ["summarization", "qa", "sql",]

for backend in backends:
    for model in models:
        for task in tasks:
            run_benchmark(
                backend=backend,
                model_name=model,
                task=task,
                base_path=base_path,
                samples=500,
                verbose=False,
                batch_size=100,
            )