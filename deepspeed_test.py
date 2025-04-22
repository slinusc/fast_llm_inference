from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Your pre-quantized 4bit or 8bit model path
model_path = "/home/ubuntu/fast_llm_inference/llama-3.1-8B-Instruct-quantizised/llama-3.1-8B-4bit"

# Load model directly (DeepSpeed will handle backend if launched properly)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto"  # DeepSpeed + Accelerate handles this internally
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Inference pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# Run generation
prompt = "What is the theory of relativity?"
outputs = pipe(prompt, max_new_tokens=100)

print("🧠 Output:\n", outputs[0]['generated_text'])