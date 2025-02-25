import os
import time
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import warnings

warnings.filterwarnings("ignore")

class LlamaModelHandler:
    def __init__(self, model_name: str, hf_token_env: str = "HF_TOKEN", precision: str = "fp16"):
        """
        Initialize the LlamaModelHandler class.

        :param model_name: The Hugging Face model identifier.
        :param hf_token_env: The environment variable where the Hugging Face token is stored.
        :param precision: Desired model precision for loading ('fp16', 'fp32', 'int8').
        """
        self.model_name = model_name
        self.hf_token = os.environ.get(hf_token_env)
        if not self.hf_token:
            raise ValueError(f"Hugging Face token not found in environment variable '{hf_token_env}'")

        self.tokenizer = None
        self.model = None
        self.precision = precision
        self.model_load_time = None

        # Automatically authenticate and load the model upon initialization
        self.authenticate()
        self.load_model()

    def authenticate(self):
        """
        Authenticate to Hugging Face Hub using the provided token.
        """
        login(token=self.hf_token)
        print("Authentication successful.")

    def load_model(self):
        """
        Load the tokenizer and model from Hugging Face Hub with the specified precision.
        Logs the model loading time for benchmark consistency.
        """
        print(f"Loading model '{self.model_name}' with precision '{self.precision}'...")
        start_time = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=True)

        # Determine torch dtype based on precision
        if self.precision == "fp16":
            dtype = torch.float16
        elif self.precision == "fp32":
            dtype = torch.float32
        elif self.precision == "int8":
            dtype = None  # Will be handled by load_in_8bit argument
        else:
            raise ValueError("Unsupported precision specified. Use 'fp16', 'fp32', or 'int8'.")

        # Load the model with the specified precision
        if self.precision == "int8":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=True,
                device_map="auto",
                use_auth_token=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto",
                use_auth_token=True
            )

        end_time = time.time()
        self.model_load_time = round(end_time - start_time, 4)
        print(f"Model loaded on device: {self.model.device}")
        print(f"GPU: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "Running on CPU.")
        print(f"Model dtype: {self.model.dtype}")
        print(f"Model loading time: {self.model_load_time} seconds")

    def get_model_and_tokenizer(self):
        """
        Returns the loaded model and tokenizer for external use (e.g., benchmarking),
        along with the model loading time.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before access.")
        return self.model, self.tokenizer

    def generate_text(self, prompt: str, max_new_tokens: int = 250) -> str:
        """
        Generate text based on the given prompt with controlled repetition and creativity.

        :param prompt: Input text prompt for the model.
        :param max_new_tokens: Maximum number of tokens to generate.
        :return: Generated text as a string.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generating text.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            early_stopping=True,
            no_repeat_ngram_size=3,           # Avoid repeating 3-grams
            repetition_penalty=1.2,           # Penalize repetitive phrases
            temperature=0.7,                  # Controls randomness (lower = more focused)
            top_p=0.9,                        # Nucleus sampling for coherent responses
            do_sample=True,                   # Enable sampling for diversity
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Example usage for benchmarking:
if __name__ == "__main__":
    # Example: load model in int8 precision for benchmarking
    model_handler = LlamaModelHandler("meta-llama/Llama-3.1-8b", precision="int8")
    model, tokenizer, load_time = model_handler.get_model_and_tokenizer()

    print(f"Model loading time for benchmarking: {load_time} seconds")

    # Example prompt for benchmarking
    prompt_text = "Tell me about the key features of LLaMA 3.1 8B."
    generated_text = model_handler.generate_text(prompt=prompt_text)
    print("\nGenerated Text:\n", generated_text)
