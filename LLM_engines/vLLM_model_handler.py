from vllm import LLM, SamplingParams


class vLLMHandler:
    def __init__(self, model_name: str, dtype: str = "float16", gpu_util: float = 0.9, 
                 quantization:str = "awq", seed: int = 42, max_model_len: int = 24_576):
        """
        Initialize vLLM with the given model.

        :param model_name: Hugging Face model ID.
        :param dtype: Model precision (float16, bfloat16, float32).
        :param gpu_util: GPU memory fraction to use
        """
        self.model = LLM(model=model_name, 
                         dtype=dtype, 
                         gpu_memory_utilization=gpu_util,
                         quantization=quantization,
                         seed=seed,
                         max_model_len=max_model_len)

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9):
        """
        Generate text using vLLM.

        :param prompt: Input prompt for the model.
        :param max_tokens: Maximum new tokens to generate.
        :param temperature: Sampling temperature.
        :param top_p: Top-p nucleus sampling.
        :return: Generated text.
        """
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        outputs = self.model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
