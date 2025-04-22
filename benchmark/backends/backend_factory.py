from benchmark.backends.hf_backend import HuggingFaceBackend
from benchmark.backends.vllm_backend import VLLMBackend
from benchmark.backends.llamacpp_backend import LlamaCppBackend
from benchmark.backends.lmdeploy_backend import LMDeployBackend

def get_backend(name, **kwargs):
    if name == "huggingface":
        return HuggingFaceBackend(**kwargs)
    elif name == "vllm":
        return VLLMBackend(**kwargs)
    elif name == "llama.cpp":
        return LlamaCppBackend(**kwargs)
    elif name == "lmdeploy":
        return LMDeployBackend(**kwargs)
    else:
        raise ValueError(f"Unsupported backend: {name}")