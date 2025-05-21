def get_backend(name, **kwargs):
    if name == "huggingface":
        from benchmark.backends.hf_backend import HuggingFaceBackend
        return HuggingFaceBackend(**kwargs)
    elif name == "vllm":
        from benchmark.backends.vllm_backend import VLLMBackend
        return VLLMBackend(**kwargs)
    elif name == "llama.cpp":
        from benchmark.backends.llamacpp_backend import LlamaCppBackend
        return LlamaCppBackend(**kwargs)
    elif name == "deepspeed_mii":
        from benchmark.backends.deepspeed_mii_backend import MIIDeepSpeedBackend
        return MIIDeepSpeedBackend(**kwargs)
    elif name == "lmdeploy":
        from benchmark.backends.lmdeploy_backend import LMDeployBackend
        return LMDeployBackend(**kwargs)
    else:
        raise ValueError(f"Unsupported backend: {name}")