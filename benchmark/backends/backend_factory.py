# backend_factory.py
import importlib
import vllm # don't remove this import, it is needed to register vLLM backend

_BACKENDS = {
    "huggingface": ("benchmark.backends.hf_backend",     "HuggingFaceBackend"),
    "vllm":         ("benchmark.backends.vllm_backend",   "VLLMBackend"),
    "llama.cpp":    ("benchmark.backends.llamacpp_backend","LlamaCppBackend"),
    "mii":("benchmark.backends.deepspeed_mii_backend","MIIDeepSpeedBackend"),
    "lmdeploy":     ("benchmark.backends.lmdeploy_backend","LMDeployBackend"),
    "tgi":          ("benchmark.backends.tgi_backend",     "TGIBackend"),
}

def get_backend(name: str, **kwargs):
    try:
        module_path, class_name = _BACKENDS[name]
    except KeyError:
        raise ValueError(f"Unsupported backend: {name!r}")

    module = importlib.import_module(module_path)
    backend_cls = getattr(module, class_name)
    return backend_cls(**kwargs)
